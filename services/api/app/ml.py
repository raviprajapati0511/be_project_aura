from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from torch import nn

from .charging import charging_context, charging_observation, constrain_charging_action
from .can_ids import attack_probability, frames_from_payload
from .config import MODEL_DIR
from .reasoning import cross_domain_reasoning


FEATURES = [
    "pack_voltage", "pack_current", "soc_reported", "battery_soh",
    "cell_temp_max", "motor_temp", "motor_vibration", "power_consumption",
    "speed_kmh", "component_health", "acceleration_mps2",
]
SEQ_LEN = 20


class ConvBiLSTM(nn.Module):
    """CNN feature extractor followed by temporal BiLSTM state/range heads."""

    def __init__(self, input_dim: int = len(FEATURES), channels: int = 32, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(x.transpose(1, 2)).transpose(1, 2)
        temporal, _ = self.lstm(encoded)
        return self.head(temporal[:, -1, :])


def kalman_smooth(values: np.ndarray, process_var: float = 0.02, measurement_var: float = 0.15) -> np.ndarray:
    """Small scalar Kalman filter applied independently to each sensor channel."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    result = np.empty_like(values)
    estimate = values[0].copy()
    covariance = np.ones(values.shape[1], dtype=np.float32)
    for index, measurement in enumerate(values):
        covariance += process_var
        gain = covariance / (covariance + measurement_var)
        estimate += gain * (measurement - estimate)
        covariance *= 1.0 - gain
        result[index] = estimate
    return result


@dataclass
class InferenceResult:
    anomaly: bool
    anomaly_score: float
    estimated_soc_pct: float
    estimated_range_km: float
    advisory_action: str
    reasoning_status: str
    reasoning_summary: str
    reasoning: dict[str, Any]
    signals: list[dict[str, str]]
    failure_probability: float
    remaining_useful_life: float | None
    model_status: str
    charging: dict[str, Any]
    inference_latency_ms: float


def _row_vector(row: dict[str, Any], defaults: dict[str, float]) -> np.ndarray:
    return np.asarray([
        float(row.get(key) if row.get(key) is not None else defaults.get(key, 0.0))
        for key in FEATURES
    ], dtype=np.float32)


class AuraModels:
    """Modular telemetry inference using the paper's CNN/BiLSTM, IF, and DQN modules."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        health_path = model_dir / "health_models.joblib"
        legacy_path = model_dir / "ev_health_models.joblib"
        self.bundle = joblib.load(health_path) if health_path.exists() else (
            joblib.load(legacy_path) if legacy_path.exists() else None
        )
        self.sequence_norm = joblib.load(model_dir / "sequence_norm.joblib") if (model_dir / "sequence_norm.joblib").exists() else None
        self.sequence_model = ConvBiLSTM()
        sequence_path = model_dir / "cnn_bilstm.pt"
        self.sequence_model.load_state_dict(torch.load(sequence_path, map_location="cpu", weights_only=True)) if sequence_path.exists() else None
        self.sequence_ready = sequence_path.exists()
        self.sequence_model.eval()
        charging_path = model_dir / "charging_dqn.zip"
        self.charging_model = DQN.load(charging_path, device="cpu") if charging_path.exists() else None
        charging_forecaster_path = model_dir / "charging_forecaster.joblib"
        self.charging_forecaster = joblib.load(charging_forecaster_path) if charging_forecaster_path.exists() else None
        can_path = model_dir / "can_intrusion.joblib"
        self.can_detector = joblib.load(can_path) if can_path.exists() else None

    def predict_can(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        if self.can_detector is None:
            raise RuntimeError("CAN intrusion model is not trained; run npm run setup")
        features = frames_from_payload(frames)
        probabilities = self.can_detector.predict_proba(features)
        classes = [str(label) for label in self.can_detector.classes_]
        windows = []
        for scores in probabilities:
            class_scores = {name: float(score) for name, score in zip(classes, scores)}
            predicted = max(class_scores, key=class_scores.get)
            windows.append({
                "class": predicted,
                "intrusion_score": round(attack_probability(class_scores), 4),
                "probabilities": {name: round(score, 4) for name, score in class_scores.items()},
            })
        mean_scores = {
            name: float(np.mean([window["probabilities"][name] for window in windows]))
            for name in classes
        }
        return {
            "window_frames": 100,
            "windows_scored": len(windows),
            "predicted_class": max(mean_scores, key=mean_scores.get),
            "intrusion_score": round(attack_probability(mean_scores), 4),
            "mean_class_probabilities": {name: round(score, 4) for name, score in mean_scores.items()},
            "windows": windows,
            "confidence_note": "RandomForest class probabilities are uncalibrated research scores, not a security guarantee.",
        }

    def forecast_charging_session(self, features: dict[str, Any]) -> dict[str, float]:
        if self.charging_forecaster is None:
            raise RuntimeError("Charging-session model is not trained; run npm run setup")
        numeric = self.charging_forecaster["numeric_features"]
        categorical = self.charging_forecaster["categorical_features"]
        row = {**features}
        for key in numeric:
            row.setdefault(key, 0.0)
        for key in categorical:
            row.setdefault(key, "unknown")
        prediction = self.charging_forecaster["model"].predict(pd.DataFrame([row], columns=numeric + categorical))
        return {
            "estimated_energy_kwh": round(max(0.0, float(prediction[0][0])), 2),
            "estimated_duration_hours": round(max(0.0, float(prediction[0][1])), 2),
        }

    def predict(self, rows: list[dict[str, Any]]) -> InferenceResult:
        from time import perf_counter

        started = perf_counter()
        latest = rows[-1]
        defaults = self.bundle.get("defaults", {}) if self.bundle else {}
        raw = np.stack([_row_vector(row, defaults) for row in rows[-SEQ_LEN:]])
        while len(raw) < SEQ_LEN:
            raw = np.concatenate((raw[:1], raw), axis=0)
        filtered = raw
        anomaly = False
        anomaly_score = 0.0
        failure = self._fallback_failure(latest)
        rul = None

        if self.bundle:
            vector = _row_vector(latest, defaults).reshape(1, -1)
            anomaly_scaler = self.bundle.get("anomaly_scaler", self.bundle.get("scaler"))
            anomaly_vector = anomaly_scaler.transform(vector) if anomaly_scaler else vector
            anomaly_model = self.bundle["anomaly"]
            anomaly_score = float(anomaly_model.decision_function(anomaly_vector)[0])
            anomaly = bool(anomaly_model.predict(anomaly_vector)[0] == -1)
            if "failure" in self.bundle:
                failure = float(self.bundle["failure"].predict_proba(vector)[0][1])
            if "rul" in self.bundle:
                rul = float(self.bundle["rul"].predict(vector)[0])

        soc = float(latest.get("soc_reported") if latest.get("soc_reported") is not None else 0.0)
        soh = float(latest.get("battery_soh") if latest.get("battery_soh") is not None else 1.0)
        speed = float(latest.get("speed_kmh") if latest.get("speed_kmh") is not None else 0.0)
        power = max(float(latest.get("power_consumption") or 20.0), 5.0)
        estimated_soc = soc
        estimated_range = min(650.0, max(0.0, soc / 100.0 * soh * 75.0 / power * max(speed, 30.0)))
        if self.sequence_ready and self.sequence_norm:
            mean = self.sequence_norm["mean"]
            scale = self.sequence_norm["scale"]
            sequence = kalman_smooth(((filtered - mean) / scale).astype(np.float32))
            with torch.inference_mode():
                output = self.sequence_model(torch.from_numpy(sequence[None, :, :]))[0].numpy()
            estimated_soc = float(np.clip(output[0], 0.0, 1.0) * 100.0)
            estimated_range = float(np.clip(output[1] * 350.0, 0.0, 650.0))

        reasoning = cross_domain_reasoning(latest, anomaly, failure)
        context = charging_context(soc=estimated_soc, soh=soh)
        observation = charging_observation(context)
        proposed_action_index = 0
        if self.charging_model:
            proposed_action_index, _ = self.charging_model.predict(observation, deterministic=True)
        else:
            proposed_action_index = 0 if context["soc"] >= context["target_soc"] or context["grid_load"] >= 0.9 else (2 if context["price"] <= 0.18 and context["renewable"] >= 0.35 else 1)
        constrained_action = constrain_charging_action(
            int(proposed_action_index), observation,
            anomaly=anomaly or reasoning["label"] == "component_fault",
            cell_temp_c=float(latest.get("cell_temp_max") or 0.0),
            failure_probability=failure,
            security_restrict=reasoning["label"] == "suspected_attack",
        )
        action_index = constrained_action
        charge_actions = ["defer", "charge at 3.3 kW", "charge at 7.2 kW"]
        charging = {
            **context,
            "recommended_action": charge_actions[action_index],
            "power_kw": [0.0, 3.3, 7.2][action_index],
            "proposed_action": charge_actions[int(proposed_action_index)],
            "safety_restricted": action_index != int(proposed_action_index),
            "controller": "DQN + safety shield" if self.charging_model else "rule-based fallback",
            "execution": "advisory_only",
        }

        signals = self._signals(latest, anomaly, failure)
        if reasoning["label"] == "suspected_attack":
            signals.insert(0, {"code": "suspected_attack_review", "severity": "critical", "domain": "security", "message": "Independent CAN and telemetry anomalies match a suspected-attack pattern; restrict charging and escalate for human review."})
        elif reasoning["label"] == "unresolved_security_signal":
            signals.insert(0, {"code": "can_signal_needs_review", "severity": "warning", "domain": "security", "message": "The upstream CAN detector reported an anomaly without corroborating telemetry evidence; escalate for operator review."})
        elif reasoning["label"] == "component_fault" and not any(signal["domain"] == "maintenance" for signal in signals):
            signals.insert(0, {"code": "power_motion_mismatch", "severity": "warning", "domain": "cross-domain", "message": "Measured battery power differs from the motion-based proxy; inspect the vehicle and sensor channels."})
        action = charge_actions[action_index]
        summary = signals[0]["message"] if signals else "Battery, motor, charging, and telemetry-consistency checks are within configured bounds."
        latency = (perf_counter() - started) * 1000.0
        return InferenceResult(
            anomaly=anomaly,
            anomaly_score=round(anomaly_score, 4),
            estimated_soc_pct=round(estimated_soc, 2),
            estimated_range_km=round(estimated_range, 1),
            advisory_action=action,
            reasoning_status="investigate" if signals else "nominal",
            reasoning_summary=summary,
            reasoning=reasoning,
            signals=signals,
            failure_probability=round(failure, 4),
            remaining_useful_life=round(float(rul), 1) if rul is not None else None,
            model_status="full" if self.sequence_ready and self.bundle and self.charging_model and self.charging_forecaster and self.can_detector else "partial",
            charging=charging,
            inference_latency_ms=round(latency, 3),
        )

    @staticmethod
    def _fallback_failure(row: dict[str, Any]) -> float:
        health = float(row.get("component_health") if row.get("component_health") is not None else 1.0)
        soh = float(row.get("battery_soh") if row.get("battery_soh") is not None else 1.0)
        temp = float(row.get("cell_temp_max") or 0.0)
        motor_temp = float(row.get("motor_temp") or 0.0)
        return min(1.0, max(0.0, 0.35 * (1 - health) + 0.3 * (1 - soh) + 0.2 * max(temp - 35, 0) / 30 + 0.15 * max(motor_temp - 65, 0) / 45))

    @staticmethod
    def _signals(row: dict[str, Any], anomaly: bool, failure: float) -> list[dict[str, str]]:
        signals: list[dict[str, str]] = []
        temp = float(row.get("cell_temp_max") or 0.0)
        motor_temp = float(row.get("motor_temp") or 0.0)
        health = float(row.get("component_health") if row.get("component_health") is not None else 1.0)
        speed = float(row.get("speed_kmh") or 0.0)
        current = float(row.get("pack_current") or 0.0)
        if anomaly:
            signals.append({"code": "multivariate_deviation", "severity": "warning", "domain": "telemetry", "message": "Isolation Forest detected a deviation from the learned operating profile."})
        if speed < 3.0 and abs(current) > 45.0:
            signals.append({"code": "electrical_motion_mismatch", "severity": "warning", "domain": "cross-domain", "message": "Battery current is high while vehicle speed is near zero; check charging state, sensors, and telemetry timing."})
        if failure >= 0.65:
            signals.append({"code": "predicted_failure_risk", "severity": "critical", "domain": "maintenance", "message": f"The trained model estimates elevated component failure risk ({failure:.0%})."})
        elif failure >= 0.35:
            signals.append({"code": "elevated_failure_risk", "severity": "warning", "domain": "maintenance", "message": f"The model estimates increased component failure risk ({failure:.0%})."})
        if temp >= 48:
            signals.append({"code": "battery_thermal_stress", "severity": "critical" if temp >= 55 else "warning", "domain": "battery", "message": f"Battery temperature is elevated at {temp:.1f} °C."})
        if motor_temp >= 85:
            signals.append({"code": "motor_thermal_stress", "severity": "warning", "domain": "motor", "message": f"Motor temperature is elevated at {motor_temp:.1f} °C."})
        if health < 0.5:
            signals.append({"code": "component_health", "severity": "warning", "domain": "maintenance", "message": f"Component health score is low ({health:.0%})."})
        return signals

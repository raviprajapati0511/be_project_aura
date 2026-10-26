"""Train and evaluate the modular models proposed in the AURA-EV paper."""

from __future__ import annotations

import json
import random
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.charging import EVChargingEnv, constrain_charging_action  # noqa: E402
from app.can_ids import CAN_CLASSES, CAN_FEATURES, load_can_capture  # noqa: E402
from app.config import MODEL_DIR  # noqa: E402
from app.ml import AuraModels, ConvBiLSTM, FEATURES, SEQ_LEN, kalman_smooth  # noqa: E402

DATA_ARCHIVE = ROOT / "services" / "api" / "data" / "eviot-predictivemaint-dataset.zip"
DATASET_CSV = "EV_Predictive_Maintenance_Dataset_15min.csv"
CHARGING_CSV = ROOT / "services" / "api" / "data" / "charging-patterns" / "ev_charging_patterns.csv"
CAN_DIR = ROOT / "services" / "api" / "data" / "car-hacking"
MAX_ROWS = 25000


def load_dataset() -> pd.DataFrame:
    if not DATA_ARCHIVE.exists():
        raise FileNotFoundError(f"Kaggle dataset archive missing: {DATA_ARCHIVE}")
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        frame = pd.read_csv(archive.open(DATASET_CSV))
    frame = frame.rename(columns={
        "Battery_Voltage": "pack_voltage", "Battery_Current": "pack_current",
        "SoC": "soc_fraction", "SoH": "battery_soh",
        "Battery_Temperature": "cell_temp_max", "Motor_Temperature": "motor_temp",
        "Motor_Vibration": "motor_vibration", "Power_Consumption": "power_consumption",
        "Driving_Speed": "speed_kmh", "Component_Health_Score": "component_health",
        "Failure_Probability": "failure_label", "RUL": "rul_score",
    })
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    elapsed_seconds = frame["Timestamp"].diff().dt.total_seconds().replace(0, np.nan)
    frame["acceleration_mps2"] = (frame["speed_kmh"].diff() / 3.6 / elapsed_seconds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frame) > MAX_ROWS:
        # Even spacing keeps the chronology while covering the full source period.
        frame = frame.iloc[np.linspace(0, len(frame) - 1, MAX_ROWS, dtype=int)].reset_index(drop=True)
    frame["soc_reported"] = frame["soc_fraction"] * 100.0
    return frame


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
    }


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "false_positive_rate": round(float(((predicted == 1) & (actual == 0)).sum() / max((actual == 0).sum(), 1)), 4),
    }


def train_sequence(frame: pd.DataFrame, raw: np.ndarray, defaults: dict[str, float]) -> dict:
    start = time.perf_counter()
    split_row = int(len(frame) * 0.8)
    mean = raw[:split_row].mean(axis=0).astype(np.float32)
    scale = np.maximum(raw[:split_row].std(axis=0), 1e-5).astype(np.float32)
    # Causal Kalman filter: each output depends only on current and prior data.
    normalized = kalman_smooth(((raw - mean) / scale).astype(np.float32))
    # The Kaggle source has no measured range target. This documented proxy is
    # an energy / speed estimate, not an independently observed range label.
    soc_target = frame["soc_fraction"].to_numpy(dtype=np.float32)
    speed = frame["speed_kmh"].to_numpy(dtype=np.float32)
    power = np.maximum(frame["power_consumption"].to_numpy(dtype=np.float32), 5.0)
    soh = frame["battery_soh"].to_numpy(dtype=np.float32)
    range_proxy = np.clip(soc_target * soh * 75.0 / power * np.maximum(speed, 30.0), 0.0, 650.0).astype(np.float32)
    train_sequences = np.lib.stride_tricks.sliding_window_view(
        normalized[:split_row], SEQ_LEN, axis=0,
    ).transpose(0, 2, 1).copy()
    # Test windows may use the prior 19 training observations as live history,
    # but no test target is used for fitting or model selection.
    test_history = normalized[split_row - SEQ_LEN + 1:]
    test_sequences = np.lib.stride_tricks.sliding_window_view(
        test_history, SEQ_LEN, axis=0,
    ).transpose(0, 2, 1).copy()
    test_indices = np.arange(split_row, len(frame))
    if len(test_sequences) != len(test_indices):
        raise ValueError("Chronological sequence split did not align with test targets")
    # Simulated sensor noise makes SoC estimation a real denoising task instead
    # of copying the dataset's SoC input column directly to the output.
    rng = np.random.default_rng(19)
    soc_index = FEATURES.index("soc_reported")
    train_sequences[:, :, soc_index] += rng.normal(0.0, 0.07, size=train_sequences[:, :, soc_index].shape).astype(np.float32)
    test_sequences[:, :, soc_index] += rng.normal(0.0, 0.07, size=test_sequences[:, :, soc_index].shape).astype(np.float32)
    train_indices = np.arange(SEQ_LEN - 1, split_row)
    tx = torch.from_numpy(train_sequences)
    ty = torch.from_numpy(np.column_stack([soc_target[train_indices], range_proxy[train_indices] / 350.0]).astype(np.float32))
    vx = torch.from_numpy(test_sequences)
    vy = np.column_stack([soc_target[test_indices], range_proxy[test_indices]])
    loader = DataLoader(TensorDataset(tx, ty), batch_size=512, shuffle=True)
    model = ConvBiLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()
    for epoch in range(5):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"[train] CNN-BiLSTM epoch {epoch + 1} complete")
    model.eval()
    with torch.inference_mode():
        predicted = model(vx).numpy()
    # Baseline is the noisy sensor value at the final step of each test window.
    noisy_soc = np.clip(test_sequences[:, -1, soc_index] * scale[soc_index] + mean[soc_index], 0, 1)
    return {
        "model": model,
        "normalization": {"mean": mean, "scale": scale},
        "metrics": {
            "soc": regression_metrics(vy[:, 0] * 100.0, np.clip(predicted[:, 0], 0, 1) * 100.0),
            "soc_noisy_sensor_baseline": regression_metrics(vy[:, 0] * 100.0, noisy_soc * 100.0),
            "range_proxy": regression_metrics(vy[:, 1], np.clip(predicted[:, 1] * 350.0, 0, 650.0)),
            "range_target_note": "Derived from SoC × SoH × 75 kWh and reported power/speed; source has no measured range label.",
            "train_windows": int(len(ty)),
            "test_windows": int(len(vy)),
            "split_note": "Chronological 80/20 split by raw record; no test target used for training or early stopping. Test windows reuse only the preceding training history.",
            "training_seconds": round(time.perf_counter() - start, 3),
        },
    }


def train_charging_policy() -> tuple[DQN, dict]:
    env = EVChargingEnv(seed=17)
    model = DQN(
        "MlpPolicy", env, learning_rate=0.0007, buffer_size=6000,
        learning_starts=500, batch_size=64, gamma=0.96, exploration_fraction=0.35,
        verbose=0, seed=17, device="cpu",
    )
    start = time.perf_counter()
    model.learn(total_timesteps=10000, progress_bar=False)
    policy_eval = {"cost": [], "peak_load": [], "constraint_violations": [], "target_met": []}
    raw_eval = {"cost": [], "target_met": []}
    baseline_eval = {"cost": [], "peak_load": [], "constraint_violations": [], "target_met": []}
    eval_env = EVChargingEnv(seed=97)
    for episode in range(100):
        state, _ = eval_env.reset(seed=episode + 500)
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, terminated, truncated, raw_info = eval_env.step(int(action))
            done = terminated or truncated
        raw_eval["cost"].append(raw_info["cost"])
        raw_eval["target_met"].append(raw_info["target_met"])
        # Reset the same episode seed and evaluate the deployed constrained
        # policy independently from the raw DQN outcome.
        state, _ = eval_env.reset(seed=episode + 500)
        done = False
        while not done:
            proposed, _ = model.predict(state, deterministic=True)
            action = constrain_charging_action(int(proposed), state)
            state, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        for key in policy_eval:
            policy_eval[key].append(info[key])
        state, _ = eval_env.reset(seed=episode + 500)
        done = False
        while not done:
            soc, price, load, renewable, _hour, target = state
            action = 0 if soc >= target else (2 if price < 0.42 and load < 0.75 and renewable > 0.2 else 1)
            state, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        for key in baseline_eval:
            baseline_eval[key].append(info[key])
    metrics = {
        "episodes": 100,
        "dqn_mean_cost": round(float(np.mean(policy_eval["cost"])), 4),
        "raw_dqn_mean_cost": round(float(np.mean(raw_eval["cost"])), 4),
        "raw_dqn_target_completion_rate": round(float(np.mean(raw_eval["target_met"])), 4),
        "rule_baseline_mean_cost": round(float(np.mean(baseline_eval["cost"])), 4),
        "dqn_mean_peak_load": round(float(np.mean(policy_eval["peak_load"])), 4),
        "rule_baseline_mean_peak_load": round(float(np.mean(baseline_eval["peak_load"])), 4),
        "dqn_mean_constraint_violations": round(float(np.mean(policy_eval["constraint_violations"])), 4),
        "rule_baseline_mean_constraint_violations": round(float(np.mean(baseline_eval["constraint_violations"])), 4),
        "dqn_target_completion_rate": round(float(np.mean(policy_eval["target_met"])), 4),
        "rule_baseline_target_completion_rate": round(float(np.mean(baseline_eval["target_met"])), 4),
        "training_seconds": round(time.perf_counter() - start, 3),
        "environment_note": "Reported DQN metrics include a target/peak safety shield; raw DQN results are reported separately. Synthetic tariffs/grid; not measured utility or station data.",
    }
    return model, metrics


def train_charging_forecasters() -> tuple[object, dict]:
    """Train session energy/duration regressors on an independent Kaggle source."""
    if not CHARGING_CSV.exists():
        raise FileNotFoundError(f"Kaggle charging dataset missing: {CHARGING_CSV}; run `npm run data` first")
    frame = pd.read_csv(CHARGING_CSV)
    frame["Charging Start Time"] = pd.to_datetime(frame["Charging Start Time"], errors="coerce")
    numeric = [
        "Battery Capacity (kWh)", "Charging Rate (kW)", "State of Charge (Start %)",
        "State of Charge (End %)", "Temperature (°C)", "Vehicle Age (years)",
        "Distance Driven (since last charge) (km)",
    ]
    categorical = ["Vehicle Model", "Charger Type", "User Type"]
    targets = ["Energy Consumed (kWh)", "Charging Duration (hours)"]
    frame = frame.dropna(subset=["Charging Start Time", *numeric, *categorical, *targets]).sort_values("Charging Start Time")
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = make_pipeline(transformer, RandomForestRegressor(
        n_estimators=160, min_samples_leaf=2, max_features=0.9, random_state=23, n_jobs=2,
    ))
    model.fit(train[numeric + categorical], train[targets])
    predicted = model.predict(test[numeric + categorical])
    metrics = {}
    for index, target in enumerate(targets):
        metrics[target] = {
            **regression_metrics(test[target].to_numpy(), predicted[:, index]),
            "r2": round(float(r2_score(test[target], predicted[:, index])), 4),
        }
    metrics.update({
        "dataset": "Electric Vehicle Charging Patterns, Kaggle, Apache-2.0",
        "rows": int(len(frame)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split": "Chronological 80/20 holdout by charging start time.",
        "features": numeric + categorical,
        "targets": targets,
        "source_note": "Kaggle session dataset; forecasts are correlational estimates and not charger commands.",
    })
    return {
        "model": model,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "targets": targets,
    }, metrics


def train_can_intrusion_model() -> tuple[object, dict]:
    """Train on window features with per-capture chronological, purged splits."""
    captures = [
        ("normal_run_data.txt", "benign"),
        ("DoS_dataset.csv", "dos"),
        ("Fuzzy_dataset.csv", "fuzzy"),
        ("RPM_dataset.csv", "rpm_spoofing"),
        ("gear_dataset.csv", "gear_spoofing"),
    ]
    train_x, train_y, test_x, test_y = [], [], [], []
    capture_summary = {}
    for filename, class_name in captures:
        path = CAN_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Kaggle CAN capture missing: {path}; run `npm run data` first")
        stat = path.stat()
        cache = CAN_DIR / f".{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.windows.joblib"
        if cache.exists():
            features, labels = joblib.load(cache)
        else:
            features, labels = load_can_capture(path, class_name)
            joblib.dump((features, labels), cache, compress=1)
        # Attack activity is localized in HCRL recordings. Split each class
        # in temporal order so rare attack windows have a genuine later holdout.
        capture_train, capture_test = [], []
        class_splits = {}
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            train_end = int(len(indices) * 0.70)
            test_start = int(len(indices) * 0.85)
            if train_end < 1 or len(indices) - test_start < 1:
                raise ValueError(f"Not enough {label} windows to split CAN capture {filename}")
            capture_train.extend(indices[:train_end])
            capture_test.extend(indices[test_start:])
            class_splits[str(label)] = {
                "windows": int(len(indices)),
                "train_windows": int(train_end),
                "validation_gap_windows": int(test_start - train_end),
                "test_windows": int(len(indices) - test_start),
            }
        capture_train.sort()
        capture_test.sort()
        train_x.append(features[capture_train])
        train_y.append(labels[capture_train])
        test_x.append(features[capture_test])
        test_y.append(labels[capture_test])
        capture_summary[filename] = {
            "windows": int(len(features)),
            "train_windows": int(len(capture_train)),
            "validation_gap_windows": int(sum(part["validation_gap_windows"] for part in class_splits.values())),
            "test_windows": int(len(capture_test)),
            "class_splits": class_splits,
            "window_labels": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    x_train, y_train = np.concatenate(train_x), np.concatenate(train_y)
    x_test, y_test = np.concatenate(test_x), np.concatenate(test_y)
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError("CAN capture partition lacks multiple classes; inspect the source Flag labels")
    model = RandomForestClassifier(
        n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample",
        max_features=0.9, random_state=31, n_jobs=2,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    labels = list(model.classes_)
    positive_actual = y_test != "benign"
    positive_predicted = predicted != "benign"
    report = {
        "dataset": "HCRL Car-Hacking dataset, Kaggle mirror; original HCRL capture labels",
        "source_url": "https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset",
        "original_source": "https://sites.google.com/hksecurity.net/hcrl/Datasets/car-hacking-dataset",
        "split": "Per-capture, per-class chronological 70/15/15; first 70% of each class train, middle 15% unused validation gap, last 15% test. This preserves later examples for rare, temporally localized attacks but is not a continuous future-stream or independent-vehicle test.",
        "window_frames": 100,
        "features": CAN_FEATURES,
        "classes": labels,
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "macro_f1": round(float(f1_score(y_test, predicted, labels=labels, average="macro", zero_division=0)), 4),
        "per_class_f1": {name: round(float(score), 4) for name, score in zip(labels, f1_score(y_test, predicted, labels=labels, average=None, zero_division=0))},
        "binary_intrusion_precision": round(float(precision_score(positive_actual, positive_predicted, zero_division=0)), 4),
        "binary_intrusion_recall": round(float(recall_score(positive_actual, positive_predicted, zero_division=0)), 4),
        "binary_intrusion_f1": round(float(f1_score(positive_actual, positive_predicted, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "capture_splits": capture_summary,
        "label_note": "Injected Flag=T messages identify attack windows; all frames in a flagged window are aggregated, so a windowed warning can precede or lag a precise injected frame.",
        "calibration_note": "RandomForest probabilities are not calibrated confidence values; evaluate on independent vehicles/sessions before operational use.",
    }
    return model, report


def benchmark_inference(frame: pd.DataFrame) -> dict[str, float | str]:
    rows = frame.tail(SEQ_LEN)[FEATURES].to_dict(orient="records")
    model = AuraModels(MODEL_DIR)
    for _ in range(5):
        model.predict(rows)
    timings = []
    process = psutil.Process()
    process.cpu_percent(interval=None)
    for _ in range(50):
        started = time.perf_counter()
        model.predict(rows)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "requests": len(timings),
        "latency_mean_ms": round(float(np.mean(timings)), 3),
        "latency_p50_ms": round(float(np.percentile(timings, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(timings, 95)), 3),
        "process_memory_mb_after_benchmark": round(process.memory_info().rss / (1024 ** 2), 2),
        "process_cpu_percent_during_benchmark": round(float(process.cpu_percent(interval=None)), 2),
        "hardware_note": "Local CPU benchmark; results are machine-specific and are not edge-device measurements.",
    }


def main() -> None:
    torch.set_num_threads(2)
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    start_all = time.perf_counter()
    frame = load_dataset()
    defaults = {key: float(frame[key].median()) for key in FEATURES}
    raw = frame[FEATURES].fillna(defaults).to_numpy(dtype=np.float32)
    split_row = int(len(frame) * 0.8)
    x_train, x_test = raw[:split_row], raw[split_row:]
    anomaly_scaler = StandardScaler().fit(x_train)
    train_scaled = anomaly_scaler.transform(x_train)
    test_scaled = anomaly_scaler.transform(x_test)
    anomaly = IsolationForest(n_estimators=100, contamination=0.04, random_state=42, n_jobs=2)
    anomaly.fit(train_scaled)
    y_train_failure = frame["failure_label"].iloc[:split_row].astype(int).to_numpy()
    y_test_failure = frame["failure_label"].iloc[split_row:].astype(int).to_numpy()
    # Failure/RUL are auxiliary supervised maintenance heads. The evaluation
    # report uses a chronological holdout and exposes dataset-label limitations.
    failure = RandomForestClassifier(n_estimators=40, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=2)
    failure.fit(x_train, y_train_failure)
    rul = RandomForestRegressor(n_estimators=40, min_samples_leaf=4, random_state=42, n_jobs=2)
    rul.fit(x_train, frame["rul_score"].iloc[:split_row].to_numpy(dtype=np.float32))
    test_anomaly = (anomaly.predict(test_scaled) == -1).astype(int)
    test_failure = failure.predict(x_test)
    threshold_anomaly = (
        (frame["cell_temp_max"].iloc[split_row:].to_numpy() >= 48)
        | (frame["component_health"].iloc[split_row:].to_numpy() < 0.5)
        | (frame["motor_temp"].iloc[split_row:].to_numpy() >= 85)
    ).astype(int)
    fused_anomaly = np.maximum(test_anomaly, threshold_anomaly)
    report = {
        "dataset": "EVIoT-PredictiveMaint Dataset, version 1; evenly sampled chronological records",
        "split": "Chronological 80/20 holdout; no random row leakage across the time split.",
        "records": len(frame),
        "source_records": 175393,
        "sampled_records": len(frame),
        "modules": {
            "soc_range_cnn_bilstm": None,
            "isolation_forest_vs_failure_label_proxy": classification_metrics(y_test_failure, test_anomaly),
            "threshold_baseline_vs_failure_label_proxy": classification_metrics(y_test_failure, threshold_anomaly),
            "cross_domain_fusion_ablation_vs_failure_label_proxy": classification_metrics(y_test_failure, fused_anomaly),
            "failure_classifier": classification_metrics(y_test_failure, test_failure),
            "rul_regressor": regression_metrics(frame["rul_score"].iloc[split_row:].to_numpy(dtype=np.float32), rul.predict(x_test)),
            "anomaly_evaluation_note": "Failure label is only a proxy for anomaly ground truth; source contains no labeled cyberattack or fault-event taxonomy.",
        },
        "datasets": {
            "ev_telemetry": {"name": "EVIoT-PredictiveMaint", "kaggle": "datasetengineer/eviot-predictivemaint-dataset", "records": len(frame), "tasks": ["SoC", "range proxy", "anomaly proxy", "failure", "RUL"]},
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURES, "defaults": defaults, "anomaly_scaler": anomaly_scaler, "anomaly": anomaly, "failure": failure, "rul": rul}, MODEL_DIR / "health_models.joblib", compress=3)
    sequence = train_sequence(frame, raw, defaults)
    torch.save(sequence["model"].state_dict(), MODEL_DIR / "cnn_bilstm.pt")
    joblib.dump(sequence["normalization"], MODEL_DIR / "sequence_norm.joblib", compress=3)
    report["modules"]["soc_range_cnn_bilstm"] = sequence["metrics"]
    charging_model, charging_metrics = train_charging_policy()
    charging_model.save(MODEL_DIR / "charging_dqn.zip")
    report["modules"]["smart_charging_dqn"] = charging_metrics
    charging_forecaster, charging_forecast_metrics = train_charging_forecasters()
    joblib.dump(charging_forecaster, MODEL_DIR / "charging_forecaster.joblib", compress=3)
    report["modules"]["charging_session_forecaster"] = charging_forecast_metrics
    report["datasets"]["charging_sessions"] = {"name": "Electric Vehicle Charging Patterns", "kaggle": "valakhorasani/electric-vehicle-charging-patterns", "license": "Apache-2.0", "rows": charging_forecast_metrics["rows"], "tasks": ["energy consumed", "session duration"]}
    can_model, can_metrics = train_can_intrusion_model()
    joblib.dump(can_model, MODEL_DIR / "can_intrusion.joblib", compress=3)
    report["modules"]["can_intrusion_detection"] = can_metrics
    report["datasets"]["can_security"] = {"name": "HCRL Car-Hacking dataset", "kaggle": "pranavjha24/car-hacking-dataset", "license": "MIT Kaggle mirror; cite and honor original HCRL research terms", "tasks": ["benign/attack", "DoS", "fuzzy", "RPM spoofing", "gear spoofing"]}
    report["deployment"] = benchmark_inference(frame)
    report["deployment"]["training_total_seconds"] = round(time.perf_counter() - start_all, 3)
    report["deployment"]["device"] = "CPU"
    report["limitations"] = [
        "Kaggle source values are used as published and are not independently validated against vehicle hardware.",
        "Range target is a derived proxy, not ground-truth vehicle range.",
        "The DQN charging policy is evaluated in the synthetic Gymnasium environment; the separate charging-session regressor is evaluated on Kaggle session rows.",
        "CAN windows are labeled from HCRL injection flags and randomly generated/vehicle-independent deployments are not represented; the temporal holdout is within the same capture files.",
        "Cross-modal co-simulation, real BMS/charger control, independent-vehicle testing, and embedded-device benchmarks remain future validation work.",
    ]
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[api] trained AURA modules from {len(frame):,} records -> {MODEL_DIR}")
    print(f"[api] evaluation report -> {MODEL_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def cross_domain_reasoning(
    row: dict[str, Any], telemetry_anomaly: bool, failure_probability: float
) -> dict[str, Any]:
    """Explain physical/telemetry evidence without treating it as a cyber verdict.

    CAN evidence is accepted only when an upstream CAN detector supplies a score.
    The bundled EV dataset has no CAN frames, so absence is represented explicitly.
    The power residual is a transparent proxy, not a calibrated vehicle energy model.
    """
    voltage = _number(row, "pack_voltage")
    current = _number(row, "pack_current")
    measured_power_kw = voltage * current / 1000.0
    speed_ms = max(0.0, _number(row, "speed_kmh") / 3.6)
    acceleration = _number(row, "acceleration_mps2")
    # Lightweight rolling/aerodynamic/acceleration/auxiliary load proxy.
    predicted_power_kw = max(0.25, 0.18 * speed_ms + 0.0009 * speed_ms**3 + 0.6 + 0.08 * acceleration * speed_ms)
    # Scale is a conservative demo spread. It must be calibrated on normal vehicle data.
    residual_sigma_kw = 2.5
    residual_z = (measured_power_kw - predicted_power_kw) / residual_sigma_kw
    physical_deviation = abs(residual_z) >= 2.0

    can_score_value = row.get("can_anomaly_score")
    can_score = None if can_score_value is None else max(0.0, min(1.0, _number(row, "can_anomaly_score")))
    can_available = can_score is not None
    can_anomaly = can_available and can_score >= 0.7
    telemetry_high = telemetry_anomaly

    if telemetry_high and can_anomaly and not physical_deviation:
        label = "suspected_attack"
        action = "restrict_and_escalate"
        confidence = min(0.9, 0.55 + 0.2 * can_score)
        explanation = "Telemetry and independently supplied CAN evidence are abnormal while the power residual is small. Treat as a suspected attack and review; this is not an automated cyber verdict."
    elif physical_deviation and telemetry_high and not can_anomaly:
        label = "component_fault"
        action = "derate_and_alert"
        confidence = 0.66
        explanation = "Measured electrical power differs from the motion-based proxy and telemetry is anomalous; inspect the vehicle and sensor channels."
    elif telemetry_high and not physical_deviation and not can_anomaly:
        label = "sensor_or_communication_fault" if can_available else "unresolved_telemetry_anomaly"
        action = "flag_and_review"
        confidence = 0.58 if can_available else 0.4
        explanation = "Telemetry is anomalous without a large power residual. CAN evidence is unavailable." if not can_available else "Telemetry is anomalous while the power residual and CAN score remain low; inspect sensor or communication quality."
    elif physical_deviation or failure_probability >= 0.65:
        label = "component_fault"
        action = "derate_and_alert"
        confidence = 0.58
        explanation = "Physical power mismatch or elevated maintenance risk warrants an inspection."
    elif can_anomaly:
        label = "unresolved_security_signal"
        action = "escalate_to_operator"
        confidence = 0.45
        explanation = "The supplied CAN detector reports an anomaly without corroborating telemetry evidence; escalate for review."
    elif not can_available:
        label = "insufficient_security_evidence"
        action = "none"
        confidence = 0.0
        explanation = "No CAN/network detector is connected; the system makes no attack assessment."
    else:
        label = "benign"
        action = "none"
        confidence = 0.75
        explanation = "No configured cross-domain evidence pattern is active."

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "recommended_action": action,
        "explanation": explanation,
        "evidence": {
            "measured_power_kw": round(measured_power_kw, 3),
            "motion_power_proxy_kw": round(predicted_power_kw, 3),
            "power_residual_z": round(residual_z, 3),
            "telemetry_anomaly": bool(telemetry_high),
            "failure_probability": round(float(failure_probability), 4),
            "can_detector_available": can_available,
            "can_anomaly_score": round(can_score, 4) if can_score is not None else None,
        },
        "limitations": "Research decision support only. The power model and residual threshold are uncalibrated; CAN input must come from an independently validated detector.",
    }

from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .ml import AuraModels
from .config import MODEL_DIR


app = FastAPI(title="AURA ML Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = AuraModels()


def encode_row(row: dict[str, Any]) -> dict[str, Any]:
    encoded = dict(row)
    ts = encoded.get("time")
    if isinstance(ts, datetime):
        encoded["time"] = ts.isoformat()
    return encoded


@app.get("/health")
def health():
    db_ready = True
    try:
        db.fetch_vehicle_ids()
    except Exception:
        db_ready = False
    modules = {
        "kalman": True,
        "cnn_bilstm": models.sequence_ready,
        "isolation_forest": bool(models.bundle),
        "dqn_charging": models.charging_model is not None,
        "cross_domain_reasoning": True,
        "can_intrusion_detector": models.can_detector is not None,
        "charging_session_forecaster": models.charging_forecaster is not None,
    }
    return {
        "status": "ok" if db_ready and all(modules.values()) else "degraded",
        "db": db_ready,
        "modules": modules,
    }


@app.get("/vehicles")
def vehicles():
    return {"vehicles": db.fetch_vehicle_ids()}


@app.get("/telemetry/latest")
def telemetry_latest(vehicle_id: str | None = None):
    row = db.fetch_latest(vehicle_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no telemetry found")
    return encode_row(row)


@app.get("/telemetry/recent")
def telemetry_recent(
    vehicle_id: str | None = None,
    limit: int = Query(default=120, ge=1, le=1000),
):
    return {"items": [encode_row(row) for row in db.fetch_recent(vehicle_id, limit)]}


@app.get("/inference")
def inference(
    vehicle_id: str | None = None,
    limit: int = Query(default=80, ge=20, le=250),
):
    rows = db.fetch_recent(vehicle_id, limit)
    if not rows:
        raise HTTPException(status_code=404, detail="no telemetry found")
    result = models.predict(rows)
    latest = encode_row(rows[-1])
    return {
        "vehicle_id": latest["vehicle_id"],
        "time": latest["time"],
        "latest": latest,
        "anomaly": result.anomaly,
        "anomaly_score": result.anomaly_score,
        "estimated_range_km": result.estimated_range_km,
        "estimated_soc_pct": result.estimated_soc_pct,
        "failure_probability": result.failure_probability,
        "remaining_useful_life": result.remaining_useful_life,
        "model_status": result.model_status,
        "charging": result.charging,
        "inference_latency_ms": result.inference_latency_ms,
        "advisory_action": result.advisory_action,
        "reasoning_status": result.reasoning_status,
        "reasoning_summary": result.reasoning_summary,
        "reasoning": result.reasoning,
        "signals": result.signals,
    }


@app.get("/fleet/summary")
def fleet_summary():
    latest_rows = db.fetch_fleet_latest()
    vehicles = []
    for row in latest_rows:
        rows = db.fetch_recent(row["vehicle_id"], limit=40)
        if not rows:
            continue
        result = models.predict(rows)
        vehicle = encode_row(row)
        vehicles.append({
            **vehicle,
            "estimated_soc_pct": result.estimated_soc_pct,
            "estimated_range_km": result.estimated_range_km,
            "failure_probability": result.failure_probability,
            "anomaly": result.anomaly,
            "reasoning_status": result.reasoning_status,
            "signals": result.signals,
        })
    return {
        "vehicles": vehicles,
        "vehicle_count": len(vehicles),
        "at_risk_count": sum(v["reasoning_status"] == "investigate" for v in vehicles),
        "mean_soc_pct": round(sum(float(v.get("soc_reported") or 0) for v in vehicles) / len(vehicles), 1) if vehicles else None,
        "mean_battery_health_pct": round(sum(float(v.get("battery_soh") or 0) * 100 for v in vehicles) / len(vehicles), 1) if vehicles else None,
    }


@app.get("/research/evaluation")
def research_evaluation():
    path = MODEL_DIR / "evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model evaluation report not found; run the model training script")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/charging/forecast")
def charging_forecast(
    battery_capacity_kwh: float = Query(default=75.0, gt=0, le=250),
    charging_rate_kw: float = Query(default=7.2, gt=0, le=500),
    soc_start_pct: float = Query(default=30.0, ge=0, le=100),
    soc_target_pct: float = Query(default=80.0, ge=0, le=100),
    temperature_c: float = Query(default=25.0, ge=-50, le=100),
    vehicle_age_years: float = Query(default=2.0, ge=0, le=100),
    distance_since_charge_km: float = Query(default=100.0, ge=0, le=2000),
    vehicle_model: str = Query(default="unknown", max_length=80),
    charger_type: str = Query(default="unknown", max_length=80),
    user_type: str = Query(default="unknown", max_length=80),
):
    if soc_target_pct < soc_start_pct:
        raise HTTPException(status_code=422, detail="soc_target_pct must be at least soc_start_pct")
    try:
        result = models.forecast_charging_session({
            "Battery Capacity (kWh)": battery_capacity_kwh,
            "Charging Rate (kW)": charging_rate_kw,
            "State of Charge (Start %)": soc_start_pct,
            "State of Charge (End %)": soc_target_pct,
            "Temperature (°C)": temperature_c,
            "Vehicle Age (years)": vehicle_age_years,
            "Distance Driven (since last charge) (km)": distance_since_charge_km,
            "Vehicle Model": vehicle_model,
            "Charger Type": charger_type,
            "User Type": user_type,
        })
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "forecast": result,
        "execution": "advisory_only",
        "dataset": "Kaggle Electric Vehicle Charging Patterns",
        "confidence_note": "Chronological holdout performance is weak on this dataset; treat these values as illustrative estimates.",
    }


@app.post("/security/can/inference")
def can_inference(payload: dict[str, Any] = Body(...)):
    frames = payload.get("frames")
    if not isinstance(frames, list) or not 100 <= len(frames) <= 5000:
        raise HTTPException(status_code=422, detail="frames must contain between 100 and 5000 CAN records")
    if models.can_detector is None:
        raise HTTPException(status_code=503, detail="CAN intrusion model is not trained; run `npm run setup`")
    try:
        result = models.predict_can(frames)
        vehicle_id = payload.get("vehicle_id")
        if vehicle_id is not None:
            if not isinstance(vehicle_id, str) or not vehicle_id.strip() or len(vehicle_id) > 64:
                raise HTTPException(status_code=422, detail="vehicle_id must be a non-empty string of at most 64 characters")
            attached = db.store_can_anomaly_score(vehicle_id.strip(), result["intrusion_score"])
            if not attached:
                raise HTTPException(status_code=404, detail="no telemetry found for vehicle_id")
        result["vehicle_id"] = vehicle_id.strip() if isinstance(vehicle_id, str) else None
        result["telemetry_score_attached"] = vehicle_id is not None
        return result
    except HTTPException:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=f"invalid CAN frame window: {error}") from error

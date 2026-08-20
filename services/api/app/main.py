from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .ml import AuraModels


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
    try:
        db.fetch_vehicle_ids()
        return {"status": "ok", "db": True}
    except Exception as exc:
        return {"status": "degraded", "db": False, "error": str(exc)}


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
        "advisory_action": result.advisory_action,
        "advisory_confidence": result.advisory_confidence,
    }

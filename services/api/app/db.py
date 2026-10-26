from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from urllib.parse import unquote, urlparse

import mysql.connector

from .config import DATABASE_URL


@contextmanager
def connection():
    parsed = urlparse(DATABASE_URL)
    conn = mysql.connector.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=unquote(parsed.username or "root"),
        password=unquote(parsed.password or ""),
        database=parsed.path.lstrip("/"),
        connection_timeout=2,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_latest(vehicle_id: str | None = None) -> dict[str, Any] | None:
    sql = """
        SELECT id, time, vehicle_id, pack_voltage, pack_current, soc_reported,
               cell_temp_max, speed_kmh, odometer_km, latitude, longitude,
               battery_soh, motor_temp, motor_vibration, power_consumption, acceleration_mps2,
               component_health, failure_probability, rul_score, maintenance_type, can_anomaly_score
        FROM vehicle_telemetry
        WHERE (%s IS NULL OR vehicle_id = %s)
        ORDER BY time DESC, id DESC
        LIMIT 1
    """
    with connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (vehicle_id, vehicle_id))
            return cur.fetchone()


def fetch_recent(vehicle_id: str | None = None, limit: int = 120) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    sql = """
        SELECT id, time, vehicle_id, pack_voltage, pack_current, soc_reported,
               cell_temp_max, speed_kmh, odometer_km, latitude, longitude,
               battery_soh, motor_temp, motor_vibration, power_consumption, acceleration_mps2,
               component_health, failure_probability, rul_score, maintenance_type, can_anomaly_score
        FROM vehicle_telemetry
        WHERE (%s IS NULL OR vehicle_id = %s)
        ORDER BY time DESC, id DESC
        LIMIT %s
    """
    with connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (vehicle_id, vehicle_id, limit))
            rows = cur.fetchall()
    rows.reverse()
    return rows


def fetch_vehicle_ids() -> list[str]:
    sql = "SELECT DISTINCT vehicle_id FROM vehicle_telemetry ORDER BY vehicle_id"
    with connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql)
            return [row["vehicle_id"] for row in cur.fetchall()]


def fetch_fleet_latest() -> list[dict[str, Any]]:
    sql = """
        SELECT t.id, t.time, t.vehicle_id, t.pack_voltage, t.pack_current, t.soc_reported,
               t.cell_temp_max, t.speed_kmh, t.odometer_km, t.latitude, t.longitude,
               t.battery_soh, t.motor_temp, t.motor_vibration, t.power_consumption, t.acceleration_mps2,
               t.component_health, t.failure_probability, t.rul_score, t.maintenance_type, t.can_anomaly_score
        FROM vehicle_telemetry AS t
        INNER JOIN (
            SELECT vehicle_id, MAX(id) AS latest_id
            FROM vehicle_telemetry GROUP BY vehicle_id
        ) AS latest ON latest.vehicle_id = t.vehicle_id AND latest.latest_id = t.id
        ORDER BY t.vehicle_id
    """
    with connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql)
            return cur.fetchall()


def store_can_anomaly_score(vehicle_id: str, score: float) -> bool:
    """Attach an IDS window score to the latest telemetry sample for a vehicle."""
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM vehicle_telemetry WHERE vehicle_id = %s ORDER BY time DESC, id DESC LIMIT 1",
                (vehicle_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute("UPDATE vehicle_telemetry SET can_anomaly_score = %s WHERE id = %s", (score, row[0]))
            conn.commit()
            return cur.rowcount == 1

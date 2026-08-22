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
        SELECT time, vehicle_id, pack_voltage, pack_current, soc_reported,
               cell_temp_max, speed_kmh, odometer_km, latitude, longitude
        FROM vehicle_telemetry
        WHERE (%s IS NULL OR vehicle_id = %s)
        ORDER BY time DESC
        LIMIT 1
    """
    with connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (vehicle_id, vehicle_id))
            return cur.fetchone()


def fetch_recent(vehicle_id: str | None = None, limit: int = 120) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    sql = """
        SELECT time, vehicle_id, pack_voltage, pack_current, soc_reported,
               cell_temp_max, speed_kmh, odometer_km, latitude, longitude
        FROM vehicle_telemetry
        WHERE (%s IS NULL OR vehicle_id = %s)
        ORDER BY time DESC
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

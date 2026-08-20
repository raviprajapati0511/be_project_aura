from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL


@contextmanager
def connection():
    with psycopg.connect(DATABASE_URL, connect_timeout=2, row_factory=dict_row) as conn:
        yield conn


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
        with conn.cursor() as cur:
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
        with conn.cursor() as cur:
            cur.execute(sql, (vehicle_id, vehicle_id, limit))
            rows = cur.fetchall()
    rows.reverse()
    return rows


def fetch_vehicle_ids() -> list[str]:
    sql = "SELECT DISTINCT vehicle_id FROM vehicle_telemetry ORDER BY vehicle_id"
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row["vehicle_id"] for row in cur.fetchall()]

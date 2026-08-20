"""AURA — EV Telemetry Simulator.

Generates fake OBD-II / CAN-bus style telemetry for one vehicle, drives it along
a route, models battery SOC/thermal/current, and injects occasional anomalies.
Sends each sample to the Node.js ingestion service over HTTP.

Run:  python -m aura_simulator    (after activating venv in services/api)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error
import json


def load_root_env():
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_root_env()

VEHICLE_ID = os.environ.get("SIM_VEHICLE_ID", "EV-AURA-001")
INGEST_BASE_URL = os.environ.get("INGEST_URL") or os.environ.get(
    "VITE_INGEST_URL", "http://localhost:4000"
)
INGEST_URL = INGEST_BASE_URL.rstrip("/") + "/ingest"
INTERVAL_MS = int(os.environ.get("SIM_INTERVAL_MS", "1000"))

# Nominal pack: 96s Li-ion, ~400V nominal, 75 kWh gross.
PACK_CAPACITY_KWH = 75.0
NOMINAL_PACK_V = 396.0
CELLS_SERIES = 96


@dataclass
class Telemetry:
    time: str
    vehicle_id: str
    pack_voltage: float
    pack_current: float
    soc_reported: float
    cell_temp_max: float
    speed_kmh: float
    odometer_km: float
    latitude: float
    longitude: float


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


class EVSimulator:
    """Stateful EV model driven along a closed loop route."""

    def __init__(self):
        self.soc = 80.0  # %
        self.odometer = 14253.7  # km
        self.cell_temp = 24.0  # C
        self.speed = 0.0
        self.t = 0
        # Loop route around a town square (roughly ~10 km loop).
        self.route = [
            (37.7749, -122.4194),
            (37.7849, -122.4194),
            (37.7849, -122.4094),
            (37.7749, -122.4094),
        ]
        self.seg = 0
        self.seg_progress = 0.0
        self.lat, self.lon = self.route[0]
        # Inject anomalies every N ticks.
        self.tick_anomaly_at = random.randint(60, 120)
        self.anomaly_remaining = 0

    # Physically-plausible battery + vehicle model.
    def step(self):
        self.t += 1
        target_speed = self._target_speed()
        # Smooth toward target speed (simple low-pass).
        self.speed += (target_speed - self.speed) * 0.25
        if self.speed < 0:
            self.speed = 0.0

        # Advance along route.
        dist_km = self.speed / 3600.0  # 1s tick
        self._advance(dist_km)

        # Power model: P = F_drag*v + accel + aux; simplify to current draw.
        v_ms = self.speed / 3.6
        # Rolling + aero + aux, plus regen when decelerating.
        base_kw = 0.18 * v_ms + 0.0009 * (v_ms ** 3) + 0.6
        if target_speed < self.speed - 1.0 and self.speed > 5.0:
            # Regenerative braking returns some energy.
            base_kw = -0.6 * base_kw
        # Occasional spike from HVAC/heat.
        base_kw += random.uniform(-0.3, 0.8)

        pack_v = NOMINAL_PACK_V + (self.soc - 50.0) * 0.35
        pack_i = (base_kw * 1000.0) / pack_v if pack_v > 0 else 0.0

        # SOC integration in Ah-equivalent: dSOC = -I * dt / (kWh*1000/V) *100
        capacity_ah = PACK_CAPACITY_KWH * 1000.0 / NOMINAL_PACK_V
        d_soc = -(pack_i * (INTERVAL_MS / 1000.0) / 3600.0) / capacity_ah * 100.0
        self.soc += d_soc
        self.soc = max(0.0, min(100.0, self.soc))

        # Thermal: simple cell heating from |I| + cooling toward ambient.
        heat = abs(pack_i) * 0.0015
        self.cell_temp += heat - (self.cell_temp - 22.0) * 0.002
        self.cell_temp = max(15.0, min(55.0, self.cell_temp))

        # Anomaly injection: over-voltage / over-current / thermal event.
        fv, fi, ft = pack_v, pack_i, self.cell_temp
        if self.t == self.tick_anomaly_at:
            kind = random.choice(["voltage", "current", "thermal"])
            self.anomaly_remaining = random.randint(3, 8)
            self._anomaly_kind = kind
        if self.anomaly_remaining > 0:
            self.anomaly_remaining -= 1
            if self._anomaly_kind == "voltage":
                fv = pack_v + random.uniform(8.0, 14.0)
            elif self._anomaly_kind == "current":
                fi = pack_i + random.uniform(60.0, 110.0)
            elif self._anomaly_kind == "thermal":
                ft = self.cell_temp + random.uniform(6.0, 12.0)
            if self.anomaly_remaining == 0:
                self.tick_anomaly_at = self.t + random.randint(60, 120)

        return Telemetry(
            time=datetime.now(timezone.utc).isoformat(),
            vehicle_id=VEHICLE_ID,
            pack_voltage=round(fv, 3),
            pack_current=round(fi, 3),
            soc_reported=round(self.soc, 2),
            cell_temp_max=round(ft, 2),
            speed_kmh=round(self.speed, 2),
            odometer_km=round(self.odometer, 3),
            latitude=round(self.lat, 6),
            longitude=round(self.lon, 6),
        )

    def _target_speed(self):
        # Urban drive cycle-ish profile: smooth accel/cruise/coast/brake.
        phase = (self.t % 60) / 60.0
        if phase < 0.2:
            return random.uniform(10, 30)
        if phase < 0.55:
            return random.uniform(45, 65)
        if phase < 0.7:
            return random.uniform(20, 35)
        return random.uniform(0, 10)

    def _advance(self, dist_km):
        if dist_km <= 0:
            return
        remaining = dist_km
        for _ in range(4):
            a = self.route[self.seg]
            b = self.route[(self.seg + 1) % len(self.route)]
            seg_len = haversine_km(a[0], a[1], b[0], b[1]) or 0.0001
            headroom = (1.0 - self.seg_progress) * seg_len
            if remaining <= headroom:
                self.seg_progress += remaining / seg_len
                self.lat, self.lon = self._lerp(a, b, self.seg_progress)
                self.odometer += dist_km
                return
            remaining -= headroom
            self.seg_progress = 0.0
            self.seg = (self.seg + 1) % len(self.route)
        self.odometer += dist_km

    @staticmethod
    def _lerp(a, b, t):
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def send(payload: Telemetry) -> bool:
    body = json.dumps(asdict(payload)).encode()
    req = urllib.request.Request(
        INGEST_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 201
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(f"[sim] ingest unreachable: {exc}", file=sys.stderr)
        return False


def main():
    print(f"[sim] vehicle={VEHICLE_ID} interval={INTERVAL_MS}ms -> {INGEST_URL}")
    sim = EVSimulator()
    while True:
        sample = sim.step()
        if not send(sample):
            # Backoff so we don't spam a dead server.
            time.sleep(INTERVAL_MS / 1000.0)
        time.sleep(INTERVAL_MS / 1000.0)


if __name__ == "__main__":
    main()

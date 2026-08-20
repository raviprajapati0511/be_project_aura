# AURA - Autonomous Unified Reasoning Architecture

Local Proof-of-Concept for an EV intelligence platform:

```text
Python simulator -> Node.js ingest -> TimescaleDB -> FastAPI ML -> React dashboard
```

## Stack

- Data ingestion: Node.js, Express, Socket.io
- Database: PostgreSQL with TimescaleDB
- ML backend: Python 3.10+, FastAPI
- ML libraries: scikit-learn Isolation Forest, torch BiLSTM, stable-baselines3 DQN
- Frontend: React, Vite, Tailwind CSS, Mapbox GL JS

## Prerequisites

- Node.js 20+
- Python 3.10+
- Docker with `docker compose`
- Mapbox access token for the live map

## Quick Start

```powershell
copy .env.example .env
# Edit .env and set VITE_MAPBOX_TOKEN.

npm install
npm run setup
npm start
```

Open <http://localhost:5173>.

The first setup installs Node dependencies, creates `.venv`, installs Python
requirements, and trains the small local PoC models into `services/api/models`.

## Manual Run

Use this when you want separate terminals per service.

```powershell
docker compose up db
npm --prefix services/ingest start
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir services/api --host 0.0.0.0 --port 8000
npm --prefix services/web run dev
.\.venv\Scripts\python services/simulator/aura_simulator.py
```

## Services

| Service | Port | Directory |
| --- | ---: | --- |
| TimescaleDB | 5433 | `docker-compose.yml` |
| Ingestion API + Socket.io | 4000 | `services/ingest` |
| FastAPI ML backend | 8000 | `services/api` |
| React dashboard | 5173 | `services/web` |
| Telemetry simulator | n/a | `services/simulator` |

## API Surface

Node ingest:

- `GET /health`
- `GET /healthz`
- `POST /ingest`
- Socket.io event: `telemetry:new`

FastAPI:

- `GET /health`
- `GET /vehicles`
- `GET /telemetry/latest`
- `GET /telemetry/recent?limit=120`
- `GET /inference?limit=80`

## Data Contract

```sql
CREATE TABLE vehicle_telemetry (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id VARCHAR(64),
    pack_voltage DOUBLE PRECISION,
    pack_current DOUBLE PRECISION,
    soc_reported DOUBLE PRECISION,
    cell_temp_max DOUBLE PRECISION,
    speed_kmh DOUBLE PRECISION,
    odometer_km DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);
SELECT create_hypertable('vehicle_telemetry', 'time');
```

## Scope

Included: simulated OBD-II/CAN-style telemetry, end-to-end local data flow,
TimescaleDB persistence, ML inference, and geospatial dashboarding.

Excluded: hardware serial/CAN connections and authentication.

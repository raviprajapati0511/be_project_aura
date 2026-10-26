# AURA EV Intelligence

A local EV fleet monitoring and predictive maintenance application. A repeatable
Kaggle telemetry replay feeds the same ingestion API used by a vehicle sender,
MySQL stores each sample, FastAPI scores battery and motor health, and the React
dashboard presents live fleet status.

```text
Kaggle EV telemetry -> replay adapter -> Node.js ingest -> MySQL
                                               |             |
                                               +-> Socket.io +-> FastAPI ML -> React dashboard
```

## Dataset

The bundled archive at `services/api/data/eviot-predictivemaint-dataset.zip` is
the [EVIoT-PredictiveMaint Dataset on Kaggle](https://www.kaggle.com/datasets/datasetengineer/eviot-predictivemaint-dataset),
version 1. It contains 175,393 15-minute telemetry records, including battery
SoC/SoH, voltage/current/temperature, motor measurements, driving speed,
component health, a failure label, and a remaining useful life score. The
dataset is licensed **CC BY-NC-SA 4.0**. Attribute DatasetEngineer and the
Kaggle dataset page when redistributing derived work; the license excludes
commercial use and requires share-alike terms.

The source has no vehicle identifiers or GPS. For a useful local demonstration,
the replay adapter distributes records across five clearly synthetic AURA demo
vehicle IDs and plots generated coordinates. Those IDs and coordinates are not
source observations. It stamps replayed rows with the current time; the original
source interval is 15 minutes. The included `SIM_MODE=synthetic` option runs the
older procedural single-vehicle simulator instead.

Dataset values and labels are used as published; they are not independently
validated against vehicle hardware. Treat model outputs as a portfolio/research
demonstration and review the source documentation before drawing operational
conclusions.

## Features

- Live five-vehicle selector, service health, telemetry trends, route context,
  recent activity, and reconnecting WebSocket updates.
- Fleet comparison, filtering, live stream pause/resume, and CSV export.
- Battery SoC/range estimates from a Kalman-preprocessed CNN-BiLSTM, plus
  Isolation Forest anomalies, Random Forest failure classification, and RUL.
- DQN smart charging recommendations with tariff, renewable, and grid context; a
  telemetry safety filter derates or defers recommendations for anomaly, heat,
  maintenance risk, or suspected-attack evidence. Policy metrics are compared
  with a rule baseline in a synthetic environment.
- Research results export, inference latency, and local ingestion security and
  data-quality counters.
- Explainable battery, motor, health-score, anomaly, and predicted-risk alerts,
  plus a power-versus-motion residual and optional external CAN detector score.
- Validated ingestion endpoint, MySQL persistence, schema migration for
  existing databases, and a local synthetic telemetry mode.

Models are trained from three task-specific sources: the bundled Kaggle EVIoT
telemetry dataset, Kaggle EV Charging Patterns, and a Kaggle mirror of the HCRL
Car-Hacking captures. Setup downloads the latter two. The model report is
[`services/api/models/evaluation.json`](services/api/models/evaluation.json).
The latest local run used 25,000 telemetry records, 1,131 cleaned charging
sessions, and all five CAN captures.

### Held-out results

- SoC CNN-BiLSTM: **6.21 percentage-point MAE** on a chronological holdout.
  Range MAE is **30.90 km** against a derived proxy, not measured range.
- Charging session model: **18.53 kWh energy MAE** and **0.87 hour duration
  MAE** on 227 later sessions. Both R² values are negative, so treat these
  estimates as illustrative; the dashboard displays this limitation.
- CAN classifier: **0.9976 macro F1** and **0.9987 binary intrusion F1** on
  26,341 test windows. Later windows from each class are held out within the
  same captures; this does not establish performance on unseen vehicles or a
  continuous future stream. Probabilities are uncalibrated.
- Local telemetry inference benchmark: **101 ms p50 / 135 ms p95** on this
  machine. Training took about **13 minutes** here; both timings vary by system.

Failure and RUL predictions are research signals, not validated vehicle
controls or safety predictions. The source does not specify RUL units. Charging
recommendations use a synthetic tariff/grid simulator and do not control a real
charger. The CAN model is a windowed batch detector; it does not connect to a
physical CAN bus. See [`docs/research-alignment.md`](docs/research-alignment.md)
for module coverage and limitations against the attached paper.

## Requirements

- Node.js 20+
- Python 3.11 (Windows Python launcher `py -3.11`)
- Docker Desktop with Docker Compose, or an existing local MySQL 8 server

Initial setup needs internet access to download the auxiliary Kaggle datasets
(about 1 GB of raw CAN captures); Kaggle may require an account/API token
depending on its download policy. The EVIoT archive is bundled. Docker Desktop
must be running for MySQL. A Mapbox token only enables geographic tiles.

## Quick start (Windows PowerShell)

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run setup
npm start
```

Open <http://localhost:5173>. The first setup installs dependencies, downloads
the auxiliary data, and trains/evaluates task-specific models. To download the
datasets separately, run `npm run data`. `npm start` launches Docker MySQL,
ingestion, FastAPI, the dashboard, and dataset replay. If MySQL is already
running locally, create the configured database and apply `db/init.sql`, then
run `npm run start:local` to start the four app services without Docker. The
dashboard populates as replayed samples arrive. Full model training took about
13 minutes in this workspace; timing varies with CPU and disk speed.

## Run services separately

```powershell
npm run db
npm run ingest
npm run api
npm run web
npm run sim
```

Set `SIM_MODE=synthetic` in `.env` to use the procedural simulator. Set
`SIM_INTERVAL_MS` to control replay pace. The default dataset replay sends one
row per interval and cycles through the source records continuously.

## API

Ingest service (port 4000):

- `GET /health` and `GET /healthz`
- `POST /ingest` accepts telemetry and persists it to MySQL
- Socket.io `telemetry:new` event broadcasts accepted samples

FastAPI (port 8000):

- `GET /health`
- `GET /vehicles`
- `GET /telemetry/latest?vehicle_id=EV-AURA-001`
- `GET /telemetry/recent?vehicle_id=EV-AURA-001&limit=120`
- `GET /inference?vehicle_id=EV-AURA-001&limit=120`
- `GET /fleet/summary`
- `GET /research/evaluation`
- `GET /charging/forecast?soc_start_pct=30&soc_target_pct=80`
- `POST /security/can/inference` with `{"frames":[{"timestamp":1.0,"can_id":"0316","dlc":8,"data":["05","21","68","09","21","21","00","6f"]}, ...]}`; provide 100-5,000 frames for windowed predictions.

Ingestion service (port 4000) also provides `GET /security/summary`. These
counters cover HTTP payload validation and local rate limiting only.

## Configuration

Copy `.env.example` to `.env`. Set the MySQL connection values consistently
across `DATABASE_URL`, `PY_DATABASE_URL`, and the `MYSQL_*` variables. For an
existing MySQL server, create the database named by `MYSQL_DATABASE` and apply
`db/init.sql` to it before running `npm run start:local`. Optional frontend
settings include `VITE_MAPBOX_TOKEN`, `VITE_API_URL`, and `VITE_INGEST_URL`.

## Project structure

- `services/simulator/` dataset replay adapter and synthetic simulator
- `services/ingest/` telemetry validation, MySQL writes, Socket.io broadcast
- `services/api/` database reads, inference, and model training
- `services/web/` React/Vite fleet dashboard
- `db/init.sql` telemetry schema

# AURA FastAPI ML service

Runs the EV intelligence layer for the local PoC.

```powershell
py -3.11 -m venv ..\..\.venv
..\..\.venv\Scripts\python -m pip install -r requirements.txt
..\..\.venv\Scripts\python scripts\train_models.py
..\..\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /vehicles`
- `GET /telemetry/latest`
- `GET /telemetry/recent?limit=120`
- `GET /inference?limit=80`


## Auxiliary Kaggle datasets

From the repository root, `npm run data` downloads Kaggle EV Charging Patterns
and the HCRL Car-Hacking capture files. `npm run setup` downloads them before
training. The downloads are kept under `services/api/data/` and are intentionally
not committed. Training reports dataset provenance and chronological holdouts in
`services/api/models/evaluation.json`; the CAN evaluation takes later windows
from each class because attack periods are localized in the source captures.

The session endpoint forecasts energy and duration from charger/session inputs:

```text
GET /charging/forecast?soc_start_pct=30&soc_target_pct=80
```

The CAN endpoint accepts frame windows in JSON. Provide `vehicle_id` to attach
the resulting intrusion score to the latest telemetry row; the cross-domain
reasoner then consumes it on subsequent inference requests.

```json
{
  "vehicle_id": "EV-AURA-001",
  "frames": [
    {"timestamp": 1478191030.045, "can_id": "0316", "dlc": 8, "data": ["05", "22", "68", "09", "22", "20", "00", "75"]}
  ]
}
```

The endpoint requires 100 to 5,000 frames per request and returns predictions
for non-overlapping 100-frame windows. Model probabilities are exploratory and
uncalibrated.

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

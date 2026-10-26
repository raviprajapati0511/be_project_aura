from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_dotenv() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

DATABASE_URL = os.environ.get(
    "PY_DATABASE_URL",
    os.environ.get("DATABASE_URL", "mysql://root:change_me@localhost:3306/fitness_db"),
)
API_PORT = int(os.environ.get("API_PORT", "8000"))

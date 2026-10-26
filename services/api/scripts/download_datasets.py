"""Download the public task-specific Kaggle datasets used by AURA training."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "services" / "api" / "data"
KAGGLE = Path(sys.executable).with_name("kaggle.exe")
if not KAGGLE.exists():
    KAGGLE = Path(shutil.which("kaggle") or "kaggle")


def download(dataset: str, filename: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[data] already present: {target.relative_to(ROOT)}")
        return
    command = [
        str(KAGGLE), "datasets", "download", "-d", dataset,
        "-f", filename, "-p", str(destination), "--unzip",
    ]
    print(f"[data] downloading {dataset}:{filename}")
    subprocess.run(command, cwd=ROOT, check=True)
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Kaggle download did not produce {target}")


def main() -> None:
    download(
        "valakhorasani/electric-vehicle-charging-patterns",
        "ev_charging_patterns.csv",
        DATA / "charging-patterns",
    )
    can = DATA / "car-hacking"
    for filename in (
        "normal_run_data.txt", "DoS_dataset.csv", "Fuzzy_dataset.csv",
        "RPM_dataset.csv", "gear_dataset.csv",
    ):
        download("pranavjha24/car-hacking-dataset", filename, can)
    print("[data] all auxiliary Kaggle datasets are ready")


if __name__ == "__main__":
    main()

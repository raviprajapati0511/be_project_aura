from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from stable_baselines3 import DQN
from torch import nn

from .config import MODEL_DIR


FEATURES = [
    "pack_voltage",
    "pack_current",
    "soc_reported",
    "cell_temp_max",
    "speed_kmh",
    "odometer_km",
]
SEQ_LEN = 20


class BiLSTMRangeEstimator(nn.Module):
    def __init__(self, input_dim: int = len(FEATURES), hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class InferenceResult:
    anomaly: bool
    anomaly_score: float
    estimated_range_km: float
    advisory_action: str
    advisory_confidence: float


def row_to_vector(row: dict[str, Any]) -> np.ndarray:
    return np.array([float(row.get(k) or 0.0) for k in FEATURES], dtype=np.float32)


def normalize_sequence(rows: list[dict[str, Any]]) -> np.ndarray:
    vectors = [row_to_vector(row) for row in rows[-SEQ_LEN:]]
    if not vectors:
        vectors = [np.zeros(len(FEATURES), dtype=np.float32)]
    while len(vectors) < SEQ_LEN:
        vectors.insert(0, vectors[0].copy())
    arr = np.stack(vectors).astype(np.float32)
    scale = np.array([430.0, 220.0, 100.0, 70.0, 140.0, 200000.0], dtype=np.float32)
    return arr / scale


class AuraModels:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.isolation_forest = self._load_isolation_forest()
        self.range_model = self._load_range_model()
        self.dqn = self._load_dqn()

    def _load_isolation_forest(self) -> IsolationForest:
        path = self.model_dir / "isolation_forest.pkl"
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        rng = np.random.default_rng(7)
        normal = rng.normal(
            loc=[402.0, 25.0, 70.0, 29.0, 42.0, 15000.0],
            scale=[5.0, 18.0, 9.0, 4.0, 18.0, 1000.0],
            size=(600, len(FEATURES)),
        )
        return IsolationForest(contamination=0.04, random_state=7).fit(normal)

    def _load_range_model(self) -> BiLSTMRangeEstimator:
        model = BiLSTMRangeEstimator()
        path = self.model_dir / "bilstm_range.pt"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def _load_dqn(self) -> DQN | None:
        path = self.model_dir / "eco_dqn.zip"
        if path.exists():
            return DQN.load(path)
        return None

    def predict(self, rows: list[dict[str, Any]]) -> InferenceResult:
        latest = rows[-1] if rows else {}
        feature_vec = row_to_vector(latest).reshape(1, -1)
        raw_score = float(self.isolation_forest.decision_function(feature_vec)[0])
        anomaly_flag = bool(self.isolation_forest.predict(feature_vec)[0] == -1)

        seq = normalize_sequence(rows)
        with torch.no_grad():
            pred = self.range_model(torch.tensor(seq[None, :, :], dtype=torch.float32))
        estimated_range = float(torch.clamp(pred, min=0.0, max=650.0).item())

        action_idx, confidence = self._predict_advisory(latest)
        actions = ["maintain", "coast", "brake"]
        return InferenceResult(
            anomaly=anomaly_flag,
            anomaly_score=round(raw_score, 4),
            estimated_range_km=round(estimated_range, 1),
            advisory_action=actions[action_idx],
            advisory_confidence=round(confidence, 2),
        )

    def _predict_advisory(self, row: dict[str, Any]) -> tuple[int, float]:
        speed = float(row.get("speed_kmh") or 0.0)
        current = float(row.get("pack_current") or 0.0)
        temp = float(row.get("cell_temp_max") or 0.0)
        soc = float(row.get("soc_reported") or 0.0)
        obs = np.array([speed / 140.0, current / 220.0, temp / 70.0, soc / 100.0], dtype=np.float32)
        if self.dqn is not None:
            action, _ = self.dqn.predict(obs, deterministic=True)
            return int(action), 0.82
        if temp > 43 or current > 95:
            return 2, 0.74
        if speed > 72 or current > 55:
            return 1, 0.68
        return 0, 0.61

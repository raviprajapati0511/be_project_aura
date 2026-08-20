from __future__ import annotations

import pickle
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sklearn.ensemble import IsolationForest
from stable_baselines3 import DQN
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import MODEL_DIR  # noqa: E402
from app.ml import BiLSTMRangeEstimator, FEATURES, SEQ_LEN  # noqa: E402


class EcoDriveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.rng = np.random.default_rng(11)
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        speed, current, temp, soc = self.state
        energy_penalty = current * 0.6 + speed * 0.2
        thermal_penalty = max(0.0, temp - 0.62) * 1.4
        action_bonus = 0.0
        if action == 1 and (speed > 0.55 or current > 0.32):
            action_bonus = 0.9
        if action == 2 and (temp > 0.64 or current > 0.5):
            action_bonus = 1.1
        if action == 0 and current < 0.28 and temp < 0.58:
            action_bonus = 0.7
        reward = action_bonus - energy_penalty - thermal_penalty
        self.state = self._sample_state()
        return self.state, float(reward), False, False, {}

    def _sample_state(self):
        return np.array(
            [
                self.rng.uniform(0.0, 0.9),
                self.rng.uniform(-0.1, 0.75),
                self.rng.uniform(0.25, 0.82),
                self.rng.uniform(0.12, 0.95),
            ],
            dtype=np.float32,
        )


def synthetic_rows(n: int = 1200):
    rng = np.random.default_rng(5)
    normal = np.column_stack(
        [
            rng.normal(402.0, 6.0, n),
            rng.normal(24.0, 22.0, n),
            rng.uniform(18.0, 92.0, n),
            rng.normal(30.0, 4.5, n),
            rng.uniform(0.0, 95.0, n),
            rng.normal(14800.0, 900.0, n),
        ]
    ).astype(np.float32)
    return normal


def train_isolation_forest() -> None:
    model = IsolationForest(contamination=0.04, random_state=7)
    model.fit(synthetic_rows())
    with (MODEL_DIR / "isolation_forest.pkl").open("wb") as fh:
        pickle.dump(model, fh)


def train_range_model() -> None:
    rng = np.random.default_rng(9)
    x = []
    y = []
    for _ in range(600):
        soc = rng.uniform(8.0, 95.0)
        speed = rng.uniform(15.0, 95.0)
        current = rng.uniform(4.0, 80.0)
        temp = rng.uniform(20.0, 44.0)
        odo = rng.uniform(1000.0, 60000.0)
        voltage = 386.0 + soc * 0.22 + rng.normal(0.0, 2.0)
        seq = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
        for t in range(SEQ_LEN):
            drift = (t - SEQ_LEN) / SEQ_LEN
            seq[t] = [
                voltage + rng.normal(0.0, 1.2),
                current + rng.normal(0.0, 4.0),
                soc + drift * 0.25,
                temp + rng.normal(0.0, 0.8),
                speed + rng.normal(0.0, 3.0),
                odo + t * speed / 3600.0,
            ]
        efficiency = 5.3 - min(speed, 100.0) * 0.012 - max(temp - 35.0, 0.0) * 0.04
        range_km = max(0.0, soc / 100.0 * 75.0 * max(efficiency, 3.0))
        x.append(seq / np.array([430.0, 220.0, 100.0, 70.0, 140.0, 200000.0], dtype=np.float32))
        y.append(range_km)

    device = torch.device("cpu")
    model = BiLSTMRangeEstimator().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.006)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.tensor(np.stack(x), dtype=torch.float32, device=device)
    ty = torch.tensor(np.array(y), dtype=torch.float32, device=device)
    model.train()
    for _ in range(18):
        opt.zero_grad()
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), MODEL_DIR / "bilstm_range.pt")


def train_dqn() -> None:
    env = EcoDriveEnv()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=2000,
        learning_starts=100,
        batch_size=32,
        gamma=0.92,
        verbose=0,
        seed=17,
    )
    model.learn(total_timesteps=1200)
    model.save(MODEL_DIR / "eco_dqn.zip")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_isolation_forest()
    train_range_model()
    train_dqn()
    print(f"[api] trained models -> {MODEL_DIR}")


if __name__ == "__main__":
    main()

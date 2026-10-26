from __future__ import annotations

from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def charging_context(soc: float, soh: float = 1.0, now: datetime | None = None) -> dict[str, float | str]:
    """Transparent demo tariff/grid context; replace with a utility feed in deployment."""
    now = now or datetime.now(timezone.utc)
    hour = now.hour
    price = 0.12 if hour < 7 or hour >= 22 else (0.32 if 16 <= hour < 21 else 0.21)
    grid_load = 0.46 + (0.34 if 17 <= hour < 21 else 0.0)
    renewable = float(max(0.0, min(0.75, 0.5 * np.sin((hour - 6) * np.pi / 14))))
    target_soc = 0.8 if soc < 80 else 0.95
    return {
        "soc": float(np.clip(soc / 100.0, 0, 1)),
        "soh": float(np.clip(soh, 0, 1)),
        "target_soc": target_soc,
        "price": price,
        "grid_load": grid_load,
        "renewable": renewable,
        "hour": float(hour),
        "currency": "USD/kWh (demo tariff)",
    }


def charging_observation(context: dict[str, float | str]) -> np.ndarray:
    return np.asarray([
        float(context["soc"]),
        float(context["price"]) / 0.5,
        float(context["grid_load"]),
        float(context["renewable"]),
        float(context["hour"]) / 24.0,
        float(context["target_soc"]),
    ], dtype=np.float32)


def constrain_charging_action(
    action: int,
    observation: np.ndarray,
    *,
    anomaly: bool = False,
    cell_temp_c: float = 25.0,
    failure_probability: float = 0.0,
    security_restrict: bool = False,
) -> int:
    """Apply grid, target, and conservative telemetry safety restrictions."""
    soc, price_norm, grid_load, renewable, _hour, target = map(float, observation)
    if soc >= target:
        return 0
    if action == 0 or (action == 1 and price_norm < 0.42 and grid_load < 0.75 and renewable > 0.2):
        action = 2 if price_norm < 0.42 and grid_load < 0.75 and renewable > 0.2 else 1
    if action == 2 and grid_load + EVChargingEnv.POWER_KW[2] / 24.0 > 0.98:
        action = 1
    # Apply restrictions after baseline scheduling so its target logic cannot
    # turn a safety defer into a charge request.
    if security_restrict or cell_temp_c >= 55.0 or failure_probability >= 0.85:
        return 0
    if anomaly or cell_temp_c >= 48.0 or failure_probability >= 0.65:
        action = min(int(action), 1)
    return int(action)


class EVChargingEnv(gym.Env):
    """Toy 15-minute single-vehicle charging environment for policy research."""

    metadata = {"render_modes": []}
    POWER_KW = (0.0, 3.3, 7.2)

    def __init__(self, seed: int = 17):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(6, dtype=np.float32)
        self.steps = 0
        self.cost = 0.0
        self.peak = 0.0
        self.violations = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.cost = 0.0
        self.peak = 0.0
        self.violations = 0
        # Every generated trip is feasible within the modeled departure window.
        # This lets the policy learn a meaningful cost/peak/completion tradeoff
        # rather than reward it for deferring an impossible charging request.
        soc = self.rng.uniform(0.25, 0.68)
        target = min(0.95, soc + self.rng.uniform(0.08, 0.26))
        hour = self.rng.uniform(0.0, 1.0)
        self.state = np.asarray([
            soc,
            *self._grid_context(hour),
            hour,
            target,
        ], dtype=np.float32)
        return self.state, {}

    @staticmethod
    def _grid_context(hour: float) -> tuple[float, float, float]:
        hour_of_day = hour * 24.0
        price = 0.12 if hour_of_day < 7 or hour_of_day >= 22 else (0.32 if 16 <= hour_of_day < 21 else 0.21)
        grid_load = 0.46 + (0.34 if 17 <= hour_of_day < 21 else 0.0)
        renewable = max(0.0, min(0.75, 0.5 * np.sin((hour_of_day - 6) * np.pi / 14)))
        return price / 0.5, grid_load, renewable

    def step(self, action):
        soc, price_norm, grid_load, renewable, hour, target = self.state
        power = self.POWER_KW[int(action)]
        energy = power * 0.25
        next_soc = min(1.0, soc + energy / 75.0)
        price = price_norm * 0.5
        cost = energy * price * (1.0 - 0.35 * renewable)
        peak = max(0.0, grid_load + power / 24.0 - 0.98)
        self.cost += cost
        self.peak = max(self.peak, grid_load + power / 24.0)
        self.violations += int(peak > 0.0)
        self.steps += 1
        missed_target = self.steps >= 12 and next_soc < target
        # Reward progress toward the requested departure charge so that the
        # learned policy cannot minimize cost by simply never charging.
        reward = 4.0 * (next_soc - soc) - cost * 2.0 - peak * 0.5
        if next_soc >= target:
            reward += 3.0
        if missed_target:
            reward -= 5.0 + 20.0 * (target - next_soc)
        terminated = next_soc >= target or self.steps >= 12
        next_hour = (hour + 0.0104167) % 1.0
        next_price, next_load, next_renewable = self._grid_context(next_hour)
        self.state = np.asarray([next_soc, next_price, next_load, next_renewable, next_hour, target], dtype=np.float32)
        return self.state, float(reward), terminated, False, {
            "cost": self.cost,
            "peak_load": self.peak,
            "constraint_violations": self.violations,
            "target_met": next_soc >= target,
        }

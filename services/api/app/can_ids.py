from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WINDOW_FRAMES = 100
CAN_FEATURES = [
    "message_rate_hz", "mean_interarrival_ms", "std_interarrival_ms",
    "max_interarrival_ms", "unique_id_count", "id_entropy_bits",
    "zero_id_fraction", "dominant_id_fraction", "mean_dlc", "std_dlc",
    "payload_mean", "payload_std", "payload_zero_fraction",
]
CAN_CLASSES = ["benign", "dos", "fuzzy", "rpm_spoofing", "gear_spoofing"]


def _clean_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _can_id(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    raw = str(value).strip().lower().replace("0x", "")
    try:
        return int(raw, 16)
    except ValueError:
        return int(float(raw))


def _byte(value: Any) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return 0
    if isinstance(value, (int, np.integer, float, np.floating)):
        return max(0, min(255, int(value)))
    raw = str(value).strip().lower().replace("0x", "")
    try:
        return max(0, min(255, int(raw, 16)))
    except ValueError:
        try:
            return max(0, min(255, int(float(raw))))
        except ValueError:
            return 0


def normalize_can_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize HCRL/Kaggle and API frame columns to a shared representation."""
    frame = frame.copy()
    frame.columns = [_clean_name(col) for col in frame.columns]
    if "canid" not in frame:
        for candidate in ("id", "arbitrationid", "identifier"):
            if candidate in frame:
                frame["canid"] = frame[candidate]
                break
    if "timestamp" not in frame:
        for candidate in ("time", "ts"):
            if candidate in frame:
                frame["timestamp"] = frame[candidate]
                break
    if "dlc" not in frame or "canid" not in frame or "timestamp" not in frame:
        raise ValueError("CAN records need timestamp, CAN ID, and DLC fields")

    frame["_timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["_can_id"] = frame["canid"].map(_can_id).astype(np.int64)
    frame["_dlc"] = pd.to_numeric(frame["dlc"], errors="coerce").fillna(0).clip(0, 8)
    data_cols = [f"data{i}" for i in range(8)]
    for col in data_cols:
        if col not in frame:
            frame[col] = 0
        frame[col] = frame[col].map(_byte).astype(np.float32)
    frame["_injected"] = 0.0
    if "flag" in frame:
        frame["_injected"] = frame["flag"].astype(str).str.upper().str.strip().eq("T").astype(np.float32)
    elif "label" in frame:
        labels = frame["label"].astype(str).str.lower()
        frame["_injected"] = (~labels.isin({"0", "normal", "benign", "r", "false"})).astype(np.float32)
    return frame[["_timestamp", "_can_id", "_dlc", *data_cols, "_injected"]].dropna(subset=["_timestamp"])


def _features_for_window(window: pd.DataFrame) -> np.ndarray:
    ids = window["_can_id"].to_numpy(dtype=np.int64)
    dlc = window["_dlc"].to_numpy(dtype=np.float32)
    payload = window[[f"data{i}" for i in range(8)]].to_numpy(dtype=np.float32).reshape(-1)
    timestamps = window["_timestamp"].to_numpy(dtype=np.float64)
    delta = np.diff(timestamps) * 1000.0
    delta = delta[np.isfinite(delta) & (delta >= 0)]
    if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
        rate = len(timestamps) / (timestamps[-1] - timestamps[0])
    else:
        rate = float(len(timestamps))
    counts = np.unique(ids, return_counts=True)[1]
    proportions = counts / max(1, len(ids))
    entropy = -float(np.sum(proportions * np.log2(np.maximum(proportions, 1e-12))))
    return np.asarray([
        rate,
        float(np.mean(delta)) if len(delta) else 0.0,
        float(np.std(delta)) if len(delta) else 0.0,
        float(np.max(delta)) if len(delta) else 0.0,
        float(len(counts)), entropy,
        float(np.mean(ids == 0)), float(np.max(proportions)) if len(proportions) else 0.0,
        float(np.mean(dlc)) if len(dlc) else 0.0,
        float(np.std(dlc)) if len(dlc) else 0.0,
        float(np.mean(payload)) if len(payload) else 0.0,
        float(np.std(payload)) if len(payload) else 0.0,
        float(np.mean(payload == 0)) if len(payload) else 0.0,
    ], dtype=np.float32)


def window_can_frames(frame: pd.DataFrame, window_size: int = WINDOW_FRAMES) -> np.ndarray:
    normalized = normalize_can_frame(frame)
    count = len(normalized) // window_size
    if count == 0:
        raise ValueError(f"at least {window_size} CAN frames are required")
    return np.stack([
        _features_for_window(normalized.iloc[index * window_size:(index + 1) * window_size])
        for index in range(count)
    ])


def load_can_capture(path: Path, class_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a HCRL capture and label each fixed, non-overlapping frame window.

    HCRL's Flag column marks injected messages (T) versus normal messages (R).
    If a file omits that flag, its filename supplies only a weak capture label.
    """
    features = []
    labels = []
    file_attack = class_name != "benign"
    carry = pd.DataFrame()

    def append_windows(normalized: pd.DataFrame) -> None:
        nonlocal carry
        combined = pd.concat([carry, normalized], ignore_index=True)
        count = len(combined) // WINDOW_FRAMES
        if count:
            used = combined.iloc[:count * WINDOW_FRAMES]
            timestamps = used["_timestamp"].to_numpy(dtype=np.float64).reshape(count, WINDOW_FRAMES)
            ids = used["_can_id"].to_numpy(dtype=np.int64).reshape(count, WINDOW_FRAMES)
            dlc = used["_dlc"].to_numpy(dtype=np.float32).reshape(count, WINDOW_FRAMES)
            payload = used[[f"data{i}" for i in range(8)]].to_numpy(dtype=np.float32).reshape(count, -1)
            injected = used["_injected"].to_numpy(dtype=np.float32).reshape(count, WINDOW_FRAMES)
            deltas = np.diff(timestamps, axis=1) * 1000.0
            deltas[deltas < 0] = np.nan
            batch_features = np.zeros((count, len(CAN_FEATURES)), dtype=np.float32)
            for index in range(count):
                unique_ids, id_counts = np.unique(ids[index], return_counts=True)
                proportions = id_counts / WINDOW_FRAMES
                delta = deltas[index]
                delta = delta[np.isfinite(delta)]
                elapsed = timestamps[index, -1] - timestamps[index, 0]
                batch_features[index] = [
                    WINDOW_FRAMES / elapsed if elapsed > 0 else float(WINDOW_FRAMES),
                    float(np.mean(delta)) if len(delta) else 0.0,
                    float(np.std(delta)) if len(delta) else 0.0,
                    float(np.max(delta)) if len(delta) else 0.0,
                    float(len(unique_ids)),
                    float(-np.sum(proportions * np.log2(np.maximum(proportions, 1e-12)))),
                    float(np.mean(ids[index] == 0)),
                    float(np.max(proportions)) if len(proportions) else 0.0,
                    float(np.mean(dlc[index])),
                    float(np.std(dlc[index])),
                    float(np.mean(payload[index])),
                    float(np.std(payload[index])),
                    float(np.mean(payload[index] == 0)),
                ]
            has_flag = "attack_flag_available" in normalized.attrs
            for index, feature in enumerate(batch_features):
                is_attack = float(np.mean(injected[index])) > 0 if has_flag else file_attack
                features.append(feature)
                labels.append(class_name if is_attack else "benign")
        carry = combined.iloc[count * WINDOW_FRAMES:].reset_index(drop=True)

    if path.suffix.lower() == ".txt":
        pattern = re.compile(r"Timestamp:\s*([0-9.]+)\s+ID:\s*([0-9a-fA-F]+)\s+[0-9a-fA-F]{3}\s+DLC:\s*(\d+)\s*(.*)")
        batch = []
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                match = pattern.search(line)
                if not match:
                    continue
                timestamp, can_id, dlc, data = match.groups()
                values = re.findall(r"[0-9a-fA-F]{1,2}", data)[:8]
                batch.append({
                    "timestamp": timestamp, "can_id": can_id, "dlc": int(dlc),
                    **{f"data{i}": byte for i, byte in enumerate(values)},
                })
                if len(batch) >= WINDOW_FRAMES * 100:
                    append_windows(normalize_can_frame(pd.DataFrame(batch)))
                    batch.clear()
        if batch:
            append_windows(normalize_can_frame(pd.DataFrame(batch)))
    else:
        columns = ["Timestamp", "CAN ID", "DLC", *[f"DATA[{i}]" for i in range(8)], "Flag"]
        for raw in pd.read_csv(path, header=None, names=columns, chunksize=250_000, dtype=str, low_memory=False):
            has_flag = "Flag" in raw
            normalized = normalize_can_frame(raw)
            if has_flag:
                normalized.attrs["attack_flag_available"] = True
            append_windows(normalized)

    if len(features) < 5:
        raise ValueError(f"Capture is too small or unrecognized: {path.name}")
    return np.stack(features), np.asarray(labels, dtype=object)


def frames_from_payload(frames: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for item in frames:
        data = item.get("data", [])
        if isinstance(data, str):
            data = [part for part in re.split(r"[\s,]+", data.strip()) if part]
        if not isinstance(data, list) or len(data) > 8:
            raise ValueError("CAN data must contain at most eight bytes")
        rows.append({
            "timestamp": item.get("timestamp", item.get("time")),
            "can_id": item.get("can_id", item.get("id")),
            "dlc": item.get("dlc", len(data)),
            **{f"data{i}": value for i, value in enumerate(data)},
        })
    return window_can_frames(pd.DataFrame(rows))


def attack_probability(probabilities: dict[str, float]) -> float:
    return float(max(0.0, min(1.0, 1.0 - probabilities.get("benign", 0.0))))

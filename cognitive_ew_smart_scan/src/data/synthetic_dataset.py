"""Synthetic fallback dataset builder for local training.

This creates small, repeatable TSRD-like pulse-train files when the official
TSRD package or remote dataset is unavailable. It is intentionally lightweight
so the project can train safely without downloading a massive external dataset.
"""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np


def make_synthetic_pulse_train(
    n_pulses: int = 400,
    n_emitters: int = 4,
    seed: int = 42,
    time_horizon_us: float = 50000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a TSRD-like pulse train with file-local emitter labels."""
    rng = np.random.default_rng(seed)
    data = []
    labels = []
    for emitter_id in range(n_emitters):
        pulses = int(n_pulses * (0.18 + 0.12 * emitter_id))
        pri = rng.uniform(900.0, 2200.0)
        centre = rng.uniform(2000.0, 16000.0)
        width = rng.uniform(0.5, 8.0)
        amplitude = rng.uniform(8.0, 60.0)
        aoa = rng.uniform(-60.0, 60.0)
        toa = np.sort(rng.uniform(0.0, time_horizon_us, size=pulses))
        toa = toa + emitter_id * 42.0
        cf = np.clip(centre + rng.normal(0.0, 150.0, size=pulses), 0.0, 18000.0)
        pw = np.clip(np.abs(rng.normal(width, width * 0.45, size=pulses)), 0.2, 20.0)
        amp = np.clip(np.abs(rng.normal(amplitude, amplitude * 0.35, size=pulses)), 0.5, 100.0)
        aoa_vec = np.clip(aoa + rng.normal(0.0, 9.0, size=pulses), -90.0, 90.0)
        batch = np.column_stack([toa, cf, pw, aoa_vec, amp]).astype(np.float32)
        data.append(batch)
        labels.append(np.full(pulses, emitter_id, dtype=np.int32))
    noise = rng.uniform(0.0, time_horizon_us, size=max(30, int(n_pulses * 0.12)))
    noise_cf = rng.uniform(0.0, 18000.0, size=noise.shape[0])
    noise_pw = rng.uniform(0.25, 12.0, size=noise.shape[0])
    noise_amp = rng.uniform(0.5, 15.0, size=noise.shape[0])
    noise_aoa = rng.uniform(-90.0, 90.0, size=noise.shape[0])
    data.append(np.column_stack([noise, noise_cf, noise_pw, noise_aoa, noise_amp]).astype(np.float32))
    labels.append(np.full(noise.shape[0], -1, dtype=np.int32))
    all_data = np.vstack(data)
    all_labels = np.concatenate(labels)
    order = rng.permutation(len(all_data))
    return all_data[order], all_labels[order]


def write_synthetic_dataset(
    data_root: str | Path,
    mode: str = "scan",
    split: str = "train",
    n_files: int = 8,
    seed: int = 42,
) -> list[Path]:
    """Write synthetic .h5 pulse trains to a dataset root."""
    root = Path(data_root) / mode / split
    root.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for idx in range(n_files):
        path = root / f"synthetic_{idx:03d}.h5"
        data, labels = make_synthetic_pulse_train(seed=seed + idx)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=data, dtype="float32")
            handle.create_dataset("labels", data=labels, dtype="int32")
        out.append(path)
    return out


def ensure_local_fallback_dataset(data_root: str | Path = "data", seed: int = 42) -> dict[str, list[Path]]:
    """Create a tiny local dataset if no real TSRD files are present."""
    root = Path(data_root)
    created: dict[str, list[Path]] = {}
    for mode in ["scan", "stare"]:
        for split in ["train", "val", "test"]:
            target = root / mode / split
            h5_files = sorted(target.glob("*.h5")) if target.exists() else []
            if h5_files:
                created.setdefault(mode, []).extend(h5_files)
                continue
            generated = write_synthetic_dataset(root, mode=mode, split=split, n_files=2 if split == "train" else 1, seed=seed + len(created))
            created.setdefault(mode, []).extend(generated)
    return created

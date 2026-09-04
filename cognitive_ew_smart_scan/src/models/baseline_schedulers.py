"""Deterministic baseline schedulers for comparison against the learned MoE.

Each baseline implements the same `act(observation) -> (action, info)` and
`step(observation) -> action` interface as `RandomScheduler`, so it can be
dropped into the evaluation harness to measure relative performance.

The population space is the canonical time-frequency joint action space:
``action = band * n_modes + mode``. Baselines select a band and emit it encoded
with the NORMAL_DWELL mode (the neutral dwell duration), so their output is a
valid flat action consumed directly by the env.

The observation is the canonical band-major, 10-feature layout: for band ``b``,
``observation[b * features_per_band : (b + 1) * features_per_band]`` holds
``[occupancy, det_rate, miss_rate, uncertainty, age, emitter_count,
deint_conf, per_stab, agility, priority]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts import NORMAL_DWELL, encode_action


def _emitted_action(n_modes: int, band: int) -> int:
    """Encode a band with the neutral NORMAL dwell mode.

    Modes are 0-indexed; when only one mode exists (legacy n_modes=1) the only
    valid mode index is 0, so the emitted flat action equals the band itself.
    """
    if n_modes:
        mode_idx = min(NORMAL_DWELL, int(n_modes) - 1)
        return encode_action(band, mode=mode_idx, n_modes=int(n_modes))
    return band


def _feature(observation: np.ndarray | Any, feature_index: int, features_per_band: int = 10) -> np.ndarray:
    """Extract a single per-band feature across all bands as a float array.

    Args:
        observation: Observation array. If 1-D it is treated as a single frame;
            if 2-D as ``(n_bands,)`` feature rows.
        feature_index: Index of the feature within each band's block.
        features_per_band: Number of features per band (canonical default 10).

    Returns:
        ``np.ndarray`` of shape ``(n_bands,)`` with the requested feature per band.
    """
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        n = obs.shape[0] // features_per_band
        return obs[np.arange(n) * features_per_band + feature_index]
    if obs.ndim == 2:
        return obs[:, feature_index]
    raise ValueError(f"Unsupported observation shape {obs.shape}")


class RoundRobinScheduler:
    """Cycles through bands in fixed order — the most naive baseline.

    Emits each band encoded with the canonical NORMAL_DWELL mode.
    """

    def __init__(self, n_bands: int = 36, n_modes: int | None = None) -> None:
        """Initialise round-robin baseline.

        Args:
            n_bands: Number of discrete bands to cycle through.
            n_modes: Dwell modes per band (defaults to canonical taxonomy).
        """
        self.n_bands = int(n_bands)
        self.n_modes = None if n_modes is None else int(n_modes)
        self._step: int = 0

    def _emitted_action(self, band: int) -> int:
        return _emitted_action(self.n_modes, band)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the next band in round-robin order.

        Args:
            observation: Unused by this baseline (kept for interface parity).

        Returns:
            Tuple ``(action, info)``.
        """
        action = self._step % self.n_bands
        self._step += 1
        return self._emitted_action(action), {"source": "round_robin", "step": self._step - 1, "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the next band action without extra metadata.

        Args:
            observation: Unused.

        Returns:
            Flat time-frequency action ``int``.
        """
        action, _ = self.act(observation)
        return action


class HighestOccupancyScheduler:
    """Picks the band with the highest occupancy (feature index 0).

    Deterministic tie-break by lowest band index. Occupancy is the canonical
    10-feature block's first field, so ``obs[::10]``.
    """

    def __init__(self, n_bands: int = 36, features_per_band: int = 10, n_modes: int | None = None) -> None:
        """Initialise occupancy-driven baseline.

        Args:
            n_bands: Number of bands (validate feature extraction capacity).
            features_per_band: Features per band (canonical 10).
            n_modes: Dwell modes per band (defaults to canonical taxonomy).
        """
        self.n_bands = int(n_bands)
        self.features_per_band = int(features_per_band)
        self.n_modes = None if n_modes is None else int(n_modes)

    def _emitted_action(self, band: int) -> int:
        return _emitted_action(self.n_modes, band)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the highest-occupancy band.

        Args:
            observation: Observation with per-band occupancy.

        Returns:
            Tuple ``(action, info)``.
        """
        occ = _feature(observation, 0, self.features_per_band)
        action = int(np.argmax(occ))
        return self._emitted_action(action), {"source": "highest_occupancy", "occupancy": float(occ[action % self.n_bands]), "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the highest-occupancy band action.

        Args:
            observation: Observation with per-band occupancy.

        Returns:
            Flat time-frequency action ``int``.
        """
        action, _ = self.act(observation)
        return action


class HighestUncertaintyScheduler:
    """Picks the band with the highest uncertainty (feature index 3).

    Deterministic tie-break by lowest band index.
    """

    def __init__(self, n_bands: int = 36, features_per_band: int = 10, n_modes: int | None = None) -> None:
        """Initialise uncertainty-driven baseline.

        Args:
            n_bands: Number of bands.
            features_per_band: Features per band (canonical 10).
            n_modes: Dwell modes per band (defaults to canonical taxonomy).
        """
        self.n_bands = int(n_bands)
        self.features_per_band = int(features_per_band)
        self.n_modes = None if n_modes is None else int(n_modes)

    def _emitted_action(self, band: int) -> int:
        return _emitted_action(self.n_modes, band)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the highest-uncertainty band.

        Args:
            observation: Observation with per-band uncertainty.

        Returns:
            Tuple ``(action, info)``.
        """
        unc = _feature(observation, 3, self.features_per_band)
        action = int(np.argmax(unc))
        return self._emitted_action(action), {"source": "highest_uncertainty", "uncertainty": float(unc[action % self.n_bands]), "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the highest-uncertainty band action.

        Args:
            observation: Observation with per-band uncertainty.

        Returns:
            Flat time-frequency action ``int``.
        """
        action, _ = self.act(observation)
        return action
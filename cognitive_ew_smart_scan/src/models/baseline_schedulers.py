"""Deterministic baseline schedulers for comparison against the learned MoE.

Each baseline implements the same `act(observation) -> (action, info)` and
`step(observation) -> action` interface as `RandomScheduler`, so it can be
dropped into the evaluation harness to measure relative performance.

The observation is the canonical band-major, 10-feature layout: for band ``b``,
``observation[b * features_per_band : (b + 1) * features_per_band]`` holds
``[occupancy, det_rate, miss_rate, uncertainty, age, emitter_count,
deint_conf, per_stab, agility, priority]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    """Cycles through bands in fixed order — the most naive baseline."""

    def __init__(self, n_bands: int = 36) -> None:
        """Initialise round-robin baseline.

        Args:
            n_bands: Number of discrete bands to cycle through.
        """
        self.n_bands = int(n_bands)
        self._step: int = 0

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the next band in round-robin order.

        Args:
            observation: Unused by this baseline (kept for interface parity).

        Returns:
            Tuple ``(action, info)``.
        """
        action = self._step % self.n_bands
        self._step += 1
        return action, {"source": "round_robin", "step": self._step - 1, "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the next band index without extra metadata.

        Args:
            observation: Unused.

        Returns:
            Band index ``int``.
        """
        action, _ = self.act(observation)
        return action


class HighestOccupancyScheduler:
    """Picks the band with the highest occupancy (feature index 0).

    Deterministic tie-break by lowest band index. Occupancy is the canonical
    10-feature block's first field, so ``obs[::10]``.
    """

    def __init__(self, n_bands: int = 36, features_per_band: int = 10) -> None:
        """Initialise occupancy-driven baseline.

        Args:
            n_bands: Number of bands (validate feature extraction capacity).
            features_per_band: Features per band (canonical 10).
        """
        self.n_bands = int(n_bands)
        self.features_per_band = int(features_per_band)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the highest-occupancy band.

        Args:
            observation: Observation with per-band occupancy.

        Returns:
            Tuple ``(action, info)``.
        """
        occ = _feature(observation, 0, self.features_per_band)
        action = int(np.argmax(occ))
        return action, {"source": "highest_occupancy", "occupancy": float(occ[action]), "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the highest-occupancy band index.

        Args:
            observation: Observation with per-band occupancy.

        Returns:
            Band index ``int``.
        """
        action, _ = self.act(observation)
        return action


class HighestUncertaintyScheduler:
    """Picks the band with the highest uncertainty (feature index 3).

    Deterministic tie-break by lowest band index.
    """

    def __init__(self, n_bands: int = 36, features_per_band: int = 10) -> None:
        """Initialise uncertainty-driven baseline.

        Args:
            n_bands: Number of bands.
            features_per_band: Features per band (canonical 10).
        """
        self.n_bands = int(n_bands)
        self.features_per_band = int(features_per_band)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        """Select the highest-uncertainty band.

        Args:
            observation: Observation with per-band uncertainty.

        Returns:
            Tuple ``(action, info)``.
        """
        unc = _feature(observation, 3, self.features_per_band)
        action = int(np.argmax(unc))
        return action, {"source": "highest_uncertainty", "uncertainty": float(unc[action]), "observation": observation}

    def step(self, observation: Any) -> int:
        """Return the highest-uncertainty band index.

        Args:
            observation: Observation with per-band uncertainty.

        Returns:
            Band index ``int``.
        """
        action, _ = self.act(observation)
        return action
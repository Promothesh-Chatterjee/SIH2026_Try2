"""Baseline sweep schedulers for Electronic Warfare receiver control.

Implements standard classical baselines:
- RoundRobinScheduler: Sequential systematic band / mode sweeping.
- RandomScheduler: Uniform random exploration across bands / modes.
- Action conversion utilities between joint action space and (band, mode) pairs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    NORMAL_DWELL,
    band_of_action,
    encode_action,
    mode_of_action,
    n_actions_for,
)


def action_to_band(action: int, n_modes: int = CANONICAL_N_MODES) -> int:
    """Decode flat joint action index to frequency band index."""
    return band_of_action(action, n_modes=n_modes)


def action_to_mode(action: int, n_modes: int = CANONICAL_N_MODES) -> int:
    """Decode flat joint action index to dwell mode index."""
    return mode_of_action(action, n_modes=n_modes)


def band_mode_to_action(
    band: int,
    mode: int = NORMAL_DWELL,
    n_modes: int = CANONICAL_N_MODES,
) -> int:
    """Encode (band, mode) pair into flat joint action index."""
    return encode_action(band=band, mode=mode, n_modes=n_modes)


class RoundRobinScheduler:
    """Sequential sweep scheduler that cycles across frequency bands in fixed order.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands across the receiver spectrum. Default is 36.
    n_modes : int, optional
        Number of dwell modes. Default is 5.
    joint : bool
        If True, emits flat joint actions (band * n_modes + mode) where mode defaults
        to NORMAL_DWELL. If False, emits band indices directly in [0, n_bands).
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: Optional[int] = CANONICAL_N_MODES,
        joint: bool = False,
    ) -> None:
        self.n_bands = max(1, int(n_bands))
        self.n_modes = CANONICAL_N_MODES if n_modes is None else max(1, int(n_modes))
        self.joint = bool(joint)
        self._current_step: int = 0

    def reset(self) -> None:
        """Reset internal step counter to band 0."""
        self._current_step = 0

    def act(self, observation: Any = None) -> Tuple[int, Dict[str, Any]]:
        """Select next band/action in cyclic sequence.

        Parameters
        ----------
        observation : Any, optional
            Ignored; retained for interface compliance.

        Returns
        -------
        action : int
            Selected action.
        info : dict
            Diagnostic metadata.
        """
        band = self._current_step % self.n_bands
        self._current_step += 1

        if self.joint:
            action = band_mode_to_action(band=band, mode=NORMAL_DWELL, n_modes=self.n_modes)
        else:
            action = band

        info = {
            "source": "round_robin",
            "band": band,
            "step": self._current_step - 1,
        }
        return action, info

    def step(self, observation: Any = None) -> int:
        """Return next action directly."""
        action, _ = self.act(observation)
        return action


class RandomScheduler:
    """Uniform random scheduler over frequency bands or joint actions.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands. Default is 36.
    n_modes : int, optional
        Number of dwell modes. Default is 5.
    joint : bool
        If True, emits flat joint action in [0, n_bands * n_modes).
        If False, emits band index in [0, n_bands).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: Optional[int] = CANONICAL_N_MODES,
        joint: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.n_bands = max(1, int(n_bands))
        self.n_modes = CANONICAL_N_MODES if n_modes is None else max(1, int(n_modes))
        self.joint = bool(joint)
        self._rng = random.Random(seed)

    def act(self, observation: Any = None) -> Tuple[int, Dict[str, Any]]:
        """Sample an action uniformly at random."""
        if self.joint:
            n_actions = n_actions_for(self.n_bands, self.n_modes)
            action = self._rng.randrange(n_actions)
            band = action_to_band(action, self.n_modes)
            mode = action_to_mode(action, self.n_modes)
        else:
            band = self._rng.randrange(self.n_bands)
            action = band
            mode = NORMAL_DWELL

        info = {
            "source": "random",
            "band": band,
            "mode": mode,
        }
        return action, info

    def step(self, observation: Any = None) -> int:
        """Return random action directly."""
        action, _ = self.act(observation)
        return action

"""Simple random scheduler used to validate the receiver-driven RF loop."""

from __future__ import annotations

import random
from typing import Any

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, n_actions_for


class RandomScheduler:
    """Minimal scheduler for the non-trained path.

    Returns a random discrete action index for the current observation.

    The population space is the canonical time-frequency joint action space
    (``n_bands * n_modes``).
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_bands = int(n_bands)
        # Keep the historical explicit ``n_bands=action_space.n`` call valid.
        if n_modes is None and self.n_bands != CANONICAL_N_BANDS:
            self.n_modes = 1
            self.n_actions = self.n_bands
        else:
            self.n_modes = CANONICAL_N_MODES if n_modes is None else int(n_modes)
            self.n_actions = n_actions_for(self.n_bands, self.n_modes)
        self.rng = random.Random(seed)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        action = self.rng.randrange(self.n_actions)
        return action, {"source": "random", "observation": observation}

    def step(self, observation: Any) -> int:
        action, _ = self.act(observation)
        return action

"""Simple random scheduler used to validate the receiver-driven RF loop."""

from __future__ import annotations

import random
from typing import Any


class RandomScheduler:
    """Minimal scheduler for the non-trained path.

    Returns a random discrete action index for the current observation.
    """

    def __init__(self, n_bands: int = 36, seed: int | None = None) -> None:
        self.n_bands = int(n_bands)
        self.rng = random.Random(seed)

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]:
        action = self.rng.randrange(self.n_bands)
        return action, {"source": "random", "observation": observation}

    def step(self, observation: Any) -> int:
        action, _ = self.act(observation)
        return action

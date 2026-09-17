"""Simulator adapter for RF environment interfacing.

Supports zero-dependency "python" simulation mode as well as future external/SDR interfaces.
"""

from typing import Any, Dict, Optional
from .spectrum_env import SpectrumEnvironment


class SimulatorAdapter:
    """Adapter decoupling the scan scheduler from the specific underlying RF simulator engine."""

    def __init__(
        self,
        mode: str = "python",
        n_bands: int = 32,
        t_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        self.mode = mode.lower()
        self.n_bands = int(n_bands)
        self.t_steps = int(t_steps)
        self.kwargs = kwargs
        self._env: Optional[SpectrumEnvironment] = None

        if self.mode == "python":
            self._env = SpectrumEnvironment(
                n_bands=self.n_bands,
                t_steps=self.t_steps,
                **self.kwargs,
            )
        else:
            raise ValueError(f"Unsupported simulation mode '{self.mode}'. Currently supported: ['python']")

    def get_env(self) -> SpectrumEnvironment:
        """Return the underlying Gymnasium environment instance."""
        if self._env is None:
            raise RuntimeError("Simulator environment has not been initialized.")
        return self._env

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the underlying environment."""
        return self.get_env().reset(*args, **kwargs)

    def step(self, action: int) -> Any:
        """Step the underlying environment."""
        return self.get_env().step(action)

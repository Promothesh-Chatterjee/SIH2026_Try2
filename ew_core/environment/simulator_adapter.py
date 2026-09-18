"""Simulator adapter for RF environment interfacing.

Supports zero-dependency "python" simulation mode as well as "gnuradio" mode
reading energy measurements from a GNU Radio ZeroMQ socket.
"""

from typing import Any, Dict, Optional
import numpy as np
import gymnasium

from .spectrum_env import SpectrumEnvironment


class GNURadioSpectrumEnvironment(SpectrumEnvironment):
    """SpectrumEnvironment reading live energy vectors from GNU Radio via ZeroMQ."""

    def __init__(
        self,
        n_bands: int = 32,
        t_steps: int = 500,
        zmq_endpoint: str = "tcp://127.0.0.1:55556",
        timeout_ms: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_bands=n_bands, t_steps=t_steps, **kwargs)
        self.zmq_endpoint = zmq_endpoint
        self.timeout_ms = int(timeout_ms)
        self._zmq_context: Optional[Any] = None
        self._zmq_socket: Optional[Any] = None

    def _init_zmq(self) -> None:
        """Initialize ZMQ subscriber if pyzmq is available."""
        if self._zmq_socket is None:
            try:
                import zmq

                self._zmq_context = zmq.Context.instance()
                self._zmq_socket = self._zmq_context.socket(zmq.SUB)
                self._zmq_socket.connect(self.zmq_endpoint)
                self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            except Exception:
                self._zmq_socket = None

    def _compute_observation(self, t: int) -> np.ndarray:
        """Read energy measurement vector from GNU Radio ZMQ socket with fallback."""
        self._init_zmq()
        if self._zmq_socket is not None:
            try:
                raw_bytes = self._zmq_socket.recv()
                energy_arr = np.frombuffer(raw_bytes, dtype=np.float32)
                if len(energy_arr) == self.n_bands:
                    return energy_arr
            except Exception:
                # Timeout or connection error; fall back to baseline computation
                pass

        return super()._compute_observation(t)

    def close(self) -> None:
        """Close ZMQ subscriber socket."""
        if self._zmq_socket is not None:
            try:
                self._zmq_socket.close()
            except Exception:
                pass
            self._zmq_socket = None
        super().close()


class SimulatorAdapter:
    """Adapter decoupling the scan scheduler from the specific underlying RF simulator engine.

    Modes
    -----
    "python" : Pure Python emitter models and synthetic RF spectrum environment.
               Zero external dependencies, ideal for high-throughput RL training.
    "gnuradio" : High-fidelity RF simulation stream reading energy measurements from
                 the GNU Radio band energy estimator over ZeroMQ.
    """

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
        elif self.mode == "gnuradio":
            self._env = GNURadioSpectrumEnvironment(
                n_bands=self.n_bands,
                t_steps=self.t_steps,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported simulation mode '{self.mode}'. Currently supported: ['python', 'gnuradio']"
            )

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

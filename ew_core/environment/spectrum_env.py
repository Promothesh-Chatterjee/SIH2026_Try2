"""SpectrumEnvironment: Gymnasium-compatible RF scan environment.

Implements the standard DRDO SIH 2026 Electronic Support (ES) receiver scheduler environment:
- Wide frequency spectrum partitioned into N frequency bands
- Single channel receiver that tunes to one band per time slot
- No prior intelligence about emitter characteristics (strictly no leak of parameters)
- Returns standard (obs, reward, terminated, truncated, info)
- Provides get_truth_matrix() as the sanctioned backchannel for metrics and verification
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium
from gymnasium import spaces

from .emitter_models import (
    BaseEmitter,
    StaticEmitter,
    FreqAgileEmitter,
    PeriodicScanEmitter,
)


class SpectrumEnvironment(gymnasium.Env):
    """Gymnasium environment simulating an RF spectrum with agile and scanning emitters."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_bands: int = 32,
        t_steps: int = 500,
        emitter_configs: Optional[List[BaseEmitter]] = None,
        noise_std: float = 0.1,
        snr_db: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.t_steps = int(t_steps)
        self.noise_std = float(noise_std)
        self.snr_db = float(snr_db)

        # Action space: receiver tunes to discrete frequency band 0..n_bands-1
        self.action_space = spaces.Discrete(self.n_bands)

        # Observation space: RF energy detected across all bands (or tuned band)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.n_bands,),
            dtype=np.float32,
        )

        self._emitter_configs = emitter_configs
        self._emitters: List[BaseEmitter] = []
        self._t = 0
        self._rng = np.random.default_rng()
        self._signal_power = float(10.0 ** (self.snr_db / 10.0) * (self.noise_std ** 2))
        if self._signal_power <= 0.0:
            self._signal_power = 1.0

        self._current_log: Dict[str, list] = self.init_episode_log()

    def _init_default_emitters(self) -> List[BaseEmitter]:
        """Create a default configuration of emitters across the spectrum."""
        b1 = max(0, min(self.n_bands - 1, self.n_bands // 4))
        b2 = max(0, min(self.n_bands - 1, self.n_bands // 2))
        hop_set = [b2, (b2 + 1) % self.n_bands, (b2 + 2) % self.n_bands]
        scan_pattern = [0, 1, 2, 3] if self.n_bands >= 4 else [0]

        return [
            StaticEmitter(band_idx=b1, duty_cycle=0.5, pri=2, emitter_id=1),
            FreqAgileEmitter(hop_set=hop_set, hop_interval=5, pattern="fixed", emitter_id=2),
            PeriodicScanEmitter(scan_period=20, dwell_time=3, scan_pattern=scan_pattern, emitter_id=3),
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to t=0 and initialize emitters."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        if self._emitter_configs is not None:
            self._emitters = list(self._emitter_configs)
        else:
            self._emitters = self._init_default_emitters()

        self._current_log = self.init_episode_log()
        obs = self._compute_observation(self._t)
        info = {
            "step": self._t,
            "n_emitters": len(self._emitters),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one receiver scan step.

        Parameters
        ----------
        action : int
            Band index tuned by the receiver.

        Returns
        -------
        obs : np.ndarray
            Observation energy vector for the next time step.
        reward : float
            1.0 if receiver intercepted an active emitter, else 0.0.
        terminated : bool
            True if t reached t_steps.
        truncated : bool
            False.
        info : dict
            Diagnostic dictionary containing "hit", "active_bands", "emitter_ids", and "step".
        """
        t = self._t
        active_bands: List[int] = []
        emitter_ids: List[int] = []

        for emitter in self._emitters:
            if emitter.step(t):
                b = emitter.get_band(t)
                active_bands.append(b)
                emitter_ids.append(emitter.emitter_id)

        hit = int(action) in active_bands
        reward = 1.0 if hit else 0.0

        info = {
            "hit": hit,
            "active_bands": active_bands,
            "emitter_ids": emitter_ids,
            "step": t,
        }

        self._t += 1
        terminated = bool(self._t >= self.t_steps)
        truncated = False

        obs_t = self._t if not terminated else (self._t - 1)
        obs = self._compute_observation(obs_t)

        return obs, reward, terminated, truncated, info

    def _compute_observation(self, t: int) -> np.ndarray:
        """Synthesize observation vector across all bands at time t."""
        # Folded normal / absolute noise floor
        obs = np.abs(self._rng.normal(0.0, self.noise_std, size=self.n_bands)).astype(np.float32)

        for emitter in self._emitters:
            if emitter.step(t):
                b = emitter.get_band(t)
                if 0 <= b < self.n_bands:
                    obs[b] += float(self._signal_power)

        return obs

    def get_truth_matrix(self) -> np.ndarray:
        """Sanctioned backchannel: returns ground-truth spectrum occupancy matrix of shape (n_bands, t_steps)."""
        truth = np.zeros((self.n_bands, self.t_steps), dtype=bool)
        for t in range(self.t_steps):
            for emitter in self._emitters:
                if emitter.step(t):
                    b = emitter.get_band(t)
                    if 0 <= b < self.n_bands:
                        truth[b, t] = True
        return truth

    def init_episode_log(self) -> Dict[str, list]:
        """Initialize telemetry/metrics tracker for an episode."""
        return {
            "hits": [],
            "chosen_bands": [],
            "active_bands_per_step": [],
            "rewards": [],
        }

    def update_episode_log(
        self, log: Dict[str, list], action: int, reward: float, info: Dict[str, Any]
    ) -> None:
        """Record step metrics in the provided episode log."""
        log["hits"].append(info.get("hit", False))
        log["chosen_bands"].append(action)
        log["active_bands_per_step"].append(info.get("active_bands", []))
        if "rewards" in log:
            log["rewards"].append(reward)

    def get_episode_log(self) -> Dict[str, list]:
        """Retrieve current episode log."""
        return self._current_log


# Register with Gymnasium
try:
    if "SmartScanEW-SimpleSim-v0" not in gymnasium.envs.registry:
        gymnasium.register(
            id="SmartScanEW-SimpleSim-v0",
            entry_point="ew_core.environment.spectrum_env:SpectrumEnvironment",
        )
    if "SmartScanEW-v0" not in gymnasium.envs.registry:
        gymnasium.register(
            id="SmartScanEW-v0",
            entry_point="ew_core.environment.cognitive_rf_scan_env:CognitiveRFScanEnv",
        )
    else:
        gymnasium.envs.registry["SmartScanEW-v0"].entry_point = "ew_core.environment.cognitive_rf_scan_env:CognitiveRFScanEnv"
except Exception:
    try:
        gymnasium.register(
            id="SmartScanEW-SimpleSim-v0",
            entry_point="ew_core.environment.spectrum_env:SpectrumEnvironment",
        )
    except Exception:
        pass
    try:
        gymnasium.register(
            id="SmartScanEW-v0",
            entry_point="ew_core.environment.cognitive_rf_scan_env:CognitiveRFScanEnv",
        )
    except Exception:
        pass


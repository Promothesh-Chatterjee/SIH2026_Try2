"""
LEGACY Gymnasium Environment for RF Scanning Scheduler.

Deprecated in favour of CognitiveRFScanEnv (src.environment.cognitive_rf_scan_env),
which is the single canonical environment for all training, evaluation and API
pipelines. This module exists only for backward compatibility (tests, docs, old
checkpoints) and must not be used to train new policies.

Obsolete contract (superseded):
  Obs: [occupancy_per_band (0/1), normalised_time_since_last_visit] shape (2*n_bands,)
  Action: Discrete(n_bands)
New canonical contract:
  Obs: (36, 10) = 360 feature vector per band (see checkpoint_meta.FEATURE_ORDER)
  Action: Discrete(n_bands * n_modes) time-frequency (band, dwell-mode) select
"""

import logging
import random
import warnings
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    from typing import Any

    PulseTrain = Any  # type: ignore

from .state_matrix import build_transmission_matrix, get_pdws_in_band
from ..evaluation.metrics import FiguresOfMerit
from ..training.reward import compute_reward

logger = logging.getLogger(__name__)


class LegacyRFScanEnv(gym.Env):
    """LEGACY Cognitive EW Gymnasium environment for frequency-scanning ES receiver.

    Deprecated: use CognitiveRFScanEnv instead. Construction emits a DeprecationWarning
    and synthetic fallback data (when no TSRD .h5 files exist) is only produced when
    ``allow_synthetic_fallback=True`` is passed explicitly. Use for validation of old
    artifacts only — never for new training runs.

    The agent chooses a band to monitor for dwell_slots time slots. Reward is
    shaped via compute_reward (novel intercepts, timing penalty, miss penalty).
    Tracks FiguresOfMerit internally; exposes via get_fom().

    Attributes:
        observation_space: Box(0,1, shape=(2*n_bands,)) dtype float32.
        action_space: Discrete(n_bands).
    """

    LEGACY = True
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict,
        data_dir: str | Path = "data",
        subset: str = "train",
        mode: str = "scan",
        seed: int | None = 42,
        allow_synthetic_fallback: bool = False,
    ) -> None:
        """Initialise environment.

        Args:
            config: Dict with keys n_bands, freq_min_mhz, freq_max_mhz,
                time_resolution_us, dwell_slots, w1..w4.
            data_dir: Root data directory containing stare/scan splits.
            subset: train/val/test.
            mode: stare (oracle) or scan (realistic).
            seed: RNG seed.
            allow_synthetic_fallback: If True, generate synthetic PulseTrain when
                no .h5 files are found (for unit tests of legacy behaviour only).

        Raises:
            FileNotFoundError: If data_dir/subset/mode has no .h5 files and
                allow_synthetic_fallback is False.
        """
        super().__init__()
        warnings.warn(
            "LegacyRFScanEnv is deprecated; use CognitiveRFScanEnv (canonical "
            "36-band x 10-feature x time-frequency Discrete(n_bands*n_modes) contract).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.allow_synthetic_fallback = bool(allow_synthetic_fallback)
        self.n_bands: int = int(config.get("n_bands", 36))
        self.freq_min: float = float(config.get("freq_min_mhz", 0.0))
        self.freq_max: float = float(config.get("freq_max_mhz", 18000.0))
        self.time_resolution: float = float(config.get("time_resolution_us", 100.0))
        self.dwell_slots: int = int(config.get("dwell_slots", 5))
        self.dwell_time_us: float = self.dwell_slots * self.time_resolution

        self.w1: float = float(config.get("w1", 5.0))
        self.w2: float = float(config.get("w2", 8.0))
        self.w3: float = float(config.get("w3", 0.1))
        self.w4: float = float(config.get("w4", 4.0))

        self.data_dir = Path(data_dir) / mode / subset
        self.mode = mode
        self.subset = subset

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.n_bands,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_bands)

        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # Discover .h5 files lazily to avoid OOM
        self._files: list[Path] = []
        if self.data_dir.exists():
            self._files = sorted(self.data_dir.glob("*.h5"))
        if not self._files:
            logger.warning("No .h5 files found in %s — legacy env has no data", self.data_dir)

        # Episode state
        self.current_pt: PulseTrain | None = None
        self.transmission_matrix: np.ndarray | None = None
        self.current_slot: int = 0
        self.max_slots: int = 0
        self.current_time_us: float = 0.0
        self.min_toa: float = 0.0
        self.time_since_visit: np.ndarray = np.zeros(self.n_bands, dtype=np.float32)
        self.occupancy_estimate: np.ndarray = np.zeros(self.n_bands, dtype=np.float32)
        self.intercepted_emitters: set[int] = set()
        self.fom = FiguresOfMerit()
        self.band_edges: np.ndarray = np.linspace(self.freq_min, self.freq_max, self.n_bands + 1)

        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _load_random_pt(self) -> PulseTrain | None:
        """Load a random PulseTrain from disk (lazy).

        Returns:
            PulseTrain or None if no files / load fails.
        """
        if not self._files:
            return None
        # Lazy import to avoid hard dependency at import time
        try:
            from turing_deinterleaving_challenge import PulseTrain as PT  # type: ignore
        except ImportError:
            logger.warning("turing_deinterleaving_challenge not installed")
            return None

        # Pick random file
        fpath = self._py_rng.choice(self._files)
        try:
            pt = PT.load(str(fpath))
            logger.debug("Loaded %s with %d pulses", fpath, len(pt))
            return pt
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fpath, exc)
            # Try another file
            for alt in self._files[:3]:
                try:
                    return PT.load(str(alt))
                except Exception:
                    continue
            return None

    def _make_synthetic_pt(self) -> Any:
        """Create minimal synthetic PulseTrain for testing without dataset."""
        # Simple container mimicking PulseTrain interface
        class _SynPT:
            def __init__(self) -> None:
                rng = np.random.default_rng(0)
                n = 200
                toa = np.sort(rng.uniform(0, 50000, size=n))
                cf = rng.uniform(0, 18000, size=n)
                pw = rng.uniform(0.5, 10, size=n)
                aoa = rng.uniform(-60, 60, size=n)
                amp = rng.uniform(0, 60, size=n)
                self.data = np.column_stack([toa, cf, pw, aoa, amp]).astype(np.float32)
                self.labels = rng.integers(0, 5, size=n).astype(np.int32)

            def __len__(self) -> int:
                return self.data.shape[0]

        return _SynPT()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment to new random pulse train.

        Args:
            seed: Optional new RNG seed.
            options: Unused.

        Returns:
            Tuple (observation (2*n_bands,) float32, info dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._py_rng = random.Random(seed)
            np.random.seed(seed)
            random.seed(seed)

        pt = self._load_random_pt()
        if pt is None:
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"No PulseTrain available from {self.data_dir} and synthetic fallback "
                    "is disabled. Use CognitiveRFScanEnv with a valid TSRD split, or pass "
                    "allow_synthetic_fallback=True for legacy unit tests."
                )
            logger.warning("Legacy env: using SYNTHETIC PulseTrain (explicit opt-in)")
            pt = self._make_synthetic_pt()

        self.current_pt = pt
        # Handle empty gracefully
        try:
            if len(pt) == 0 or pt.data.size == 0:
                self.transmission_matrix = np.zeros((1, self.n_bands), dtype=np.int8)
                self.min_toa = 0.0
                self.max_slots = 1
            else:
                self.min_toa = float(np.min(pt.data[:, 0]))
                self.transmission_matrix = build_transmission_matrix(
                    pt, self.n_bands, self.time_resolution, self.freq_min, self.freq_max
                )
                self.max_slots = self.transmission_matrix.shape[0]
        except Exception as exc:
            logger.warning("Failed to build transmission matrix: %s", exc)
            self.transmission_matrix = np.zeros((1, self.n_bands), dtype=np.int8)
            self.min_toa = 0.0
            self.max_slots = 1

        self.current_slot = 0
        self.current_time_us = self.min_toa
        self.time_since_visit = np.zeros(self.n_bands, dtype=np.float32)
        self.occupancy_estimate = np.zeros(self.n_bands, dtype=np.float32)
        self.intercepted_emitters = set()
        self.fom.reset()

        obs = self._get_obs()
        info: dict = {}
        return obs.astype(np.float32), info

    def _get_obs(self) -> np.ndarray:
        """Build observation: [occupancy_estimate, normalised_time_since_visit].

        Returns:
            Array shape (2*n_bands,) float32 in [0,1].
        """
        # Normalise time_since to [0,1] by dividing by max seen + eps
        max_t = float(np.max(self.time_since_visit)) if np.max(self.time_since_visit) > 0 else 1.0
        time_norm = (self.time_since_visit / max_t).astype(np.float32)
        # Clip occupancy estimate already 0/1
        occ = np.clip(self.occupancy_estimate, 0, 1).astype(np.float32)
        return np.concatenate([occ, time_norm]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance dwell_slots steps, return PDWs in chosen band, compute reward.

        Args:
            action: Band index to tune to.

        Returns:
            Tuple (next_obs, reward, terminated, truncated, info).
            info contains keys: pdws, labels, hit, intercept_time_error_us,
            missed_opportunity, ground_truth_active.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert self.current_pt is not None and self.transmission_matrix is not None

        t_start = self.min_toa + self.current_slot * self.time_resolution
        t_end = t_start + self.dwell_time_us

        # Ground truth active bands during dwell (oracle from matrix)
        slot_end = min(self.current_slot + self.dwell_slots, self.max_slots)
        if self.current_slot < self.max_slots:
            gt_active = np.any(self.transmission_matrix[self.current_slot : slot_end, :], axis=0).astype(np.int8)
        else:
            gt_active = np.zeros(self.n_bands, dtype=np.int8)

        # Observation returned to agent: PDWs in chosen band
        try:
            pdws, labels = get_pdws_in_band(
                self.current_pt, int(action), float(t_start), float(t_end), self.n_bands, self.freq_min, self.freq_max
            )
        except Exception as exc:
            logger.warning("get_pdws_in_band failed: %s", exc)
            pdws = np.empty((0, 5), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)

        hit = bool(pdws.shape[0] > 0)
        missed_opportunity = bool(np.any(gt_active) and not hit)

        # Intercept time error: |first pulse ToA - t_start| if hit else 0
        intercept_time_error_us = 0.0
        if hit:
            first_toa = float(np.min(pdws[:, 0]))
            intercept_time_error_us = abs(first_toa - float(t_start))

        # Compute shaped reward
        reward, new_emitters = compute_reward(
            hit=hit,
            labels_intercepted=labels,
            intercepted_emitters=self.intercepted_emitters,
            missed_opportunity=missed_opportunity,
            intercept_time_error_us=intercept_time_error_us,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
        )
        self.intercepted_emitters.update(new_emitters)

        # Update belief state: occupancy estimate from actual observation
        # Simple: set chosen band's occupancy to hit, decay others slightly
        self.occupancy_estimate[int(action)] = 1.0 if hit else 0.0
        # Time since last visit
        self.time_since_visit += float(self.dwell_slots)
        self.time_since_visit[int(action)] = 0.0

        # Update FoM
        self.fom.update(
            band_chosen=int(action),
            ground_truth_active=gt_active,
            pred_active=hit,
            intercept_time_error_us=intercept_time_error_us,
            reward=float(reward),
        )

        # Advance time
        self.current_slot += self.dwell_slots
        self.current_time_us = t_end

        terminated = bool(self.current_slot >= self.max_slots)
        truncated = False

        next_obs = self._get_obs()
        info = {
            "pdws": pdws,
            "labels": labels,
            "hit": hit,
            "band_chosen": int(action),
            "intercept_time_error_us": float(intercept_time_error_us),
            "missed_opportunity": missed_opportunity,
            "ground_truth_active": gt_active,
            "new_emitters": new_emitters,
        }
        return next_obs.astype(np.float32), float(reward), terminated, truncated, info

    def get_fom(self) -> dict[str, float]:
        """Return current FiguresOfMerit summary dict.

        Returns:
            Dict from FiguresOfMerit.summary().
        """
        return self.fom.summary()

    def render(self, mode: str = "human") -> None:
        """Print terminal visualisation of current spectrum state.

        Args:
            mode: Only 'human' supported.
        """
        if mode != "human":
            return
        # Simple ASCII bar per band showing visit age and occupancy
        bar_len = 40
        lines = []
        lines.append(f"=== RFScanEnv t={self.current_time_us:.0f}us slot={self.current_slot}/{self.max_slots} ===")
        for b in range(min(self.n_bands, 20)):  # show first 20 for brevity
            occ = "█" if self.occupancy_estimate[b] > 0.5 else "·"
            age = int(self.time_since_visit[b])
            age_bar = "#" * min(bar_len, age // 5)
            lines.append(f"band {b:03d} [{occ}] age={age:4d} {age_bar}")
        if self.n_bands > 20:
            lines.append(f"... ({self.n_bands-20} more bands)")
        fom = self.fom.summary()
        lines.append(f"Pd={fom['Pd']:.3f} Pfa={fom['Pfa']:.3f} hits={fom['n_hits']:.0f}/{fom['n_steps']:.0f}")
        print("\n".join(lines))


# Backward-compatible alias for legacy references (tests, docs, old dashboards).
RFScanEnv = LegacyRFScanEnv

"""
Master Cognitive RF Scan Environment (gym.Env).

Connects: RF Scenario -> RadioEnvironment -> SieveReceiver -> Perception -> Belief -> Reward.

ML-clock driven control loop:
    reset()
      ↓
    scheduler chooses action (receiver center-frequency / STEP / DWELL)
      ↓
    receiver.execute_action(action)          # tune/step/dwell
      ↓
    env advances RadioEnvironment events through the dwell          # world evolves
      ↓
    receiver.get_observation()               # causal, no future leak
      ↓
    perception (deinterleaver + emitter tracker) processes detections
      ↓
    belief.update(perception)                # occupancy, revisit age, ...
      ↓
    reward via ground-truth only for shaping  (kept out of observation)
      ↓
    return (obs_vec, reward, terminated, truncated, info)

GROUND TRUTH (emitter_id, gt_active) is used ONLY for reward shaping and
evaluation/info. It is NEVER included in the observation vector returned to the
policy. This is enforced by construction: `_build_observation` builds only from
the ReceiverObservation + belief fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.receiver import SieveReceiver, ReceiverObservation
from src.environment.radio_environment import ActivePulse, PulseRecord, RadioEnvironment, SimulationEvent
from src.evaluation.metrics import FiguresOfMerit
from src.perception import EmitterTracker, build_band_belief_from_tracks
from src.cognitive.memory import SemanticMemory, EmitterProfile
from src.cognitive.periodic_interceptor import PeriodicScanInterceptor

logger = logging.getLogger(__name__)


STATE_FEATURES_PER_BAND = 10


@dataclass
class BeliefState:
    """Mutable cognitive belief derived from receiver observations only.

    Canonical 10 features per band:
    1. current/estimated occupancy (EMA of hit indicator)
    2. recent detection/hit rate (hits / visits)
    3. recent miss rate (1 - detection_rate)
    4. uncertainty (peaked at 0.5 occupancy or unvisited)
    5. time since last visit (normalized revisit age)
    6. estimated emitter count (observable diversity in band)
    7. deinterleaver confidence (clustering confidence proxy)
    8. PRI/periodicity stability (PRI coefficient of variation inverse)
    9. frequency-agility indicator (intra-band frequency dispersion)
    10. risk/priority score (composite cognitive urgency)
    """

    def __init__(self, n_bands: int = 36):
        self.n_bands = n_bands
        self.reset()

    def reset(self) -> None:
        n = self.n_bands
        self.occupancy_prob = np.zeros(n, dtype=np.float32)
        self.detection_rate = np.zeros(n, dtype=np.float32)
        self.revisit_age = np.ones(n, dtype=np.int64)
        self.uncertainty = np.ones(n, dtype=np.float32)  # max uncertainty when no data
        self.estimated_emitter_count = np.zeros(n, dtype=np.float32)
        self.deinterleaver_confidence = np.zeros(n, dtype=np.float32)
        self.periodicity_stability = np.zeros(n, dtype=np.float32)
        self.agility_indicator = np.zeros(n, dtype=np.float32)
        self.priority_score = np.full(n, 0.5, dtype=np.float32)
        self._visits = np.zeros(n, dtype=np.int64)
        self._hits = np.zeros(n, dtype=np.int64)
        self._last_visit_slot = np.zeros(n, dtype=np.int64)
        self._band_pulse_history: list[list[float]] = [[] for _ in range(n)]

    def record_visit(self, band: int, hit: bool, detections: Sequence[Any] | None = None, ema_alpha: float = 0.3) -> None:
        band = int(band)
        if not (0 <= band < self.n_bands):
            return
        self._visits[band] += 1
        if hit:
            self._hits[band] += 1
        self.detection_rate[band] = float(self._hits[band] / max(1, self._visits[band]))
        target = 1.0 if hit else 0.0
        self.occupancy_prob[band] = self.occupancy_prob[band] * (1.0 - ema_alpha) + float(target) * ema_alpha

        # Update physical observable features from detected pulses (zero truth leakage)
        if detections and len(detections) > 0:
            toas = [float(getattr(d, "time_us", 0.0)) for d in detections]
            freqs = [float(getattr(d, "frequency_mhz", 0.0)) for d in detections]

            # 8. Periodicity stability via PRI consistency
            if len(toas) >= 2:
                toas_sorted = sorted(toas)
                pris = np.diff(toas_sorted)
                mean_pri = float(np.mean(pris))
                std_pri = float(np.std(pris))
                cv = float(std_pri / max(mean_pri, 1e-6))
                self.periodicity_stability[band] = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
            else:
                self._band_pulse_history[band].extend(toas[-5:])
                if len(self._band_pulse_history[band]) >= 2:
                    pris = np.diff(sorted(self._band_pulse_history[band][-5:]))
                    cv = float(np.std(pris) / max(float(np.mean(pris)), 1e-6))
                    self.periodicity_stability[band] = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

            # 9. Agility indicator via frequency dispersion within the band
            if len(freqs) >= 2:
                std_f = float(np.std(freqs))
                self.agility_indicator[band] = float(np.clip(std_f / 100.0, 0.0, 1.0))

            # 6. Estimated emitter count & 7. Deinterleaver confidence proxy from AoA / PW
            aoas = [float(getattr(d, "aoa_deg", 0.0)) for d in detections]
            unique_bearings = len(set(int(a / 15.0) for a in aoas))
            self.estimated_emitter_count[band] = float(np.clip(unique_bearings / 5.0, 0.1, 1.0))
            self.deinterleaver_confidence[band] = float(np.clip(0.6 + 0.08 * min(len(detections), 5), 0.0, 1.0))
        elif hit:
            self.estimated_emitter_count[band] = float(np.clip(self.estimated_emitter_count[band] * 0.9 + 0.1, 0.0, 1.0))
        else:
            self.periodicity_stability[band] *= 0.95
            self.agility_indicator[band] *= 0.95
            self.estimated_emitter_count[band] *= 0.9
            self.deinterleaver_confidence[band] *= 0.9

    def update_from_perception(self, perception_result: dict) -> None:
        """Update belief state from perception (emitter tracker) output.

        Args:
            perception_result: Dict from EmitterTracker.get_band_belief() with keys:
                - "bands": (n_bands, 10) feature array
                - "obs": (n_bands * 10,) flat array
                - "n_tracks": int
        """
        bands = perception_result.get("bands")
        if bands is None:
            return
        bands = np.asarray(bands, dtype=np.float32)
        if bands.shape != (self.n_bands, 10):
            logger.warning("Perception bands shape mismatch: %s vs (%d, 10)", bands.shape, self.n_bands)
            return

        # Blend perception features with existing belief (EMA)
        alpha = 0.3
        # Features from perception that we trust: emitter_count, deint_conf, per_stab, agility
        # Features we keep from local belief: occupancy, detection_rate, revisit_age, priority
        self.estimated_emitter_count = (1 - alpha) * self.estimated_emitter_count + alpha * bands[:, 5]
        self.deinterleaver_confidence = (1 - alpha) * self.deinterleaver_confidence + alpha * bands[:, 6]
        self.periodicity_stability = (1 - alpha) * self.periodicity_stability + alpha * bands[:, 7]
        self.agility_indicator = (1 - alpha) * self.agility_indicator + alpha * bands[:, 8]

        # Recompute uncertainty and priority with updated features
        self.update_uncertainty()
        self.update_priority()

    def advance_time(self) -> None:
        self.revisit_age += 1

    def touch(self, band: int) -> None:
        b = int(band)
        if 0 <= b < self.n_bands:
            self.revisit_age[b] = 0

    def update_uncertainty(self) -> None:
        p = np.clip(self.detection_rate, 0.0, 1.0)
        self.uncertainty = 1.0 - np.abs(2.0 * p - 1.0)
        never_visited = self._visits == 0
        self.uncertainty[never_visited] = 1.0

    def update_priority(self) -> None:
        norm_age = np.clip(self.revisit_age.astype(np.float32) / 50.0, 0.0, 1.0)
        self.priority_score = np.clip(0.4 * norm_age + 0.4 * self.occupancy_prob + 0.2 * self.uncertainty, 0.0, 1.0)

    def band_features(self, b: int) -> np.ndarray:
        """Return the canonical 10-feature vector for one band."""
        p = float(np.clip(self.occupancy_prob[b], 0.0, 1.0))
        age = float(min(float(self.revisit_age[b]), 50.0) / 50.0)
        self.update_uncertainty()
        self.update_priority()
        det_rate = float(self.detection_rate[b])
        miss_rate = 1.0 - det_rate
        unc = float(self.uncertainty[b])
        emit_cnt = float(self.estimated_emitter_count[b])
        deint_conf = float(self.deinterleaver_confidence[b])
        per_stab = float(self.periodicity_stability[b])
        agil = float(self.agility_indicator[b])
        prio = float(self.priority_score[b])

        return np.array([
            p,          # 1. current/estimated occupancy
            det_rate,   # 2. recent detection/hit rate
            miss_rate,  # 3. recent miss rate
            unc,        # 4. uncertainty
            age,        # 5. time since last visit (normalized)
            emit_cnt,   # 6. estimated emitter count
            deint_conf, # 7. deinterleaver confidence
            per_stab,   # 8. PRI/periodicity stability
            agil,       # 9. frequency-agility indicator
            prio,       # 10. risk/priority score
        ], dtype=np.float32)


class CognitiveRFScanEnv(gym.Env):
    """Receiver-driven cognitive scan scheduler environment.

    Wraps RadioEnvironment + SieveReceiver. The scheduler controls the receiver
    center frequency / stepping / dwells. The radio world evolves through dwells.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict,
        records: Optional[Sequence[PulseRecord]] = None,
        seed: int | None = 42,
        records_provider: Optional[Callable[[], Sequence[PulseRecord]]] = None,
        deinterleaver_model: Optional[Any] = None,
        deinterleaver_config: Optional[dict] = None,
        semantic_memory_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.records_provider = records_provider

        # Perception configuration
        self.deinterleaver_model = deinterleaver_model
        self.deinterleaver_config = deinterleaver_config or {}
        self.perception_enabled = deinterleaver_model is not None

        # Semantic memory configuration
        self.semantic_memory_path = semantic_memory_path or config.get("semantic_memory_path", "data/semantic_memory.db")

        # Periodic interceptor configuration
        self.periodic_min_obs = config.get("periodic_min_obs", 20)
        self.periodic_hist_bins = config.get("periodic_hist_bins", 100)

        self.n_bands: int = int(config.get("n_bands", 36))
        self.freq_min: float = float(config.get("freq_min_mhz", 0.0))
        self.freq_max: float = float(config.get("freq_max_mhz", 18000.0))
        self.ibw_mhz: float = float(config.get("ibw_mhz", 500.0))
        self.dwell_time_us: float = float(config.get("dwell_time_us", 500.0))
        self.frequency_step_mhz: float = float(config.get("frequency_step_mhz", 500.0))
        self.detection_threshold_db: float = float(config.get("detection_threshold_db", -140.0))
        self.max_steps_per_episode: int = int(config.get("max_steps_per_episode", 2000))

        # Reward weights
        self.w_hit: float = float(config.get("w_hit", 1.0))
        self.w_novel: float = float(config.get("w_novel", 2.0))
        self.w_miss: float = float(config.get("w_miss", -1.0))

        # Feature layout: STATE_FEATURES_PER_BAND features per band
        self.band_features = STATE_FEATURES_PER_BAND
        self.obs_dim = int(self.n_bands * self.band_features)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_bands)

        assert self.observation_space.shape[0] == self.obs_dim == self.n_bands * self.band_features, (
            f"Observation dimension mismatch: space={self.observation_space.shape[0]} vs obs_dim={self.obs_dim}"
        )

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Components built at reset
        self.receiver: SieveReceiver | None = None
        self.radio_env: RadioEnvironment | None = None
        self.belief: BeliefState | None = None
        self.emitter_tracker: EmitterTracker | None = None
        self.semantic_memory: SemanticMemory | None = None
        self.periodic_interceptor: PeriodicScanInterceptor | None = None
        self.records: list[PulseRecord] = list(records or [])
        self.fom = FiguresOfMerit()

        self.current_step = 0
        self.intercepted_emitters: set[int] = set()
        self._gt_active_ever: set[int] = set()
        self._last_dwell_start: float = 0.0

        # Perception: accumulate PDWs for windowed deinterleaving
        self._pdw_buffer: list[dict] = []
        self._min_deinterleave_pulses = self.deinterleaver_config.get("min_pulses", 50)
        self._deinterleave_interval = self.deinterleaver_config.get("interval_steps", 10)

    # ------------------------------------------------------------------ setup
    def _build_receiver(self) -> SieveReceiver:
        return SieveReceiver(
            total_bandwidth=self.freq_max - self.freq_min,
            ibw=self.ibw_mhz,
            frequency_step=self.frequency_step_mhz,
            dwell_time=self.dwell_time_us,
            detection_threshold_db=self.detection_threshold_db,
        )

    def _build_radio_env(self) -> RadioEnvironment:
        # NOTE: We do NOT auto-attach the receiver bridge here. The env controls
        # event->receiver flow explicitly in _advance_world_to so that EXIT events
        # are deferred until after dwell detection (correct causal ordering).
        return RadioEnvironment(self.records)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            np.random.seed(seed)
        if self.records_provider is not None:
            self.records = list(self.records_provider() or [])

        self.receiver = self._build_receiver()
        self.radio_env = self._build_radio_env()
        self.belief = BeliefState(self.n_bands)

        # Initialize emitter tracker for perception
        if self.perception_enabled:
            self.emitter_tracker = EmitterTracker(n_bands=self.n_bands)
        else:
            self.emitter_tracker = None

        # Initialize semantic memory
        self.semantic_memory = SemanticMemory(self.semantic_memory_path)

        # Initialize periodic interceptor
        self.periodic_interceptor = PeriodicScanInterceptor(
            min_observations=self.periodic_min_obs,
            hist_bins=self.periodic_hist_bins,
        )

        self.current_step = 0
        self.intercepted_emitters = set()
        self._gt_active_ever = set()
        self.fom.reset()

        # Perception buffer
        self._pdw_buffer = []

        # Prime the radio world with the first bunch of entries without stepping
        # the agent clock yet (just establish the initial window/time).
        # We do this lazily on the first step. Initial observation is empty.
        return self._build_observation(), {}

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        if self.receiver is None or self.radio_env is None or self.belief is None:
            raise RuntimeError("reset() must be called before step()")

        action = int(action)
        if not (0 <= action < self.action_space.n):
            raise ValueError(f"action {action} outside Discrete({self.action_space.n})")

        # 1. Translate scheduler action -> receiver action: tune center frequency
        #    to cover the chosen band, then dwell.
        band = action
        center = self._band_to_center(band)
        self.receiver.tune(center)

        dwell_start = self.receiver.current_time_us
        dwell_end = self.receiver.current_time_us + self.receiver.dwell_time_us

        # 2. Advance the RF world through ALL events up to dwell_end BEFORE the
        #    receiver dwell. The world "evolves during the dwell": entry events
        #    feed the receiver's pulse buffer, keeping it causal (receiver only
        #    sees events up to the dwell end, never future pulses). EXIT events
        #    within the dwell are deferred until AFTER detection so that pulses
        #    active within [dwell_start, dwell_end] are still detected.
        entry_count, exit_count = self._advance_world_to(dwell_end)

        # 3. Execute the dwell over exactly [dwell_start, dwell_end]. Pin the
        #    receiver clock to the dwell start so detection covers the right window,
        #    and record the dwell window on the receiver so the observation and the
        #    reward timing term reflect the true window.
        self.receiver.current_time_us = dwell_start
        self.receiver.dwell_start_us = dwell_start
        self.receiver.dwell_end_us = dwell_end
        detections = self.receiver._detect_buffered_interval(dwell_start, dwell_end)
        self.receiver._record(detections, observation_time_us=dwell_start)

        # 4. Now resolve deferred EXIT events (prune ended pulses for future dwells).
        self._resolve_exits(exit_count)
        self.receiver.current_time_us = dwell_end
        self.receiver._prune(dwell_end)

        # Determine ground truth over the dwell interval (reward/eval only, not obs)
        ground_truth_active, _novel_opportunity, active_bands_vec, active_emitters = self._ground_truth_for_dwell(dwell_start, dwell_end)

        # 4. Collect causal observations
        observation = self.receiver.get_observation()

        # 5. Perception pipeline: accumulate PDWs and run deinterleaving
        detections = getattr(observation, "detections", [])
        any_hit = len(detections) > 0

        # Accumulate detected PDWs for deinterleaving
        if any_hit:
            for d in detections:
                self._pdw_buffer.append({
                    "time_us": float(getattr(d, "time_us", 0.0)),
                    "frequency_mhz": float(d.frequency_mhz),
                    "pulse_width_us": float(d.pulse_width_us),
                    "amplitude_db": float(d.amplitude_db),
                    "aoa_deg": float(d.aoa_deg),
                    "emitter_id": getattr(d, "emitter_id", -1),  # GT for evaluation only
                })

        # Run perception at intervals if enabled
        perception_result = None
        if self.perception_enabled and self.emitter_tracker is not None:
            if len(self._pdw_buffer) >= self._min_deinterleave_pulses and self.current_step % self._deinterleave_interval == 0:
                perception_result = self._run_perception(band, dwell_start, dwell_end)
                if perception_result is not None:
                    self.belief.update_from_perception(perception_result)

                    # Update semantic memory with emitter profiles from tracker
                    self._update_semantic_memory()

                    # Update periodic interceptor with detections
                    self._update_periodic_interceptor(detections, band, dwell_start)

                # Clear buffer after processing (keep last N for continuity)
                self._pdw_buffer = self._pdw_buffer[-self._min_deinterleave_pulses:]

        # Apply semantic memory band priority boost to belief
        if self.semantic_memory is not None:
            semantic_boost = self.semantic_memory.get_band_priority_boost(
                n_bands=self.n_bands,
                freq_min=self.freq_min,
                freq_max=self.freq_max,
            )
            # Blend into priority score (feature index 9)
            alpha = 0.2
            self.belief.priority_score = (1 - alpha) * self.belief.priority_score + alpha * semantic_boost

        # Check periodic interceptor for preemptive schedule recommendation
        preemptive_band = None
        if self.periodic_interceptor is not None:
            schedule = self.periodic_interceptor.get_preemptive_schedule(
                current_time_us=self.receiver.current_time_us,
                horizon_us=self.dwell_time_us * 10,  # Look ahead 10 dwells
            )
            if schedule:
                # Use the highest confidence imminent prediction
                next_pred = schedule[0]
                if next_pred["confidence"] > 0.7:
                    preemptive_band = next_pred["expected_band"]
                    logger.debug("Periodic preemptive recommendation: band %d (conf=%.2f, t=%.0fus)",
                                preemptive_band, next_pred["confidence"], next_pred["expected_time_us"])

        # 6. Update causal belief (from observation only)
        self.belief.record_visit(band, any_hit, detections=detections)
        self.belief.advance_time()
        self.belief.touch(band)

        # Store preemptive recommendation in info for scheduler
        self._preemptive_band = preemptive_band

        # Update intercepted set from detection emitter_id (ground truth for reward/eval only)
        new_ids = set()
        for d in detections:
            eid = getattr(d, "emitter_id", None)
            if eid is not None:
                new_ids.add(int(eid))
        newly = new_ids - self.intercepted_emitters
        self.intercepted_emitters.update(new_ids)

        # 6. Calculate reward (uses ground truth ONLY for shaping)
        from src.training.reward import receiver_reward_components

        reward_components = receiver_reward_components(
            observation=observation,
            ground_truth_active=ground_truth_active,
            novel_emitter=bool(newly),
            had_any_opportunity=ground_truth_active,
            w_hit=self.w_hit,
            w_novel=self.w_novel,
            w_miss=self.w_miss,
        )
        reward = reward_components["reward"]
        self.fom.record_reward_components(reward_components)

        # 7. Update metrics (ground-truth-based eval only).
        # Real intercept-time error: earliest detected pulse ToA minus the dwell
        # onset (receiver clock). Never hard-coded to 0.0; reported only for
        # actual intercepts (a non-intercepted active band is not a 'miss' per
        # the evaluation contract).
        detect_toas = [float(getattr(d, "toa_us", getattr(d, "time_us", float("nan")))) for d in detections]
        detect_toas = [t for t in detect_toas if t == t]
        if any_hit and detect_toas:
            first_detect_toa = min(detect_toas)
            intercept_time_error_us = max(0.0, first_detect_toa - dwell_start)
        else:
            intercept_time_error_us = float("nan")
        self.fom.record_emitters(active_emitters, new_ids)
        self.fom.update(
            band_chosen=band,
            ground_truth_active=active_bands_vec,
            pred_active=any_hit,
            intercept_time_error_us=intercept_time_error_us,
            reward=float(reward),
        )

        # 8. Termination
        self.current_step += 1
        terminated = bool(self.radio_env.done and self.current_step >= 1)
        truncated = bool(self.current_step >= self.max_steps_per_episode)

        # Strip emitter_id / ground truth from observation returned to agent:
        # detections list may carry emitter_id; we remove it before building obs.
        info = {
            "detections": [d.to_dict() for d in detections],
            "hit": any_hit,
            "novel_emitter": bool(newly),
            "ground_truth_active": ground_truth_active,
            "intercept_time_error_us": float(intercept_time_error_us),
            "band_center_mhz": observation.center_frequency_mhz if observation is not None else 0.0,
            "receiver_time_us": self.receiver.current_time_us,
            "preemptive_band": getattr(self, "_preemptive_band", None),
        }

        next_obs = self._build_observation()
        return next_obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------- internals
    def _band_to_center(self, band: int) -> float:
        """Convert band index to receiver center frequency (MHz).

        Center is placed so the band's midpoint is centred within the IBW, clipped
        to the receiver's legal center range.
        """
        band = int(band)
        if self.n_bands <= 1:
            return self.freq_min + self.ibw_mhz / 2.0
        band_width = (self.freq_max - self.freq_min) / self.n_bands
        band_mid = (self.freq_max - self.freq_min) * (band + 0.5) / self.n_bands + self.freq_min
        legal_min = self.ibw_mhz / 2.0
        legal_max = (self.freq_max - self.freq_min) - self.ibw_mhz / 2.0
        center = min(max(band_mid, legal_min), legal_max)
        return float(center)

    def _band_index(self, center_frequency_mhz: float) -> int:
        """Map a frequency to its band index.

        Uses floor(frac * n_bands) with clamping to ensure correct mapping.
        """
        if self.freq_max <= self.freq_min:
            return 0
        frac = (float(center_frequency_mhz) - self.freq_min) / (self.freq_max - self.freq_min)
        frac = min(1.0, max(0.0, frac))
        # Use n_bands (not n_bands-1) so that freq_max maps to band n_bands-1
        idx = int(frac * self.n_bands)
        return min(idx, self.n_bands - 1)

    def _ground_truth_for_dwell(self, lower_us: float, upper_us: float) -> tuple[bool, bool, np.ndarray, set[int]]:
        """Evaluate ground-truth activity in [lower_us, upper_us) and any novel emitters.

        Returns (any_active, novel_emitter_found, active_bands_vector, active_emitter_ids).
        Used ONLY internally for reward shaping and evaluation; never enters the observation.
        """
        any_active = False
        novel = False
        active_bands = np.zeros(self.n_bands, dtype=np.int8)
        active_emitters: set[int] = set()
        lo, hi = float(lower_us), float(upper_us)
        band_width = (self.freq_max - self.freq_min) / max(1, self.n_bands)

        for rec in self.records:
            toa = float(rec.toa_us)
            exit_us = toa + float(rec.pulse_width_us)
            if toa < hi and exit_us > lo:
                any_active = True
                eid = int(rec.emitter_id)
                active_emitters.add(eid)
                if eid not in self.intercepted_emitters:
                    novel = True
                b = int(np.clip((float(rec.frequency_mhz) - self.freq_min) / max(band_width, 1e-6), 0, self.n_bands - 1))
                active_bands[b] = 1

        return any_active, novel, active_bands, active_emitters

    def _run_perception(self, band: int, dwell_start: float, dwell_end: float) -> Optional[dict]:
        """Run deinterleaver on accumulated PDW buffer and update emitter tracker.

        Args:
            band: Current band index.
            dwell_start: Dwell start time (µs).
            dwell_end: Dwell end time (µs).

        Returns:
            Perception result dict from EmitterTracker.get_band_belief(), or None.
        """
        if not self._pdw_buffer:
            return None

        try:
            # Prepare PDW data for deinterleaver
            pdws = np.array([[p["time_us"], p["frequency_mhz"], p["pulse_width_us"],
                             p["aoa_deg"], p["amplitude_db"]] for p in self._pdw_buffer],
                           dtype=np.float64)

            # Normalize PDWs
            from src.preprocessing.normalise import normalise_pdws
            pdws_norm, _ = normalise_pdws(pdws, fit_stats=self.deinterleaver_config.get("fit_stats"))

            # Run windowed deinterleaving
            from src.models.deinterleaver import windowed_cluster_deinterleave
            window_size = self.deinterleaver_config.get("window_size", min(2048, len(pdws_norm)))
            stride = self.deinterleaver_config.get("stride", window_size // 2)
            min_cluster_size = self.deinterleaver_config.get("min_cluster_size", 10)
            min_samples = self.deinterleaver_config.get("min_samples", 5)

            result = windowed_cluster_deinterleave(
                self.deinterleaver_model,
                pdws_norm,
                toa_us=pdws[:, 0],
                window_size=window_size,
                stride=stride,
                device=self.deinterleaver_config.get("device", "cpu"),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            # Update emitter tracker with deinterleaver results
            labels = result["labels"]
            toa_us = pdws[:, 0]
            freq_mhz = pdws[:, 1]
            aoa_deg = pdws[:, 3]
            pw_us = pdws[:, 2]
            amp_db = pdws[:, 4]

            current_time = dwell_end
            self.emitter_tracker.update_from_deinterleaver(
                labels=labels,
                toa_us=toa_us,
                freq_mhz=freq_mhz,
                aoa_deg=aoa_deg,
                pw_us=pw_us,
                amp_db=amp_db,
                current_time=current_time,
                band=band,
                min_cluster_size=min_cluster_size,
            )

            # Get band belief from updated tracks
            perception_result = self.emitter_tracker.get_band_belief(
                freq_min=self.freq_min,
                freq_max=self.freq_max,
                ema_occupancy=self.belief.occupancy_prob,
            )

            logger.debug("Perception updated: %d tracks, n_clusters=%d",
                        len(self.emitter_tracker.get_active_tracks()), result["n_clusters"])
            return perception_result

        except Exception as exc:
            logger.warning("Perception pipeline failed: %s", exc)
            return None

    def _update_semantic_memory(self) -> None:
        """Update semantic memory with profiles from active emitter tracks."""
        if self.semantic_memory is None or self.emitter_tracker is None:
            return

        for track in self.emitter_tracker.get_active_tracks():
            if track.observation_count < 5:
                continue  # Need minimum observations for reliable profile

            # Create emitter profile from track
            profile = EmitterProfile(
                emitter_id=f"track_{track.track_id}",
                mean_pri_us=track.pri_estimate_us or 0.0,
                freq_min_mhz=float(np.min(track.frequency_history)) if track.frequency_history else 0.0,
                freq_max_mhz=float(np.max(track.frequency_history)) if track.frequency_history else 0.0,
                mean_pw_us=float(np.mean(track.pw_history)) if track.pw_history else 0.0,
                aoa_mean=float(np.mean(track.aoa_history)) if track.aoa_history else 0.0,
                amplitude_mean=float(np.mean(track.amplitude_history)) if track.amplitude_history else 0.0,
                priority_score=track.get_cluster_confidence(),
                is_periodic=1 if track.pri_confidence > 0.7 else 0,
                scan_period_us=track.pri_estimate_us,
                intercept_count=track.observation_count,
                last_seen_us=track.last_seen_time,
            )
            self.semantic_memory.write_emitter(profile)

    def _update_periodic_interceptor(self, detections: list, band: int, dwell_start: float) -> None:
        """Update periodic interceptor with new detections.

        Args:
            detections: List of DetectionObservation from current dwell.
            band: Band index where detections occurred.
            dwell_start: Dwell start time (µs).
        """
        if self.periodic_interceptor is None:
            return

        for d in detections:
            emitter_id = getattr(d, "emitter_id", None)
            if emitter_id is not None:
                toa = float(getattr(d, "time_us", getattr(d, "toa_us", dwell_start)))
                # Use track_id as emitter identifier for periodic tracking
                # For real operation, would use deinterleaver cluster label
                track_id = f"track_{emitter_id}"
                self.periodic_interceptor.record_intercept(track_id, toa, band)

    def _advance_world_to(self, target_time_us: float) -> tuple[int, int]:
        """Stream the radio environment ENTRY events at-or-before target into the receiver.

        Only ENTRY events with time <= target are emitted so the receiver buffer is
        populated causally. EXIT events and their times are captured and returned so
        the caller can prune after detection. Returns (entry_count, exit_count).

        Receives future events (time > target) are left queued — no leakage.
        """
        if self.radio_env is None:
            return 0, 0
        entry_count = 0
        exit_count = 0
        self._pending_exits: list[int] = []
        while self.radio_env.remaining_events > 0:
            next_time = self.radio_env.peek_time()
            if next_time is not None and next_time <= target_time_us:
                event = self.radio_env.step()
                if event is not None:
                    if event.event_type == "entry" and event.pulse is not None:
                        # add to receiver buffer WITHOUT triggering the receiver's
                        # auto-scan timing (the scheduler owns tuning/stepping).
                        self.receiver.add_pulse(event.pulse)
                        entry_count += 1
                    elif event.event_type == "exit":
                        # defer: don't prune yet; just remember the pulse_id
                        self._pending_exits.append(event.pulse_id)
                        exit_count += 1
            else:
                break
        return entry_count, exit_count

    def _resolve_exits(self, count: int) -> None:
        """Remove exited pulse_ids from the receiver buffer (after detection)."""
        for pid in self._pending_exits[:count]:
            self.receiver.remove_pulse(pid)
        self._pending_exits = []

    # --------------------------------------------------------------- obs
    def _build_observation(self) -> np.ndarray:
        """Build a pure scheduler observation from belief only (NO ground truth)."""
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        if self.belief is None:
            return vec
        for b in range(self.n_bands):
            f = self.belief.band_features(b)
            vec[b * self.band_features:(b + 1) * self.band_features] = f
        return vec

    def get_fom(self) -> dict[str, float]:
        return self.fom.summary()

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        obs_vec = self._build_observation()
        print(f"=== CognitiveRFScanEnv t={self.receiver.current_time_us:.0f}us step={self.current_step} ===")
        print("band | occ | rate | age")
        for b in range(min(self.n_bands, 10)):
            print(f"  {b:03d}  {obs_vec[b*self.band_features]:.2f}  {obs_vec[b*self.band_features+3]:.2f}  {obs_vec[b*self.band_features+2]:.2f}")

    # ------------------------------------------------------------------ seeds
    def seed(self, seed: int | None = None) -> list[int]:
        self._seed = seed
        return [seed]
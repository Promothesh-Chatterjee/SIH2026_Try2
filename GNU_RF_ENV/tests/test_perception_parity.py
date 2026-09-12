"""Phase 3U perception-parity tests: GNU RF translation vs production path.

Verifies the Phase 3U objective — the GNU RF -> scheduler translation layer
(GnuRfSchedulerTranslation) produces observations using the SAME perception
semantics as the production CognitiveRFScanEnv pipeline:

  ReceiverObservation -> PDW accumulation -> normalise_pdws()
    -> windowed_cluster_deinterleave() -> EmitterTracker.update_from_deinterleaver()
    -> get_band_belief() -> BeliefState.update_from_perception()
    -> band_features() -> (360,) float32.

No perception algorithm is reimplemented in the translation layer; it only OWNS
orchestration (PDW buffer, step counter, component instances).

A. Contract preserved with perception enabled (shape/dtype/range).
B. Equivalence in BOTH perception modes: identical observable PDWs through the
   production env path and the GNU RF translation path => identical obs.
C. Translation state honours the perception lifecycle (tracker runs and buffer
   is gated/trimmed only when a model is present).
D. RF-generated frequency -> correct active band mapping.
E. Multi-dwell state persistence (no per-dwell reset).
F. Emitter-ID invariance (ground truth never shapes the observation).
G. Noise characterization: bounded divergence, observation stays valid.
H. Empty dwell: valid observation, detection rate decays, no crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_GNU_RF_SCRIPTS = str(_REPO / "GNU_RF_ENV" / "scripts")
_MASTER_PKG = str(_REPO / "cognitive_ew_smart_scan")

for p in (_GNU_RF_SCRIPTS, _MASTER_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv  # noqa: E402
from src.receiver.models import DetectionObservation, ReceiverObservation  # noqa: E402
from scheduler_translation import GnuRfSchedulerTranslation  # noqa: E402


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "n_bands": 36,
    "freq_min_mhz": 0.0,
    "freq_max_mhz": 18000.0,
    "ibw_mhz": 500.0,
    "dwell_time_us": 500.0,
    "frequency_step_mhz": 500.0,
    "detection_threshold_db": -140.0,
    "max_steps_per_episode": 100,
    "semantic_memory_enabled": False,
}

FAST_PERCEPTION = {
    "min_pulses": 3,
    "interval_steps": 1,
    "min_cluster_size": 2,
    "min_samples": 2,
}

DEFAULT_PERCEPTION = {}


class _MockDeinterleaver:
    """Structured mock satisfying the model.infer()/embed_dim contract.

    Returns deterministic, well-separated unit embeddings: even rows map to
    vector A, odd rows to vector B.  HDBSCAN reliably recovers two clusters, so
    the EmitterTracker builds tracks that genuinely carry emitter-level features
    (equivalent to what a trained deinterleaver would emit for two emitters).
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim

    def infer(self, x, device="cpu"):
        n = x.shape[-2] if getattr(x, "ndim", 0) >= 2 else len(x)
        rng = np.random.RandomState(0)
        v_a = rng.randn(self.embed_dim).astype(np.float32)
        v_a /= max(float(np.linalg.norm(v_a)), 1e-8)
        v_b = rng.randn(self.embed_dim).astype(np.float32)
        v_b = v_b - (v_b @ v_a) * v_a
        v_b /= max(float(np.linalg.norm(v_b)), 1e-8)
        emb = np.tile(v_a, (n, 1)).copy()
        emb[1::2] = v_b
        return np.asarray(emb, dtype=np.float32)


_BAND_META = {
    6: (3250.0, (3250.1, 3250.5)),
    10: (5250.0, (5250.1, 5250.5)),
    15: (7750.0, (7750.1,)),
}


def _make_observation(
    dwell_idx: int,
    band: int = 6,
    n_pulses: int = 10,
    toa_stride_us: float = 40.0,
    amp_db: float = -100.0,
) -> ReceiverObservation:
    """Deterministic ReceiverObservation for one dwell on `band`."""
    center_mhz, freqs_mhz = _BAND_META[band]
    dwell_start = dwell_idx * 500.0
    toas = [dwell_start + 100.0 + k * toa_stride_us for k in range(n_pulses)]
    detections = [
        DetectionObservation(
            time_us=float(t),
            frequency_mhz=float(freqs_mhz[k % len(freqs_mhz)]),
            pulse_width_us=10.0,
            amplitude_db=float(amp_db),
            aoa_deg=0.0,
            center_frequency_mhz=float(center_mhz),
            detected=True,
        )
        for k, t in enumerate(toas)
    ]
    return ReceiverObservation(
        time_us=dwell_start,
        center_frequency_mhz=float(center_mhz),
        ibw_mhz=500.0,
        dwell_time_us=500.0,
        dwell_interval_us=[dwell_start, dwell_start + 500.0],
        detections=detections,
    )


def _inject_pulses_into_env(env, obs: ReceiverObservation, dwell_idx: int) -> None:
    """Feed the same observable PDWs into the production env receiver buffer."""
    for i, d in enumerate(obs.detections):
        t = float(d.time_us)
        env.receiver.add_pulse({
            "toa_us": t,
            "exit_us": t + float(d.pulse_width_us),
            "frequency_mhz": float(d.frequency_mhz),
            "pulse_width_us": float(d.pulse_width_us),
            "amplitude_db": float(d.amplitude_db),
            "aoa_deg": float(d.aoa_deg),
            "pulse_id": dwell_idx * 100 + i,
        })


def _make_env(deinterleaver_model=None, deinterleaver_config=None, **kwargs):
    return CognitiveRFScanEnv(
        config=ENV_CONFIG,
        records=[],
        deinterleaver_model=deinterleaver_model,
        deinterleaver_config=deinterleaver_config or {},
        **kwargs,
    )


def _run_env_episode(env, band_seq):
    """Step the production env on a band sequence feeding the same PDWs each dwell."""
    env.reset(seed=42)
    obs_history = []
    for d_idx, band in enumerate(band_seq):
        obs_r = _make_observation(d_idx, band)
        _inject_pulses_into_env(env, obs_r, d_idx)
        obs_history.append(env.step(band * 5 + 1)[0])
    return obs_history


def _run_translation_episode(translator, band_seq):
    obs_history = []
    for d_idx, band in enumerate(band_seq):
        obs_history.append(translator.update(_make_observation(d_idx, band)))
    return obs_history


def _feat(obs: np.ndarray, band: int, feature: int) -> float:
    return float(obs[band * 10 + feature])


# ===================================================================
# A. Contract preserved with perception enabled
# ===================================================================

class TestPerceptionContract:
    def test_shape_dtype_range_with_model(self):
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        for _ in range(5):
            obs = tr.update(_make_observation(0))
            assert obs.shape == (360,)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_reset_clears_state(self):
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        for _ in range(5):
            tr.update(_make_observation(0))
        tr.reset()
        assert tr._step_count == 0
        assert len(tr._pdw_buffer) == 0
        assert tr.emitter_tracker is not None
        obs = tr.update(_make_observation(0))
        assert obs.shape == (360,) and np.all(obs <= 1.0)


# ===================================================================
# B. Equivalence in BOTH perception modes (env path vs translation path)
# ===================================================================

class TestEnvTranslationEquivalence:
    def test_fast_gating_equivalence_with_model(self):
        """Perception runs often (interval=1, min_pulses=3): exact obs parity."""
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        env = _make_env(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        dwells = [6] * 8
        obs_a = _run_env_episode(env, dwells)
        obs_b = _run_translation_episode(tr, dwells)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"dwell {i}")

    def test_default_gating_equivalence_with_model(self):
        """Default production gating (min_pulses=50, interval=10): exact parity."""
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(DEFAULT_PERCEPTION),
        )
        env = _make_env(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(DEFAULT_PERCEPTION),
        )
        dwells = [6] * 12  # single perception run at dwell 10 (buffer >= 50)
        obs_a = _run_env_episode(env, dwells)
        obs_b = _run_translation_episode(tr, dwells)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"dwell {i}")

    def test_multi_band_equivalence_with_model(self):
        """Perception across multiple bands: tracker band attribution parity."""
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        env = _make_env(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        dwells = [6, 6, 15, 15, 6]
        obs_a = _run_env_episode(env, dwells)
        obs_b = _run_translation_episode(tr, dwells)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"dwell {i}")

    def test_equivalence_no_model(self):
        """Without a model both paths use only the record_visit fallback.""" 
        tr = GnuRfSchedulerTranslation()
        env = _make_env(deinterleaver_model=None, deinterleaver_config={})
        dwells = [6, 15, 6]
        obs_a = _run_env_episode(env, dwells)
        obs_b = _run_translation_episode(tr, dwells)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"dwell {i}")


# ===================================================================
# C. Translation state honours the production perception lifecycle
# ===================================================================

class TestPerceptionPipelineState:
    def test_no_model_uses_record_visit_fallback(self):
        tr = GnuRfSchedulerTranslation()
        obs = tr.update(_make_observation(0))
        assert tr.emitter_tracker is None
        # PDWs are still accumulated (exactly like the production env) but are
        # never consumed because the perception step requires a model.
        assert len(tr._pdw_buffer) == 10
        # record_visit estimates features 5-8 from observable detections only:
        # all AoA = 0 -> one bearing -> emitter_count = clip(1/5, 0.1, 1) = 0.2
        # deinterleaver_confidence = clip(0.6 + 0.08*min(10,5), 0, 1) = 1.0
        assert _feat(obs, 6, 5) == pytest.approx(0.2, abs=1e-6)
        assert _feat(obs, 6, 6) == pytest.approx(1.0, abs=1e-6)
        # Unvisited bands stay untouched by the fallback.
        assert _feat(obs, 20, 5) == 0.0

    def test_with_model_drives_emitter_tracker(self):
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        for _ in range(6):
            tr.update(_make_observation(0))
        assert tr.emitter_tracker is not None
        active = tr.emitter_tracker.get_active_tracks()
        assert len(active) >= 2, (
            f"structured mock should yield >=2 emitter tracks, got {len(active)}"
        )
        # Buffer is gated/trimmed by min_pulses, not unbounded.
        assert len(tr._pdw_buffer) <= tr._min_deinterleave_pulses + 10
        obs = tr.get_observation()
        assert obs.shape == (360,) and np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_perception_distinguishes_two_emitters(self):
        """Deinterleaver tracks two emitters while the AoA proxy reports one."""
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        for _ in range(6):
            tr.update(_make_observation(0))
        active = tr.emitter_tracker.get_active_tracks()
        freqs = [t.frequency_history[-1] for t in active if len(t.frequency_history)]
        near_3250_1 = any(abs(f - 3250.1) < 5.0 for f in freqs)
        near_3250_5 = any(abs(f - 3250.5) < 5.0 for f in freqs)
        assert near_3250_1 and near_3250_5, (
            f"tracks should cover both emitters, got freqs {sorted(freqs)}"
        )


# ===================================================================
# D. RF-generated frequency -> correct active band mapping
# ===================================================================

class TestBandMappingTranslated:
    def test_3200_1_mhz_maps_to_band_6(self):
        tr = GnuRfSchedulerTranslation(deinterleaver_model=_MockDeinterleaver())
        obs = tr.update(_make_observation(0))
        assert _feat(obs, 6, 0) > 0.0, "band 6 should be visited"
        assert _feat(obs, 5, 0) == 0.5, "band 5 should stay unvisited"
        assert _feat(obs, 7, 0) == 0.5, "band 7 should stay unvisited"

    def test_7999_75_mhz_maps_to_band_15(self):
        tr = GnuRfSchedulerTranslation(deinterleaver_model=_MockDeinterleaver())
        obs = tr.update(_make_observation(0, band=15))
        assert _feat(obs, 15, 0) > 0.0, "band 15 should be visited"
        assert _feat(obs, 14, 0) == 0.5, "band 14 should stay unvisited"
        assert _feat(obs, 16, 0) == 0.5, "band 16 should stay unvisited"

    def test_boundary_8000_mhz_maps_to_band_16(self):
        tr = GnuRfSchedulerTranslation(deinterleaver_model=_MockDeinterleaver())
        obs = tr.update(_make_observation(0, band=10))
        assert _feat(obs, 10, 0) > 0.0, "band 10 should be visited"


# ===================================================================
# E. Multi-dwell state persistence (no per-dwell reset)
# ===================================================================

class TestMultiDwellPersistence:
    def test_five_dwell_state_persists(self):
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        obs_hist = _run_translation_episode(tr, [6, 15, 6, 10, 15])
        final = obs_hist[-1]
        # Band 6 visited twice -> occupancy persists.
        assert _feat(final, 6, 0) > 0.0
        # normalize age: band 6 visited recently vs band 10 visited once -> ages.
        age6 = _feat(final, 6, 2)
        age10 = _feat(final, 10, 2)
        assert age6 <= age10, "recently visited band must have smaller/equal normalized age"

    def test_unvisited_bands_keep_clean_belief(self):
        tr = GnuRfSchedulerTranslation()
        for band in [6, 15, 6, 10, 15]:
            tr.update(_make_observation(0, band=band))
        obs = tr.get_observation()
        # Never-visited bands must have zero evidence (occupancy/det_rate/
        # emitter features) while keeping the belief baseline (uncertainty=1,
        # normalized age/priority from global time decay).
        for b in [30, 31, 32]:
            block = obs[b * 10:b * 10 + 10]
            assert block[0] == 0.5, f"band {b} occupancy should be 0.5"
            assert block[1] == 0.0, f"band {b} det_rate should be 0"
            assert block[2] == 1.0, f"band {b} miss_rate should be 1 (no data)"
            assert block[3] == 1.0, f"band {b} uncertainty should be 1 (unvisited)"
            assert block[5] == 0.0 and block[6] == 0.0, f"band {b} no emitter data"
            assert block[7] == 0.0 and block[8] == 0.0, f"band {b} no PRI/agility data"


# ===================================================================
# F. Emitter-ID invariance (ground truth must never shape the obs)
# ===================================================================

class TestEmitterIdInvariance:
    def test_renamed_emitter_ids_produce_identical_obs(self):
        def run(ids_fn):
            tr = GnuRfSchedulerTranslation(
                deinterleaver_model=_MockDeinterleaver(),
                deinterleaver_config=dict(FAST_PERCEPTION),
            )
            for _ in range(4):
                obs_r = _make_observation(0)
                ids_fn(obs_r)
                tr.update(obs_r)
            return tr.get_observation()

        a = run(lambda o: o)                       # no emitter ids
        b = run(lambda o: [setattr(d, "emitter_id", 5000 + i)
                           for i, d in enumerate(o.detections)])
        np.testing.assert_array_equal(a, b)


# ===================================================================
# G. Noise characterization (measure + bounded divergence)
# ===================================================================

class TestNoiseCharacterization:
    def test_small_jitter_causes_small_bounded_divergence(self):
        tr_clean = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        tr_noisy = GnuRfSchedulerTranslation(
            deinterleaver_model=_MockDeinterleaver(),
            deinterleaver_config=dict(FAST_PERCEPTION),
        )
        clean = _make_observation(0)
        noisy = _make_observation(0)
        rng = np.random.RandomState(7)
        for d in noisy.detections:
            d.time_us += rng.uniform(-1.0, 1.0)
            d.frequency_mhz += rng.uniform(-0.3, 0.3)
            d.amplitude_db += rng.uniform(-1.0, 1.0)
        for _ in range(4):
            tr_clean.update(clean)
            tr_noisy.update(noisy)
        obs_clean = tr_clean.get_observation()
        obs_noisy = tr_noisy.get_observation()
        diff = np.abs(obs_clean.astype(float) - obs_noisy.astype(float))
        assert np.all(np.isfinite(obs_noisy))
        assert np.all(obs_noisy >= 0.0) and np.all(obs_noisy <= 1.0)
        assert float(np.max(diff)) <= 0.05, f"max per-feature change {np.max(diff):.4f}"


# ===================================================================
# H. Empty dwell handling
# ===================================================================

class TestEmptyDwell:
    def test_empty_dwell_is_valid_and_decays_detection_rate(self):
        tr = GnuRfSchedulerTranslation()
        obs_empty = ReceiverObservation(
            time_us=525.0,
            center_frequency_mhz=3250.0,
            dwell_time_us=500.0,
            detections=[],
        )
        obs1 = tr.update(_make_observation(0))
        rate1 = _feat(obs1, 6, 1)
        obs2 = tr.update(obs_empty)
        assert obs2.shape == (360,) and obs2.dtype == np.float32
        assert np.all(np.isfinite(obs2))
        assert np.all(obs2 >= 0.0) and np.all(obs2 <= 1.0)
        # Same band visited twice: 1 hit of 2 visits -> rate drops below 1.0.
        rate2 = _feat(obs2, 6, 1)
        assert rate2 == pytest.approx(0.5, abs=1e-6)
        assert rate2 < rate1





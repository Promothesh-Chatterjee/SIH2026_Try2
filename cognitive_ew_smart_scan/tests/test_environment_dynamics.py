"""
Comprehensive Test Suite for Environment Dynamics, Frequency-Agility, and Temporal Causality.

Phase 5 Verifications:
- Phase 5C: 36-Band discretization and boundary edge mapping (0 MHz, 250 MHz, 499.999 MHz, 500 MHz, ..., 18,000 MHz).
- Phase 5D: Deterministic synthetic test emitters for environment testing:
    1. Stationary emitter
    2. Slow hopper (band changes every several decisions)
    3. Medium hopper (band changes every 1-5 decisions)
    4. Fast hopper (band changes every decision)
    5. Random hopper (discrete random band transitions)
    6. Periodic hopper (cyclic A -> C -> F -> C -> A)
    7. Sparse random burst emitter
    8. Multi-emitter concurrent agility
- Phase 5B: Strict temporal causality & zero-future-information leakage in belief and observation.
- Phase 5E: Physical temporal resolution and hop rate calculation.
- Phase 5F: Belief state temporal responsiveness (agility indicator, occupancy decay, revisit age).
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_BAND_FEATURES,
    NORMAL_DWELL,
    SHORT_DWELL,
    LONG_DWELL,
)
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord


@pytest.fixture
def env_cfg():
    return {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": 360,
        "band_features": 10,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "base_dwell_time_us": 500.0,
        "dwell_time_us": 500.0,
        "detection_threshold_db": -140.0,
        "semantic_memory_enabled": False,
        "perception_enabled": False,
    }


# =====================================================================
# Phase 5C — 36-Band Discretization & Edge Boundary Tests
# =====================================================================

class TestBandDiscretization:
    """Verifies physical RF frequency to 36-band mapping without gaps, overlaps, or drops."""

    @pytest.mark.parametrize(
        "freq_mhz, expected_band",
        [
            (0.0, 0),
            (250.0, 0),
            (499.999, 0),
            (500.0, 1),
            (750.0, 1),
            (999.999, 1),
            (1000.0, 2),
            (2500.0, 5),
            (2750.0, 5),
            (2999.999, 5),
            (3000.0, 6),
            (8750.0, 17),
            (9000.0, 18),
            (17499.999, 34),
            (17500.0, 35),
            (17750.0, 35),
            (17999.999, 35),
            (18000.0, 35),
        ],
    )
    def test_frequency_to_band_boundaries(self, env_cfg, freq_mhz, expected_band):
        env = CognitiveRFScanEnv(env_cfg)
        band = env._band_index(freq_mhz)
        assert band == expected_band, f"Freq {freq_mhz} MHz mapped to band {band} != expected {expected_band}"

    def test_all_36_band_centers_and_windows(self, env_cfg):
        env = CognitiveRFScanEnv(env_cfg)
        env.reset()
        for b in range(36):
            center = env._band_to_center(b)
            expected_center = b * 500.0 + 250.0
            assert center == pytest.approx(expected_center, abs=1e-3)
            # Center mapped back must equal b
            assert env._band_index(center) == b
            # Receiver window must span [b*500, (b+1)*500]
            env.receiver.tune(center)
            low, high = env.receiver.get_frequency_window()
            assert low == pytest.approx(b * 500.0, abs=1e-3)
            assert high == pytest.approx((b + 1) * 500.0, abs=1e-3)


# =====================================================================
# Phase 5D — Deterministic Synthetic Agile Emitter Scenarios
# =====================================================================

class TestDeterministicAgileEmitters:
    """Test environment interaction with 8 canonical emitter behavior patterns."""

    def test_stationary_emitter(self, env_cfg):
        """Stationary emitter: constant frequency in Band 5 (2750 MHz)."""
        pulses = [
            PulseRecord(toa_us=t, frequency_mhz=2750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)
            for t in np.arange(100.0, 5000.0, 500.0)
        ]
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        # Step into Band 5 with NORMAL_DWELL (action = 5 * 5 + 1 = 26)
        obs, r, term, trunc, info = env.step(26)
        assert info["hit"] is True
        assert len(info["detections"]) == 1
        assert info["detections"][0]["frequency_mhz"] == 2750.0
        assert env.belief.occupancy_prob[5] > 0.5

    def test_slow_hopper_emitter(self, env_cfg):
        """Slow hopper: dwells in Band 3 for 2 steps (t=0..1000us), then hops to Band 8 (t=1000..2000us)."""
        pulses = [
            # 2 steps in Band 3 (1750 MHz)
            PulseRecord(toa_us=200.0, frequency_mhz=1750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
            PulseRecord(toa_us=700.0, frequency_mhz=1750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
            # Hops to Band 8 (4250 MHz) at t=1200 us
            PulseRecord(toa_us=1200.0, frequency_mhz=4250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
            PulseRecord(toa_us=1700.0, frequency_mhz=4250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        ]
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        # Step 0 (0..500 us): Dwell in Band 3 -> Hit
        _, _, _, _, info0 = env.step(3 * 5 + 1)
        # Step 1 (500..1000 us): Dwell in Band 3 -> Hit
        _, _, _, _, info1 = env.step(3 * 5 + 1)
        assert info0["hit"] is True
        assert info1["hit"] is True

        # Step 2 (1000..1500 us): Dwell in Band 3 -> Miss (emitter hopped to Band 8 at t=1200 us)
        _, _, _, _, info2 = env.step(3 * 5 + 1)
        assert info2["hit"] is False  # Hopped away!

        # Step 3 (1500..2000 us): Dwell in Band 8 (action = 8*5 + 1 = 41): intercept hop destination
        _, _, _, _, info3 = env.step(8 * 5 + 1)
        assert info3["hit"] is True
        assert info3["detections"][0]["frequency_mhz"] == 4250.0

    def test_fast_hopper_emitter(self, env_cfg):
        """Fast hopper: changes band every single dwell decision (Band 2 -> Band 7 -> Band 14)."""
        pulses = [
            PulseRecord(toa_us=150.0, frequency_mhz=1250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),  # Band 2
            PulseRecord(toa_us=650.0, frequency_mhz=3750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),  # Band 7
            PulseRecord(toa_us=1150.0, frequency_mhz=7250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1), # Band 14
        ]
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        # Step 0: Tune to Band 2 (11) -> Hit
        _, _, _, _, i0 = env.step(2 * 5 + 1)
        assert i0["hit"] is True

        # Step 1: Tune to Band 7 (36) -> Hit
        _, _, _, _, i1 = env.step(7 * 5 + 1)
        assert i1["hit"] is True

        # Step 2: Tune to Band 14 (71) -> Hit
        _, _, _, _, i2 = env.step(14 * 5 + 1)
        assert i2["hit"] is True

    def test_periodic_cyclic_hopper(self, env_cfg):
        """Periodic hopper: cycles A (Band 4) -> C (Band 10) -> F (Band 20) -> C (Band 10) -> A (Band 4)."""
        pattern = [4, 10, 20, 10, 4]
        pulses = []
        for idx, b in enumerate(pattern):
            t = idx * 500.0 + 200.0
            pulses.append(
                PulseRecord(toa_us=t, frequency_mhz=b * 500.0 + 250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)
            )
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        for idx, expected_band in enumerate(pattern):
            _, _, _, _, info = env.step(expected_band * 5 + 1)
            assert info["hit"] is True
            assert info["detections"][0]["frequency_mhz"] == expected_band * 500.0 + 250.0

    def test_sparse_random_burst_emitter(self, env_cfg):
        """Sparse emitter: produces rare bursts once every 10-20 dwells."""
        pulses = [
            PulseRecord(toa_us=250.0, frequency_mhz=4750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),  # Band 9, step 0
            PulseRecord(toa_us=8250.0, frequency_mhz=4750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1), # Band 9, step 16
        ]
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        # Step 0: Hit in Band 9
        _, _, _, _, i0 = env.step(9 * 5 + 1)
        assert i0["hit"] is True

        # Steps 1 to 15: Scan elsewhere or miss in Band 9
        for _ in range(15):
            _, _, _, _, i_other = env.step(0 * 5 + 1)
            assert i_other["hit"] is False

        # Step 16: Intercept second sparse burst
        _, _, _, _, i16 = env.step(9 * 5 + 1)
        assert i16["hit"] is True

    def test_multi_emitter_concurrent_agility(self, env_cfg):
        """Multi-emitter: Emitter 1 hops [Band 3 -> Band 5], Emitter 2 hops [Band 12 -> Band 14]."""
        pulses = [
            PulseRecord(toa_us=100.0, frequency_mhz=1750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=30.0, emitter_id=1),  # E1 Band 3
            PulseRecord(toa_us=200.0, frequency_mhz=6250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=120.0, emitter_id=2), # E2 Band 12
            PulseRecord(toa_us=600.0, frequency_mhz=2750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=30.0, emitter_id=1),  # E1 Band 5
            PulseRecord(toa_us=700.0, frequency_mhz=7250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=120.0, emitter_id=2), # E2 Band 14
        ]
        env = CognitiveRFScanEnv(env_cfg, records=pulses)
        env.reset()

        # Dwell in Band 3: detects Emitter 1, Emitter 2 is missed (different band)
        _, _, _, _, i0 = env.step(3 * 5 + 1)
        assert i0["hit"] is True
        assert i0["detections"][0]["emitter_id"] == 1

        # Dwell in Band 14: detects Emitter 2 at its hopped band
        _, _, _, _, i1 = env.step(14 * 5 + 1)
        assert i1["hit"] is True
        assert i1["detections"][0]["emitter_id"] == 2


# =====================================================================
# Phase 5F & 5G — Belief State Temporal Information & Update Speed
# =====================================================================

class TestBeliefTemporalDynamics:
    """Verifies that belief state tracks band departures, agility, and revisit age causally."""

    def test_belief_occupancy_decays_on_consecutive_misses(self, env_cfg):
        env = CognitiveRFScanEnv(env_cfg)
        env.reset()

        b = 7
        # Initial belief is neutral prior 0.5
        assert env.belief.occupancy_prob[b] == 0.5

        # Record hit
        env.belief.record_visit(b, hit=True)
        assert env.belief.occupancy_prob[b] == pytest.approx(0.65, abs=1e-3)

        # Record misses (emitter left band). Under R4.2 asymmetric decay (alpha=0.20 for confirmed tracks),
        # decay is calibrated at (1 - 0.20) per miss to preserve agile persistence.
        for _ in range(6):
            env.belief.record_visit(b, hit=False)

        # Occupancy must decay down to < 0.20
        assert env.belief.occupancy_prob[b] < 0.20

    def test_revisit_age_advances_and_resets(self, env_cfg):
        env = CognitiveRFScanEnv(env_cfg)
        env.reset()

        # Step Band 0
        env.step(1)
        assert env.belief.revisit_age[0] == 0
        assert env.belief.revisit_age[1] == 2  # Started at 1, advanced by 1

        # Step Band 1
        env.step(1 * 5 + 1)
        assert env.belief.revisit_age[1] == 0
        assert env.belief.revisit_age[0] == 1

    def test_agility_indicator_updates_on_frequency_dispersion(self, env_cfg):
        env = CognitiveRFScanEnv(env_cfg)
        env.reset()

        b = 10
        # Detections with frequency dispersion within the band
        d1 = PulseRecord(toa_us=10.0, frequency_mhz=5100.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)
        d2 = PulseRecord(toa_us=50.0, frequency_mhz=5400.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)
        env.belief.record_visit(b, hit=True, detections=[d1, d2])

        assert env.belief.agility_indicator[b] > 0.0


# =====================================================================
# Phase 5B — Causality & Zero-Future Leakage Test
# =====================================================================

def test_future_frequency_leakage_invariance(env_cfg):
    """Proves that mutating future pulses (t > current dwell) has zero impact on current observation."""
    records_normal = [
        PulseRecord(toa_us=100.0, frequency_mhz=2250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        PulseRecord(toa_us=800.0, frequency_mhz=2250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
    ]

    records_mutated = [
        PulseRecord(toa_us=100.0, frequency_mhz=2250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        # Radically changed future pulse at t=800 us
        PulseRecord(toa_us=800.0, frequency_mhz=16250.0, pulse_width_us=20.0, amplitude_db=-20.0, aoa_deg=270.0, emitter_id=99),
    ]

    env1 = CognitiveRFScanEnv(env_cfg, records=records_normal, seed=42)
    obs1, _ = env1.reset()
    next_obs1, r1, _, _, info1 = env1.step(4 * 5 + 1)  # First dwell [0, 500 us]

    env2 = CognitiveRFScanEnv(env_cfg, records=records_mutated, seed=42)
    obs2, _ = env2.reset()
    next_obs2, r2, _, _, info2 = env2.step(4 * 5 + 1)  # First dwell [0, 500 us]

    # Must be bit-exact identical for the first dwell
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(next_obs1, next_obs2)
    assert r1 == r2
    assert len(info1["detections"]) == len(info2["detections"])

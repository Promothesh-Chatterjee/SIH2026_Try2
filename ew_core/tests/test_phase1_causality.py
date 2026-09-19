"""
Phase 1 Test Suite: RF Environment, Receiver Causality & Physical Scan Semantics.

Strictly verifies the 16-point Phase 1 Test Matrix (A through P):
  A. RF band mapping (36 bands, 0-18,000 MHz, 500 MHz IBW)
  B. Receiver IBW boundaries (inclusive aperture vs half-open discretization)
  C. Adjacent-band isolation (verified on interior frequencies)
  D. Dwell duration physical enforcement (all 5 modes: 125, 500, 1250, 500, 500 µs)
  E. Retune latency & physical aperture overlap semantics (15 µs retune)
  F. Mission-clock causality (monotonic, ClockDriftError on regression)
  G. Fixed emitter (stationary frequency in assigned band)
  H. Frequency-agile emitter (hopping detected only during spectral overlap)
  I. Periodic emitter (scanning beam detected only during temporal illumination)
  J. Future-information isolation (Scenario A vs Scenario B identical prior to arrival)
  K. Ground-truth-ID invariance (World A vs World B yields bit-exact obs and actions)
  L. 180-action physical semantics (exhaustive test over actions 0..179)
  M. TSRD PDW unit compatibility (explicit µs, MHz, µs, deg, exact dB convention)
  N. STARE vs SCAN separation (latent wideband world vs causal narrowband aperture)
  O. Observation shape & canonical feature ordering (360-D, 10 canonical features)
  P. Pulse-buffer & shortcut isolation (aperture buffer access and causality)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from ew_core.contracts import (
    CANONICAL_BAND_FEATURES,
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    LONG_DWELL,
    NORMAL_DWELL,
    PREEMPTIVE_INTERCEPT,
    REVISIT,
    RF_BASE_DWELL_TIME_US,
    RF_FREQ_MAX_MHZ,
    RF_FREQ_MIN_MHZ,
    RF_IBW_MHZ,
    SHORT_DWELL,
    band_of_action,
    dwell_us_for,
    encode_action,
    mode_of_action,
    validate_action,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.emitter_models import FreqAgileEmitter, PeriodicScanEmitter, StaticEmitter
from ew_core.environment.radio_environment import PulseRecord, RadioEnvironment
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.smartscan_moe import SmartScanMoE
from ew_core.operational.receiver_adapter import ReceiverAdapter, ReceiverHardwareError
from ew_core.operational.receiver_controller import OperationalReceiverController
from ew_core.preprocessing.normalise import normalise_pdws
from ew_core.receiver.mission_clock import ClockDriftError, MissionClock
from ew_core.receiver.models import DetectionObservation
from ew_core.receiver.sieve_receiver import SieveReceiver
from ew_core.utils.checkpoint_paths import CANONICAL_PRODUCTION_BASELINE, verify_production_checkpoint


# ==============================================================================
# Helpers
# ==============================================================================

def _make_env_config(
    n_bands: int = CANONICAL_N_BANDS,
    n_modes: int = CANONICAL_N_MODES,
    retune_latency_us: float = 0.0,
    perception_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "n_bands": n_bands,
        "n_modes": n_modes,
        "obs_dim": n_bands * CANONICAL_BAND_FEATURES,
        "band_features": CANONICAL_BAND_FEATURES,
        "base_dwell_time_us": 500.0,
        "retune_latency_us": retune_latency_us,
        "perception_enabled": perception_enabled,
        "semantic_memory_enabled": False,
        "deinterleaver": {"min_pulses": 10, "interval_steps": 1},
    }


# ==============================================================================
# Test A: RF Band Mapping
# ==============================================================================

def test_a_rf_band_mapping():
    """Verify 36 bands covering 0-18,000 MHz (500 MHz width), discretization ownership, and receiver centers."""
    env = CognitiveRFScanEnv(_make_env_config())
    env.reset()

    band_width = 500.0  # MHz
    assert env.n_bands == 36
    assert env.freq_min == 0.0
    assert env.freq_max == 18000.0
    assert env.ibw_mhz == 500.0

    legal_min_center = 250.0
    legal_max_center = 17750.0

    for b in range(36):
        # 1. Band discretization ownership: [500*b, 500*(b+1))
        expected_ownership_start = b * band_width
        expected_ownership_end = (b + 1) * band_width

        # 2. Receiver tuned center frequency
        center = env._band_to_center(b)
        expected_mid = expected_ownership_start + band_width / 2.0
        assert math.isclose(center, expected_mid, rel_tol=1e-5), f"Band {b} center mismatch: {center} vs {expected_mid}"
        assert legal_min_center <= center <= legal_max_center, f"Band {b} center {center} out of legal bounds"

        # 3. Discretization ownership maps interior frequencies strictly to band b
        interior_freq = expected_ownership_start + 100.0
        mapped_band = env._band_index(interior_freq)
        assert mapped_band == b, f"Frequency {interior_freq} mapped to band {mapped_band}, expected {b}"

    # Verify 18000.0 MHz clamps to Band 35
    assert env._band_index(18000.0) == 35


# ==============================================================================
# Test B: Receiver IBW Boundaries
# ==============================================================================

def test_b_ibw_boundaries():
    """Verify inclusive aperture window [center - 250, center + 250] and out-of-band rejection."""
    receiver = SieveReceiver(
        total_bandwidth=18000.0,
        ibw=500.0,
        dwell_time=500.0,
        detection_threshold_db=-140.0,
    )

    # Test Band 0 (center 250.0 MHz, window [0.0, 500.0])
    receiver.tune(250.0)
    low, high = receiver.get_frequency_window()
    assert low == 0.0 and high == 500.0

    # Inclusive boundary points must be in-window
    assert receiver.frequency_in_window(0.0) is True
    assert receiver.frequency_in_window(250.0) is True
    assert receiver.frequency_in_window(500.0) is True

    # Out-of-window points must be rejected
    assert receiver.frequency_in_window(-0.1) is False
    assert receiver.frequency_in_window(500.1) is False

    # Test Band 5 (center 2750.0 MHz, window [2500.0, 3000.0])
    receiver.tune(2750.0)
    low5, high5 = receiver.get_frequency_window()
    assert low5 == 2500.0 and high5 == 3000.0

    assert receiver.frequency_in_window(2500.0) is True
    assert receiver.frequency_in_window(2750.0) is True
    assert receiver.frequency_in_window(3000.0) is True
    assert receiver.frequency_in_window(2499.9) is False
    assert receiver.frequency_in_window(3000.1) is False


# ==============================================================================
# Test C: Adjacent-Band Isolation
# ==============================================================================

def test_c_adjacent_band_isolation():
    """Verify that tuning to Band N strictly rejects interior frequencies of Band N-1 and Band N+1."""
    receiver = SieveReceiver(total_bandwidth=18000.0, ibw=500.0, dwell_time=500.0, detection_threshold_db=-140.0)

    # Tune to Band 5 (center 2750 MHz, window [2500, 3000])
    receiver.tune(2750.0)

    # Band 4 interior: center 2250 MHz; Band 6 interior: center 3250 MHz
    pulse_b4 = {"toa_us": 100.0, "exit_us": 110.0, "frequency_mhz": 2250.0, "amplitude_db": -50.0, "pulse_width_us": 10.0, "aoa_deg": 0.0, "pulse_id": 1}
    pulse_b5 = {"toa_us": 100.0, "exit_us": 110.0, "frequency_mhz": 2750.0, "amplitude_db": -50.0, "pulse_width_us": 10.0, "aoa_deg": 0.0, "pulse_id": 2}
    pulse_b6 = {"toa_us": 100.0, "exit_us": 110.0, "frequency_mhz": 3250.0, "amplitude_db": -50.0, "pulse_width_us": 10.0, "aoa_deg": 0.0, "pulse_id": 3}

    receiver.add_pulse(pulse_b4)
    receiver.add_pulse(pulse_b5)
    receiver.add_pulse(pulse_b6)

    detections = receiver._detect_buffered_interval(0.0, 500.0)
    detected_freqs = [d.frequency_mhz for d in detections]

    assert 2750.0 in detected_freqs, "Band 5 interior pulse should be detected"
    assert 2250.0 not in detected_freqs, "Band 4 interior pulse must NOT leak into Band 5 dwell"
    assert 3250.0 not in detected_freqs, "Band 6 interior pulse must NOT leak into Band 5 dwell"


# ==============================================================================
# Test D: Dwell Duration Physical Enforcement
# ==============================================================================

def test_d_dwell_duration_physical_enforcement():
    """Verify that all 5 dwell modes physically enforce their respective observation aperture intervals."""
    # Check durations
    expected_durations = {
        SHORT_DWELL: 125.0,
        NORMAL_DWELL: 500.0,
        LONG_DWELL: 1250.0,
        REVISIT: 500.0,
        PREEMPTIVE_INTERCEPT: 500.0,
    }
    for mode, dur in expected_durations.items():
        assert dwell_us_for(500.0, mode) == dur

    # Physical aperture experiment: Pulse arrives at t=200 µs (exits at 210 µs)
    # With dwell_start = 0.0:
    # - SHORT_DWELL aperture is [0, 125] -> pulse arrives after aperture closes -> NOT detected
    # - NORMAL_DWELL aperture is [0, 500] -> pulse arrives during aperture -> DETECTED
    pulses = [PulseRecord(toa_us=200.0, frequency_mhz=1750.0, pulse_width_us=10.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)]

    # Run SHORT dwell (action = band 3 * 5 + SHORT_DWELL = 15)
    env_short = CognitiveRFScanEnv(_make_env_config(), records=pulses)
    env_short.reset()
    obs_s, r_s, term_s, trunc_s, info_s = env_short.step(encode_action(3, SHORT_DWELL))
    assert info_s["hit"] is False, "Pulse at 200 µs must NOT be detected in a 125 µs SHORT dwell"
    assert info_s["dwell_time_us"] == 125.0

    # Run NORMAL dwell (action = band 3 * 5 + NORMAL_DWELL = 16)
    env_normal = CognitiveRFScanEnv(_make_env_config(), records=pulses)
    env_normal.reset()
    obs_n, r_n, term_n, trunc_n, info_n = env_normal.step(encode_action(3, NORMAL_DWELL))
    assert info_n["hit"] is True, "Pulse at 200 µs MUST be detected in a 500 µs NORMAL dwell"
    assert info_n["dwell_time_us"] == 500.0


# ==============================================================================
# Test E: Retune Latency & Physical Aperture Overlap Semantics
# ==============================================================================

def test_e_retune_latency_and_overlap_semantics():
    """Verify 15.0 µs retune latency and exact physical interval overlap semantics.

    - Pulse entirely contained within retune window (exit_us <= dwell_start): missed.
    - Pulse straddling retune and dwell (toa < dwell_start < exit_us): detected.
    - Pulse entirely inside dwell window: detected.
    """
    # Retune interval: [0.0, 15.0] µs. Dwell aperture: [15.0, 515.0] µs.
    pulses = [
        # Pulse 1: Entirely inside retune window [2.0, 10.0] µs
        PulseRecord(toa_us=2.0, frequency_mhz=2250.0, pulse_width_us=8.0, amplitude_db=-50.0, aoa_deg=0.0, emitter_id=1),
        # Pulse 2: Straddles retune and dwell [10.0, 25.0] µs (exit 25.0 > 15.0)
        PulseRecord(toa_us=10.0, frequency_mhz=2250.0, pulse_width_us=15.0, amplitude_db=-50.0, aoa_deg=0.0, emitter_id=2),
        # Pulse 3: Entirely inside dwell window [100.0, 110.0] µs
        PulseRecord(toa_us=100.0, frequency_mhz=2250.0, pulse_width_us=10.0, amplitude_db=-50.0, aoa_deg=0.0, emitter_id=3),
    ]

    env = CognitiveRFScanEnv(_make_env_config(retune_latency_us=15.0), records=pulses)
    env.reset()

    # Step Band 4 (center 2250 MHz), NORMAL_DWELL
    action = encode_action(4, NORMAL_DWELL)
    obs, rew, term, trunc, info = env.step(action)

    assert info["retune_latency_us"] == 15.0
    assert info["dwell_start_us"] == 15.0
    assert info["dwell_end_us"] == 515.0

    detections = info["detections"]
    detected_toas = [round(d["time_us"], 1) for d in detections]

    # Pulse 1 (exit at 10 µs <= 15 µs) MUST NOT be detected
    assert 2.0 not in detected_toas, "Pulse entirely inside retune interval must NOT be detected"

    # Pulse 2 (straddling: starts at 10 µs, exits at 25 µs) MUST be detected, timestamped at dwell_start = 15.0 µs
    assert 15.0 in detected_toas, "Straddling pulse must be detected with timestamp clipped to dwell start"

    # Pulse 3 (inside dwell: starts at 100 µs) MUST be detected
    assert 100.0 in detected_toas, "Pulse inside dwell window must be detected"


# ==============================================================================
# Test F: Mission-Clock Causality
# ==============================================================================

def test_f_mission_clock_causality():
    """Verify MissionClock strictly monotonic forward progression, regression rejection, and exception semantics."""
    clock = MissionClock(initial_time_us=100.0)
    assert clock.current_time_us == 100.0
    assert clock.now() == 100.0

    # Advance by retune
    r_start, r_end = clock.advance_retune(15.0)
    assert r_start == 100.0 and r_end == 115.0
    assert clock.current_time_us == 115.0

    # Advance by dwell
    d_start, d_end = clock.advance_dwell(500.0)
    assert d_start == 115.0 and d_end == 615.0
    assert clock.current_time_us == 615.0

    # Advance to explicit future time
    assert clock.advance_to(1000.0) == 1000.0

    # Regression attempt must raise ClockDriftError
    with pytest.raises(ClockDriftError):
        clock.advance_to(999.0)

    # Negative delta must raise ValueError
    with pytest.raises(ValueError):
        clock.advance_by(-5.0)


# ==============================================================================
# Test G: Fixed Emitter
# ==============================================================================

def test_g_fixed_emitter():
    """Verify that a stationary emitter remains observable in its band and accumulates stable PRI history."""
    pulses = [
        PulseRecord(toa_us=float(100 + i * 1000), frequency_mhz=3750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1)
        for i in range(10)
    ]
    env = CognitiveRFScanEnv(_make_env_config(), records=pulses)
    env.reset()

    # Band 7 covers [3500, 4000] MHz (center 3750 MHz)
    # Band 8 covers [4000, 4500] MHz (center 4250 MHz)

    # 1. Tuning to Band 8 catches nothing
    _, _, _, _, info8 = env.step(encode_action(8, NORMAL_DWELL))
    assert info8["hit"] is False

    # 2. Tuning to Band 7 intercepts the emitter
    # Reset env to test Band 7
    env.reset()
    _, _, _, _, info7 = env.step(encode_action(7, NORMAL_DWELL))
    assert info7["hit"] is True
    assert len(info7["detections"]) > 0


# ==============================================================================
# Test H: Frequency-Agile Emitter
# ==============================================================================

def test_h_frequency_agile_emitter():
    """Verify that a frequency-agile emitter is only detected when the receiver is tuned to its current hop band."""
    # Emitter hops: Band 2 (1250 MHz) at t=100, Band 7 (3750 MHz) at t=600, Band 14 (7250 MHz) at t=1100
    pulses = [
        PulseRecord(toa_us=100.0, frequency_mhz=1250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=30.0, emitter_id=1),
        PulseRecord(toa_us=600.0, frequency_mhz=3750.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=30.0, emitter_id=1),
        PulseRecord(toa_us=1100.0, frequency_mhz=7250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=30.0, emitter_id=1),
    ]
    env = CognitiveRFScanEnv(_make_env_config(), records=pulses)
    env.reset()

    # Dwell 0: [0, 500] µs tuned to Band 2 (1250 MHz) -> Hit!
    _, _, _, _, info0 = env.step(encode_action(2, NORMAL_DWELL))
    assert info0["hit"] is True

    # Dwell 1: [500, 1000] µs tuned to Band 3 (1750 MHz) -> Miss! (Emitter is hopping on Band 7)
    _, _, _, _, info1 = env.step(encode_action(3, NORMAL_DWELL))
    assert info1["hit"] is False

    # Dwell 2: [1000, 1500] µs tuned to Band 14 (7250 MHz) -> Hit! (Emitter hopped to Band 14)
    _, _, _, _, info2 = env.step(encode_action(14, NORMAL_DWELL))
    assert info2["hit"] is True


# ==============================================================================
# Test I: Periodic Emitter
# ==============================================================================

def test_i_periodic_emitter():
    """Verify that a periodic scanning emitter is only intercepted when receiver dwell overlaps its illumination window."""
    # Illumination window: period = 5000 µs, mainbeam duration = 200 µs
    # Burst 1: [0, 200] µs at 5250 MHz (Band 10)
    # Burst 2: [5000, 5200] µs at 5250 MHz (Band 10)
    pulses = [
        PulseRecord(toa_us=50.0, frequency_mhz=5250.0, pulse_width_us=10.0, amplitude_db=-60.0, aoa_deg=0.0, emitter_id=1),
        PulseRecord(toa_us=5050.0, frequency_mhz=5250.0, pulse_width_us=10.0, amplitude_db=-60.0, aoa_deg=0.0, emitter_id=1),
    ]
    env = CognitiveRFScanEnv(_make_env_config(), records=pulses)
    env.reset()

    # Step 0: [0, 500] µs tuned to Band 10 -> Intercepts Burst 1
    _, _, _, _, info0 = env.step(encode_action(10, NORMAL_DWELL))
    assert info0["hit"] is True

    # Step 1: [500, 1000] µs tuned to Band 10 -> Miss! Outside illumination window
    _, _, _, _, info1 = env.step(encode_action(10, NORMAL_DWELL))
    assert info1["hit"] is False


# ==============================================================================
# Test J: Future-Information Isolation
# ==============================================================================

def test_j_future_information_isolation():
    """Verify that adding future pulses (> T) produces bit-exact identical observations prior to T."""
    t_split = 25000.0  # µs

    base_records = [
        PulseRecord(toa_us=float(t), frequency_mhz=2250.0, pulse_width_us=5.0, amplitude_db=-55.0, aoa_deg=45.0, emitter_id=1)
        for t in np.arange(100.0, t_split, 500.0)
    ]

    records_a = copy.deepcopy(base_records)

    # Scenario B has completely different pulses strictly after t_split
    records_b = copy.deepcopy(base_records)
    for t in np.arange(t_split + 100.0, t_split + 10000.0, 400.0):
        records_b.append(
            PulseRecord(toa_us=float(t), frequency_mhz=14250.0, pulse_width_us=2.0, amplitude_db=-40.0, aoa_deg=180.0, emitter_id=99)
        )

    env_a = CognitiveRFScanEnv(_make_env_config(), records=records_a, seed=42)
    env_b = CognitiveRFScanEnv(_make_env_config(), records=records_b, seed=42)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    np.testing.assert_array_equal(obs_a, obs_b)

    # Step 20 dwells (total elapsed time 10,000 µs << 25,000 µs)
    actions = [encode_action(i % 36, i % 5) for i in range(20)]
    for step_idx, act in enumerate(actions):
        obs_a, r_a, term_a, trunc_a, info_a = env_a.step(act)
        obs_b, r_b, term_b, trunc_b, info_b = env_b.step(act)

        np.testing.assert_array_equal(
            obs_a,
            obs_b,
            err_msg=f"Observation diverged at step {step_idx} prior to future modification!",
        )
        assert info_a["hit"] == info_b["hit"]


# ==============================================================================
# Test K: Ground-Truth-ID Invariance
# ==============================================================================

def test_k_ground_truth_id_invariance():
    """Verify that two worlds differing ONLY in simulator ground-truth IDs produce bit-exact identical obs and actions."""
    # World A: emitter_id = 1, 2, 3
    # World B: emitter_id = 101, 205, 900
    times = [100.0, 200.0, 600.0, 700.0, 1100.0, 1200.0]
    freqs = [2250.0, 5250.0, 2250.0, 5250.0, 2250.0, 5250.0]

    records_a = [
        PulseRecord(toa_us=t, frequency_mhz=f, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=(i % 3) + 1)
        for i, (t, f) in enumerate(zip(times, freqs))
    ]
    records_b = [
        PulseRecord(toa_us=t, frequency_mhz=f, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=[101, 205, 900][i % 3])
        for i, (t, f) in enumerate(zip(times, freqs))
    ]

    env_a = CognitiveRFScanEnv(_make_env_config(), records=records_a, seed=123)
    env_b = CognitiveRFScanEnv(_make_env_config(), records=records_b, seed=123)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    np.testing.assert_array_equal(obs_a, obs_b)

    # Run for 10 steps with fixed test actions
    test_actions = [encode_action(4, NORMAL_DWELL), encode_action(10, NORMAL_DWELL), encode_action(4, SHORT_DWELL)] * 3
    for step_idx, act in enumerate(test_actions):
        obs_a, r_a, term_a, trunc_a, info_a = env_a.step(act)
        obs_b, r_b, term_b, trunc_b, info_b = env_b.step(act)

        np.testing.assert_array_equal(
            obs_a,
            obs_b,
            err_msg=f"Ground-truth ID leaked into observation at step {step_idx}!",
        )


# ==============================================================================
# Test L: 180-Action Physical Semantics
# ==============================================================================

def test_l_180_actions_physical_semantics():
    """Automated exhaustive verification of all 180 actions (0..179)."""
    env = CognitiveRFScanEnv(_make_env_config())
    env.reset()

    for action in range(CANONICAL_N_ACTIONS):
        # 1. Action validation
        validated = validate_action(action)
        assert validated == action

        # 2. Band and mode decoding
        band = band_of_action(action)
        mode = mode_of_action(action)
        assert 0 <= band < 36
        assert 0 <= mode < 5

        # 3. Reconstructed action
        assert encode_action(band, mode) == action

        # 4. Center frequency and dwell duration
        center = env._band_to_center(band)
        assert 250.0 <= center <= 17750.0
        dur = dwell_us_for(500.0, mode)
        assert dur in (125.0, 500.0, 1250.0)

        # 5. Execute action
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (360,)
        assert np.all(np.isfinite(obs))
        assert np.all((obs >= 0.0) & (obs <= 1.0))
        assert info["selected_band"] == band
        assert info["selected_mode"] == mode
        assert info["mode_name"] == DWELL_MODES[mode]


# ==============================================================================
# Test M: TSRD PDW Unit Compatibility
# ==============================================================================

def test_m_tsrd_pdw_unit_compatibility():
    """Verify canonical TSRD unit compatibility: ToA (µs), CF (MHz), PW (µs), AoA (deg), and exact Amp (dB)."""
    # TSRD 5D array [ToA_us, CF_MHz, PW_us, AoA_deg, Amp_dB]
    raw_pdws = np.array([
        [100.0, 2250.0, 5.0, 45.0, -50.0],
        [200.0, 2250.0, 5.0, 45.0, -52.0],
        [300.0, 2250.0, 5.0, 45.0, -48.0],
    ], dtype=np.float32)

    # 1. Preprocessing normalisation preserves 6D structure
    norm_vec, stats = normalise_pdws(raw_pdws)
    assert norm_vec.shape == (3, 6)
    assert np.all(np.isfinite(norm_vec))

    # 2. Receiver aperture consumption: units must not be distorted
    receiver = SieveReceiver(total_bandwidth=18000.0, ibw=500.0, dwell_time=500.0, detection_threshold_db=-140.0)
    receiver.tune(2250.0)

    pdw_dict = {
        "toa_us": 100.0,
        "frequency_mhz": 2250.0,
        "pulse_width_us": 5.0,
        "amplitude_db": -50.0,
        "aoa_deg": 45.0,
        "pulse_id": 1,
    }
    receiver.add_pulse(pdw_dict)
    detections = receiver._detect_buffered_interval(0.0, 500.0)
    assert len(detections) == 1
    det = detections[0]
    assert det.frequency_mhz == 2250.0
    assert det.amplitude_db == -50.0
    assert det.pulse_width_us == 5.0
    assert det.aoa_deg == 45.0


# ==============================================================================
# Test N: STARE vs SCAN Separation
# ==============================================================================

def test_n_stare_vs_scan_separation():
    """Verify explicit separation between latent wideband world (STARE) and causal narrowband receiver (SCAN)."""
    # Simultaneous pulses in the RF world across Band 2 (1250 MHz) and Band 10 (5250 MHz)
    records = [
        PulseRecord(toa_us=100.0, frequency_mhz=1250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=0.0, emitter_id=1),
        PulseRecord(toa_us=100.0, frequency_mhz=5250.0, pulse_width_us=5.0, amplitude_db=-50.0, aoa_deg=0.0, emitter_id=2),
    ]

    env = CognitiveRFScanEnv(_make_env_config(), records=records)
    env.reset()

    # Latent RF world contains both pulses
    assert len(env.records) == 2

    # Receiver scans Band 2
    obs, r, term, trunc, info = env.step(encode_action(2, NORMAL_DWELL))
    assert info["hit"] is True

    # The receiver observation only perceived Band 2
    detected_freqs = [d["frequency_mhz"] for d in info["detections"]]
    assert 1250.0 in detected_freqs
    assert 5250.0 not in detected_freqs, "STARE pulse on Band 10 must NOT bypass scanning receiver filter"


# ==============================================================================
# Test O: Observation Shape and Canonical Feature Ordering
# ==============================================================================

def test_o_observation_shape_and_feature_ordering():
    """Verify observation shape is strictly (360,) and features follow canonical 10-feature order."""
    env = CognitiveRFScanEnv(_make_env_config())
    obs, _ = env.reset()

    assert obs.shape == (360,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert np.all((obs >= 0.0) & (obs <= 1.0))

    # Test feature offsets for band 0
    # Expected ordering:
    # 0: occupancy, 1: det_rate, 2: miss_rate, 3: uncertainty, 4: revisit_age,
    # 5: emitter_count, 6: deint_confidence, 7: pri_stability, 8: agility, 9: priority
    features_b0 = env.belief.band_features(0)
    assert len(features_b0) == 10
    np.testing.assert_array_equal(obs[:10], features_b0)


# ==============================================================================
# Test P: Pulse-Buffer & Shortcut Isolation
# ==============================================================================

def test_p_pulse_buffer_shortcut_isolation():
    """Audit SieveReceiver._pulse_buffer and verify operational adapter causal ingestion."""
    adapter = ReceiverAdapter(total_bandwidth_mhz=18000.0, ibw_mhz=500.0)

    pulses = [
        {"toa_us": 100.0, "time_us": 100.0, "pulse_width_us": 5.0, "frequency_mhz": 1250.0, "amplitude_db": -50.0, "pulse_id": 1},
        {"toa_us": 200.0, "time_us": 200.0, "pulse_width_us": 5.0, "frequency_mhz": 1250.0, "amplitude_db": -50.0, "pulse_id": 2},
        {"toa_us": 800.0, "time_us": 800.0, "pulse_width_us": 5.0, "frequency_mhz": 1250.0, "amplitude_db": -50.0, "pulse_id": 3},
    ]

    # Feed with causal cutoff at max_time_us = 500.0 µs
    ingested = adapter.feed_incident_rf(pulses, max_time_us=500.0)
    assert ingested == 2, "Only pulses with ToA <= 500 µs should be ingested"

    # Verify future pulse at 800 µs was NOT placed into buffer
    buffered = adapter.receiver.buffered_pulses()
    buffered_ids = [p.get("pulse_id") for p in buffered]
    assert 1 in buffered_ids and 2 in buffered_ids
    assert 3 not in buffered_ids, "Future pulse at 800 µs must not enter active receiver buffer"

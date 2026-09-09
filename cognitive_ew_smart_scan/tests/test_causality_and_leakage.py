"""
Causality and Data Leakage Audit Test Suite.

Proves that:
1. Online action selection a_t is strictly independent of future pulse arrivals (t > t_dwell_start).
2. The agent has zero access to ground-truth scenario labels, future frequencies, or future ToAs.
3. Track classifications and temporal reservations are strictly causal from past PDW history.
4. Spatial sector prioritization operates on measured AoA circular statistics, not ground truth IDs.
"""

import copy
import math
import numpy as np
import pytest
import torch

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, NORMAL_DWELL, SHORT_DWELL, LONG_DWELL
from src.cognitive.behavior_manager import AdaptiveBehaviorManager, EmitterBehavior
from src.cognitive.reservation_manager import ReservationManager, ReservationStatus
from src.cognitive.spatial_tracker import SpatialTracker
from src.cognitive.temporal_predictor import TemporalPredictor, TrackTemporalState
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.environment.radio_environment import PulseRecord
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv


def test_future_pulse_invariance_leakage_audit():
    """Test that modifying or truncating future pulses has ZERO effect on current action selection."""
    rng = np.random.default_rng(42)
    time_horizon_us = 50_000.0

    # Variant A: Standard future pulses
    time_horizon_us = 100_000.0
    future_diverge_us = 50_000.0

    # Build base pulse train
    base_records = []
    for t in np.arange(100.0, time_horizon_us, 500.0):
        base_records.append(
            PulseRecord(
                toa_us=float(t),
                frequency_mhz=2250.0,
                pulse_width_us=2.0,
                amplitude_db=-55.0,
                aoa_deg=45.0,
                emitter_id=0,
            )
        )

    records_a = copy.deepcopy(base_records)

    # Variant B: Completely altered future pulses after t=50,000 us (different frequencies, ToAs, amplitudes)
    records_b = []
    for r in base_records:
        if r.toa_us <= future_diverge_us:
            records_b.append(copy.deepcopy(r))
        else:
            records_b.append(
                PulseRecord(
                    toa_us=r.toa_us + 123.4,
                    frequency_mhz=14250.0,  # Completely different band!
                    pulse_width_us=10.0,
                    amplitude_db=-90.0,
                    aoa_deg=270.0,
                    emitter_id=99,
                )
            )

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": 360,
        "semantic_memory_enabled": False,
        "base_dwell_time_us": 500.0,
    }

    ckpt = torch.load("checkpoints/scheduler/checkpoint_gate_110000.pt", map_location="cpu", weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, n_actions=180, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    moe_cfg = {
        "enable_t0": True,
        "enable_t1": True,
        "enable_spatial": True,
        "alpha_dirichlet": 0.10,
        "enable_guard": True,
        "exploration_guard_confidence": 0.45,
        "exploration_guard_eta_us": 1000.0,
        "tau": 0.0,
    }

    actions_a = []
    actions_b = []

    # Run variant A up to 20 dwells (t <= 25,000 us << 50,000 us)
    env_a = CognitiveRFScanEnv(env_cfg, records=records_a, seed=42, semantic_memory_path=":memory:")
    obs_a, _ = env_a.reset()
    agent_a = SmartScanMoE(copy.deepcopy(drqn), moe_cfg)
    hidden_a = None
    for step in range(20):
        action, hidden_a, attr = agent_a.select_action(obs_a, hidden_a)
        actions_a.append(int(action))
        obs_a, rew, term, trunc, info = env_a.step(action)
        agent_a.update_result(info["hit"], int(action // 5), detections=info.get("detections", []), current_time=float(env_a.receiver.current_time_us))
        agent_a.update(action)

    # Run variant B up to 20 dwells (t <= 25,000 us << 50,000 us)
    env_b = CognitiveRFScanEnv(env_cfg, records=records_b, seed=42, semantic_memory_path=":memory:")
    obs_b, _ = env_b.reset()
    agent_b = SmartScanMoE(copy.deepcopy(drqn), moe_cfg)
    hidden_b = None
    for step in range(20):
        action, hidden_b, attr = agent_b.select_action(obs_b, hidden_b)
        actions_b.append(int(action))
        obs_b, rew, term, trunc, info = env_b.step(action)
        agent_b.update_result(info["hit"], int(action // 5), detections=info.get("detections", []), current_time=float(env_b.receiver.current_time_us))
        agent_b.update(action)

    # Decision-for-decision equivalence check prior to divergence time
    assert actions_a == actions_b, f"Causality violation! Actions diverged prior to future modification: {actions_a} vs {actions_b}"


def test_behavior_manager_pure_historical_causality():
    """Verify AdaptiveBehaviorManager operates strictly on past observed pulses without oracle state."""
    abm = AdaptiveBehaviorManager()
    track = TrackTemporalState(track_id=1, n_bands=36)

    # Initially unknown
    prof0 = abm.classify_track(track)
    assert prof0.behavior == EmitterBehavior.UNKNOWN

    # Ingest 5 pulses at 800 us PRI across 4 bands (Slow hopper)
    t = 1000.0
    bands = [3, 11, 19, 27, 3]
    for b in bands:
        track.update(toa=t, freq_mhz=float(b * 500.0 + 250.0), band=b)
        t += 800.0

    prof = abm.classify_track(track)
    assert prof.behavior == EmitterBehavior.SLOW_HOPPER
    assert prof.requires_reservation is True
    assert prof.pri_us == pytest.approx(800.0, rel=0.05)


def test_reservation_manager_deadline_causality():
    """Verify ReservationManager does NOT make reservations active before their retune deadline."""
    rm = ReservationManager(retune_latency_us=15.0, default_window_half_width_us=50.0, pre_arrival_lead_us=35.0)

    # Create reservation for arrival at t=10,000 us
    # window = [9,950, 10,050], deadline = 9,950 - 15 - 35 = 9,900 us
    res = rm.create_or_update_reservation(
        track_id=10,
        target_band=4,
        expected_toa_us=10000.0,
        confidence=0.85,
        target_mode=NORMAL_DWELL,
    )
    assert res.status == ReservationStatus.PENDING

    # At t=9,000 us (far ahead): must be PENDING (not actionable)
    act_9000 = rm.get_actionable_reservation(current_time_us=9000.0, dwell_duration_us=500.0)
    assert act_9000 is None

    # At t=9,900 us (exact deadline): must transition to ACTIVE
    act_9900 = rm.get_actionable_reservation(current_time_us=9900.0, dwell_duration_us=500.0)
    assert act_9900 is not None
    assert act_9900.reservation_id == res.reservation_id
    assert act_9900.target_band == 4


def test_spatial_tracker_pure_measurement_statistics():
    """Verify SpatialTracker relies solely on circular statistics from past measured AoA samples."""
    st = SpatialTracker(n_sectors=12)

    # Ingest noisy AoA measurements around 45°
    rng = np.random.default_rng(123)
    t = 100.0
    for _ in range(10):
        measured_aoa = float(rng.normal(45.0, 2.0) % 360.0)
        st.update_from_track(track_id=42, new_aoa_deg=measured_aoa, current_time_us=t)
        t += 250.0

    sb = st.beliefs[42]
    assert abs(sb.mean_aoa_deg - 45.0) < 2.0
    assert sb.confidence > 0.95  # Tight circular cluster
    assert sb.sector_index == 1  # 45 deg // 30 deg = sector 1

    # Sector weight scaling
    st.set_sector_weight(1, 3.0)
    prio = st.get_spatial_priority(track_id=42, current_time_us=t)
    assert prio > 2.5  # Boosted by sector weight, not by privileged ground truth

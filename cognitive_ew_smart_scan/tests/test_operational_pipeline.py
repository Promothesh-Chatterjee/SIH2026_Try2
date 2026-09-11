"""
Unit tests for End-to-End Operational Pipeline & Invariant Governance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_OBS_DIM
from src.environment.cognitive_rf_scan_env import BeliefState
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.operational.receiver_adapter import ReceiverAdapter, ReceiverHardwareError
from src.operational.receiver_controller import OperationalReceiverController
from src.operational.state_builder import OperationalStateBuilder, StateContractError


def test_state_builder_canonical_contract():
    builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)
    builder.reset()
    state = builder.build_state(current_time_us=0.0)

    assert state.shape == (CANONICAL_OBS_DIM,)
    assert state.dtype == np.float32
    assert np.all(np.isfinite(state))
    assert np.all(state >= 0.0) and np.all(state <= 1.0)


def test_state_builder_matches_belief_state_priors():
    builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)
    builder.reset()
    obs_builder = builder.build_state(0.0)

    belief = BeliefState(n_bands=CANONICAL_N_BANDS)
    obs_belief = np.concatenate([belief.band_features(b) for b in range(CANONICAL_N_BANDS)])

    diff = np.max(np.abs(obs_builder - obs_belief))
    assert diff == pytest.approx(0.0, abs=1e-5)


def test_state_builder_asymmetric_miss_decay():
    builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS, ema_alpha=0.30, ema_alpha_miss_confirmed=0.20)
    builder.reset()

    band = 5
    # Step 1: Hit
    builder.record_dwell_outcome(band, hit=True, current_time_us=1000.0)
    # 0.5 * 0.7 + 1.0 * 0.3 = 0.65
    assert builder.ema_occupancy[band] == pytest.approx(0.65, abs=1e-3)

    # Step 2: Miss on confirmed track -> decay at 0.20
    builder.record_dwell_outcome(band, hit=False, current_time_us=2000.0)
    # 0.65 * (1 - 0.20) = 0.52
    assert builder.ema_occupancy[band] == pytest.approx(0.52, abs=1e-3)


def test_receiver_adapter_causal_feed_and_disconnect():
    adapter = ReceiverAdapter()
    adapter.reset()

    pulses = [
        {"time_us": 100.0, "frequency_mhz": 1200.0, "pulse_width_us": 1.0, "amplitude_db": -40.0},
        {"time_us": 500.0, "frequency_mhz": 1200.0, "pulse_width_us": 1.0, "amplitude_db": -40.0},
        {"time_us": 2000.0, "frequency_mhz": 1200.0, "pulse_width_us": 1.0, "amplitude_db": -40.0},
    ]

    # Causal feed up to 600 us -> should ingest 2 pulses
    ingested = adapter.feed_incident_rf(pulses, max_time_us=600.0)
    assert ingested == 2

    # Subsequent feed up to 2500 us -> should ingest remaining 1 pulse without re-adding prior pulses
    ingested2 = adapter.feed_incident_rf(pulses, max_time_us=2500.0)
    assert ingested2 == 1

    # Disconnect fail-safe
    adapter.set_connected(False)
    with pytest.raises(ReceiverHardwareError):
        adapter.tune(1500.0)
    with pytest.raises(ReceiverHardwareError):
        adapter.execute_dwell(0.0, 500.0)


def test_operational_receiver_controller_monotonic_clock():
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=64, lstm_layers=1)
    scheduler = build_baseline("drqn", n_bands=36, n_modes=5, drqn=drqn)

    controller = OperationalReceiverController(scheduler=scheduler)
    controller.start_mission(initial_time_us=0.0)

    pulses = [
        {"time_us": 50.0, "frequency_mhz": 250.0, "pulse_width_us": 1.0, "amplitude_db": -40.0},
        {"time_us": 600.0, "frequency_mhz": 750.0, "pulse_width_us": 1.0, "amplitude_db": -40.0},
    ]

    prior_time = 0.0
    for step in range(10):
        frame = controller.execute_operational_step(scenario_pulses=pulses)
        assert frame.timestamp_us > prior_time
        assert frame.dwell_end_us > frame.dwell_start_us
        assert frame.retune_latency_us == pytest.approx(15.0)
        assert frame.dwell_start_us >= prior_time + 15.0
        # Zero ground truth leakage
        for d in frame.detections:
            assert "emitter_id" not in d
            assert "ground_truth_emitter_id" not in d
        prior_time = frame.timestamp_us

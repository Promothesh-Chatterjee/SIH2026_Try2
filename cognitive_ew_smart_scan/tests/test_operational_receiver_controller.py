"""Unit tests for OperationalReceiverController across Progressive Validation Levels 1, 2, and 3."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from src.environment.radio_environment import PulseRecord
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.operational.receiver_controller import OperationalReceiverController, ReceiverTelemetryFrame


class MockDRQN:
    n_bands = 36
    n_modes = 5
    n_actions = 180
    def __call__(self, obs, hidden=None):
        return None, None, hidden


def test_level1_synthetic_pdw_stream():
    """Level 1: Validate controller with synthetic streaming pulses."""
    moe = SmartScanMoE(MockDRQN())
    moe.default_tau = 0.0
    moe.set_stage3_modes(enable_t1=True, enable_spatial=True)

    controller = OperationalReceiverController(
        moe_scheduler=moe,
        retune_latency_us=15.0,
    )
    controller.reset()

    pulses = [
        PulseRecord(toa_us=500.0, frequency_mhz=5250.0, pulse_width_us=1.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        PulseRecord(toa_us=1000.0, frequency_mhz=5250.0, pulse_width_us=1.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        PulseRecord(toa_us=1500.0, frequency_mhz=5250.0, pulse_width_us=1.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
        PulseRecord(toa_us=2000.0, frequency_mhz=5250.0, pulse_width_us=1.0, amplitude_db=-50.0, aoa_deg=45.0, emitter_id=1),
    ]

    obs = np.zeros(360, dtype=np.float32)
    moe.eager_agent.get_q = lambda obs, hidden=None: (
        np.array([1.0 if (b == 10 and m == 1) else 0.0 for b in range(36) for m in range(5)], dtype=np.float32),
        hidden
    )

    frames = []
    for step in range(5):
        frame = controller.execute_operational_step(obs, scenario_pulses=pulses)
        frames.append(frame)

    assert len(frames) == 5
    assert controller.clock_us > 0.0
    assert frames[0].retune_latency_us == 15.0
    assert frames[0].dwell_duration_us == 500.0
    assert frames[0].dwell_start_us == 15.0
    assert frames[0].dwell_end_us == 515.0

    f_dict = frames[0].to_dict()
    assert "cognitive_explanation" in f_dict
    assert "decision_reason" in f_dict["cognitive_explanation"]
    assert "predicted_track_id" in f_dict["cognitive_explanation"]
    assert "spatial_confidence" in f_dict["cognitive_explanation"]


def test_level3_scheduling_authority_invariant():
    """Level 3: Verify controller acts strictly as an orchestrator with zero independent policy."""
    moe = SmartScanMoE(MockDRQN())
    moe.default_tau = 0.0
    controller = OperationalReceiverController(moe_scheduler=moe)
    controller.reset()

    obs = np.random.RandomState(42).randn(360).astype(np.float32)
    moe.eager_agent.get_q = lambda obs, hidden=None: (np.zeros(180, dtype=np.float32), hidden)

    # Calling select_action directly vs through execute_operational_step
    moe.reset()
    direct_action, _, direct_attr = moe.select_action(obs)

    controller.reset()
    frame = controller.execute_operational_step(obs)

    assert frame.selected_band == (direct_action // 5)
    assert frame.selected_mode == (direct_action % 5)
    assert frame.decision_reason == direct_attr["reason"]

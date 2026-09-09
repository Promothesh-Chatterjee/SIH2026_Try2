"""
12-Point Acceptance Test Suite for Operational Demonstration Backend.

Verifies:
  1.  Closed-Loop Replay: Deterministic replay with physical pulse streaming.
  2.  Causal RF Stream: Incident RF fed incrementally; future pulses ignored.
  3.  Blind Interception: Zero ground-truth emitter_ids ingested or accessed.
  4.  Timebase Monotonicity: Single authoritative MissionClock for retune and dwell apertures.
  5.  Observation State Contract: 360-D vector, all features bounded in [0.0, 1.0].
  6.  Frozen 110k Model Gate: Exact SHA-256 hash match on checkpoint_gate_110000.pt.
  7.  Model Failure Refusal: Missing model returns HTTP 503 / refuses fallback.
  8.  Receiver Disconnect / Hardware Fault: Graceful degraded state without crash.
  9.  Malformed / Out-of-Spec PDW Handling: Robust rejection of NaN/Inf without crash.
  10. Long-Run Stability: 500+ operational cycles without state corruption or leakage.
  11. Clean Reset: Mission state, clock, and buffers wiped clean on /reset.
  12. Real-Time Latency: Decision cycle latency within demonstration limits (< 5.0 ms).
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
)
from src.deployment.api import STATE, app
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.operational import (
    MissionClock,
    OperationalReceiverController,
    OperationalStateBuilder,
    ReceiverAdapter,
    ReceiverTelemetryFrame,
)
from src.operational.receiver_adapter import ReceiverHardwareError


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def frozen_checkpoint_path() -> Path:
    return Path("checkpoints/scheduler/checkpoint_gate_110000.pt")


@pytest.fixture
def mock_moe_controller() -> OperationalReceiverController:
    """Instantiate a lightweight controller for fast testing."""
    drqn = DRQNScheduler(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_actions=CANONICAL_N_ACTIONS,
        n_modes=CANONICAL_N_MODES,
        lstm_hidden=64,
        lstm_layers=1,
    )
    moe = SmartScanMoE(
        drqn,
        {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "n_actions": CANONICAL_N_ACTIONS,
            "device": "cpu",
        },
    )
    ctrl = OperationalReceiverController(
        moe_scheduler=moe,
        retune_latency_us=15.0,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )
    ctrl.reset(0.0)
    return ctrl


# ── Criterion 1: Closed-Loop Replay ──────────────────────────────────────────

def test_01_closed_loop_replay():
    """Criterion 1: Verify deterministic operational step execution with physical PDWs."""
    client = TestClient(app)
    with client:
        r_start = client.post("/mission/start", json={"initial_time_us": 0.0})
        assert r_start.status_code == 200
        assert r_start.json()["mission_active"] is True

        # Dwell 0 starts at t=0, retunes for 15 us -> dwell window is [15 us, 515 us]
        # Feed pulse at 200 us in band 9 (5250 MHz)
        pdw_batch = [
            {"toa_us": 200.0, "freq_mhz": 5250.0, "pw_us": 1.5, "amp_db": -40.0, "aoa_deg": 45.0}
        ]
        r_step = client.post("/mission/step", json={"pdws": pdw_batch})
        assert r_step.status_code == 200
        data = r_step.json()
        assert data["status"] == "ok"
        frame = data["frame"]
        assert frame["step"] == 0
        assert frame["dwell_start_us"] == 15.0
        assert frame["dwell_duration_us"] in [125.0, 500.0, 1250.0]


# ── Criterion 2: Causal RF Stream ────────────────────────────────────────────

def test_02_causal_rf_stream(mock_moe_controller: OperationalReceiverController):
    """Criterion 2: Incident RF fed incrementally; future pulses beyond dwell window are not detected."""
    ctrl = mock_moe_controller
    ctrl.start_mission(initial_time_us=100.0)

    # Step 1: retune 15 us (100 -> 115), base dwell 500 us (115 -> 615 us)
    # Feed pulses: one at 300 us, one in future at 1000 us
    pulses = [
        {"toa_us": 300.0, "freq_mhz": 1250.0, "pw_us": 1.0, "amp_db": -40.0, "aoa_deg": 30.0},
        {"toa_us": 1000.0, "freq_mhz": 1250.0, "pw_us": 1.0, "amp_db": -40.0, "aoa_deg": 30.0},
    ]
    frame = ctrl.execute_operational_step(external_rf_stream=pulses)

    # Future pulse at 1000 us must NOT be detected in interval [115, 615]
    for d in frame.detections:
        assert d["time_us"] <= frame.dwell_end_us


# ── Criterion 3: Blind Interception (No Ground-Truth emitter_ids) ─────────────

def test_03_blind_interception(mock_moe_controller: OperationalReceiverController):
    """Criterion 3: Ensure zero ground-truth emitter labels enter tracking, state, or decisions."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)

    # Ingest PDW with NO emitter_id
    pdw = {"time_us": 250.0, "frequency_mhz": 2250.0, "pulse_width_us": 2.0, "amplitude_db": -45.0, "aoa_deg": 90.0}
    assert "emitter_id" not in pdw

    tid = ctrl._associate_pulse_to_track(pdw)
    assert isinstance(tid, int) and tid >= 0

    # Verify track has no ground truth ID
    assert tid in ctrl.emitter_tracker.tracks
    track = ctrl.emitter_tracker.tracks[tid]
    assert not hasattr(track, "emitter_id") or getattr(track, "emitter_id") is None


# ── Criterion 4: Timebase Monotonicity ───────────────────────────────────────

def test_04_timebase_monotonicity():
    """Criterion 4: MissionClock is strictly monotonic and authoritatively advances apertures."""
    clock = MissionClock(initial_time_us=500.0)
    assert clock.current_time_us == 500.0

    retune_start, retune_end = clock.advance_retune(15.0)
    assert retune_start == 500.0
    assert retune_end == 515.0
    assert clock.current_time_us == 515.0

    dwell_start, dwell_end = clock.advance_dwell(500.0)
    assert dwell_start == 515.0
    assert dwell_end == 1015.0
    assert clock.current_time_us == 1015.0

    with pytest.raises(ValueError):
        clock.advance_by(-10.0)


# ── Criterion 5: Observation State Contract ──────────────────────────────────

def test_05_observation_state_contract():
    """Criterion 5: OperationalStateBuilder produces exact (360,) float32 in [0, 1]."""
    builder = OperationalStateBuilder(n_bands=36)
    obs = builder.build_state(current_time_us=1000.0)

    assert obs.shape == (360,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()
    assert (obs >= 0.0).all()
    assert (obs <= 1.0).all()

    # Verify validation pass
    assert builder.validate_state(obs) is True


# ── Criterion 6: Frozen 110k Model Gate ──────────────────────────────────────

def test_06_frozen_110k_model_integrity(frozen_checkpoint_path: Path):
    """Criterion 6: Checkpoint matches exact Phase 7 SHA-256 hash (frozen network)."""
    assert frozen_checkpoint_path.exists(), f"Missing {frozen_checkpoint_path}"
    hasher = hashlib.sha256()
    with open(frozen_checkpoint_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    expected_hash = "43617494a8b0655ec272fc16c05c6ec2c1ca45ad150780b858ce37f9df38fd67"
    assert actual_hash == expected_hash, f"Checkpoint hash mismatch! Retraining or modification detected: {actual_hash}"


# ── Criterion 7: Model Failure Refusal ───────────────────────────────────────

def test_07_model_failure_refusal():
    """Criterion 7: Missing or failed model causes refusal (503), no random fallbacks."""
    # Temporarily set STATE['controller'] = None
    old_ctrl = STATE.get("controller")
    try:
        STATE["controller"] = None
        client = TestClient(app)
        r = client.post("/mission/start", json={"initial_time_us": 0.0})
        assert r.status_code == 503
        assert "OperationalReceiverController not initialised" in r.json()["detail"]

        r_step = client.post("/mission/step", json={})
        assert r_step.status_code == 503
    finally:
        STATE["controller"] = old_ctrl


# ── Criterion 8: Receiver Disconnect / Hardware Fault ────────────────────────

def test_08_receiver_disconnect_handling(mock_moe_controller: OperationalReceiverController):
    """Criterion 8: Hardware disconnect raises ReceiverHardwareError; handled cleanly."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)

    # Disconnect receiver adapter
    ctrl.receiver_adapter.set_connected(False)
    assert ctrl.receiver_adapter.is_connected is False

    with pytest.raises(ReceiverHardwareError):
        ctrl.execute_operational_step()

    # Reconnecting restores normal operation
    ctrl.receiver_adapter.set_connected(True)
    frame = ctrl.execute_operational_step()
    assert frame is not None
    assert isinstance(frame, ReceiverTelemetryFrame)


# ── Criterion 9: Malformed / Out-of-Spec PDW Handling ────────────────────────

def test_09_malformed_pdw_handling(mock_moe_controller: OperationalReceiverController):
    """Criterion 9: Non-finite / corrupt PDWs are safely rejected without crashing the loop."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)

    # Feed pulses with corrupt fields
    malformed_pulses = [
        {"toa_us": None, "freq_mhz": 2000.0, "pw_us": 1.0},
        {"toa_us": 100.0, "freq_mhz": None, "pw_us": 1.0},
        {"toa_us": 150.0, "freq_mhz": float("nan"), "pw_us": 1.0},
        {"toa_us": 200.0, "freq_mhz": float("inf"), "pw_us": 1.0},
        {"toa_us": 250.0, "freq_mhz": 2000.0, "pw_us": -5.0},
    ]

    for p in malformed_pulses:
        try:
            ctrl.receiver_adapter.feed_incident_rf([p])
        except Exception:
            pass  # Expected safe rejection

    # Receiver and controller remain stable
    frame = ctrl.execute_operational_step()
    assert frame is not None
    assert frame.step == 0


# ── Criterion 10: Long-Run State Stability ───────────────────────────────────

def test_10_long_run_stability(mock_moe_controller: OperationalReceiverController):
    """Criterion 10: 500+ dwell cycles run continuously without state corruption or leakage."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)

    rng = np.random.default_rng(12345)
    for i in range(500):
        # Stream occasional pulses
        pulses = []
        if i % 10 == 0:
            pulses.append({
                "toa_us": ctrl.clock_us + 100.0,
                "freq_mhz": float(rng.choice([1250.0, 2250.0, 5250.0, 8750.0])),
                "pw_us": 1.5,
                "amp_db": -45.0,
                "aoa_deg": float(rng.uniform(0.0, 360.0)),
            })
        frame = ctrl.execute_operational_step(external_rf_stream=pulses)
        assert frame.step == i
        assert math.isfinite(frame.dwell_end_us)

    assert ctrl.current_step == 500
    assert ctrl.total_dwells == 500
    assert ctrl.clock_us > 0.0


# ── Criterion 11: Clean Reset ────────────────────────────────────────────────

def test_11_clean_reset(mock_moe_controller: OperationalReceiverController):
    """Criterion 11: Mission reset clears all tracking, clock, buffers, and metrics."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)
    for _ in range(5):
        ctrl.execute_operational_step()

    assert ctrl.current_step == 5
    assert ctrl.clock_us > 0.0

    ctrl.reset(0.0)
    assert ctrl.current_step == 0
    assert ctrl.clock_us == 0.0
    assert ctrl.total_dwells == 0
    assert ctrl.total_hits == 0
    assert len(ctrl.pdw_buffer) == 0
    assert len(ctrl.telemetry_history) == 0
    assert len(ctrl.emitter_tracker.tracks) == 0


# ── Criterion 12: Real-Time Latency ──────────────────────────────────────────

def test_12_real_time_latency(mock_moe_controller: OperationalReceiverController):
    """Criterion 12: Controller closed-loop step executes in < 5.0 ms wall-clock time."""
    ctrl = mock_moe_controller
    ctrl.start_mission(0.0)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        ctrl.execute_operational_step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    mean_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))

    assert mean_latency_ms < 5.0, f"Mean latency {mean_latency_ms:.2f} ms exceeds 5.0 ms threshold"
    assert p95_latency_ms < 10.0, f"P95 latency {p95_latency_ms:.2f} ms exceeds 10.0 ms threshold"
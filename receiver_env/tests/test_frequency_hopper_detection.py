"""Unit tests for Phase 4.2 Frequency Hopper Detection and PRI-Aware Association."""

from __future__ import annotations

from collections import deque
import pytest
import numpy as np

from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import (
    DeinterleaverConfig,
    EmitterTrack,
    TrackHistory,
    TrackStatus,
)
from receiver_env.deinterleaver.track_manager import TrackManager
from receiver_env.deinterleaver.track_associator import TrackAssociator
from receiver_env.deinterleaver.deinterleaver import PDWDeinterleaver


def make_pdw(pdw_id: int, toa_us: float, freq_mhz: float, pw_us: float = 10.0, conf: float = 0.95) -> PDW:
    return PDW(
        pdw_id=pdw_id,
        pulse_id=pdw_id,
        toa_us=toa_us,
        pulse_width_us=pw_us,
        frequency_mhz=freq_mhz,
        amplitude_db=-20.0,
        confidence=conf,
        receiver_id="RX_01",
        generation_timestamp_us=toa_us + 1.0,
    )


def test_track_manager_stable_emitter_diagnostics():
    """Verify that a stable fixed-frequency emitter is not flagged as hopping."""
    manager = TrackManager()
    seed = make_pdw(1, 0.0, 3000.0)
    track = manager.create_track(seed)

    assert track.latest_frequency_mhz == 3000.0
    assert track.frequency_span_mhz == 0.0
    assert not track.frequency_hopping_detected
    assert track.hop_rate_hz == 0.0

    for i in range(1, 10):
        pdw = make_pdw(i + 1, i * 100.0, 3000.0 + np.random.uniform(-0.1, 0.1))
        manager.update_track(track, pdw)

    assert track.pulse_count == 10
    assert track.frequency_span_mhz < 0.5
    assert not track.frequency_hopping_detected
    assert track.hop_rate_hz == 0.0


def test_track_manager_hopping_emitter_diagnostics():
    """Verify that a frequency-hopping emitter correctly updates span, hopping flag, and hop rate."""
    manager = TrackManager()
    seed = make_pdw(1, 0.0, 3400.0)
    track = manager.create_track(seed)

    hops = [3500.0, 3600.0, 3700.0, 3400.0]
    for i, f in enumerate(hops, start=1):
        pdw = make_pdw(i + 1, i * 100.0, f)
        manager.update_track(track, pdw)

    assert track.pulse_count == 5
    assert track.latest_frequency_mhz == 3400.0
    assert track.frequency_span_mhz == 300.0  # 3700 - 3400
    assert track.frequency_hopping_detected
    assert track.hop_rate_hz > 0.0


def test_track_associator_pri_locked_known_hop_channel():
    """Verify that an established PRI-locked track can associate known hop channels."""
    associator = TrackAssociator()

    track = EmitterTrack(
        track_id=1,
        emitter_id="TRACK_0001",
        pulse_count=10,
        mean_frequency_mhz=3550.0,
        mean_pw_us=10.0,
        estimated_pri_us=100.0,
        pri_confidence=0.90,
        first_toa_us=0.0,
        last_toa_us=900.0,
        track_confidence=0.95,
        status=TrackStatus.CONFIRMED,
        frequency_history=deque([3400.0, 3500.0, 3600.0, 3700.0], maxlen=128),
        latest_frequency_mhz=3700.0,
        frequency_span_mhz=300.0,
        frequency_hopping_detected=True,
    )

    # Next pulse at ToA = 1000.0 us (residual = 0.0 us) on known hop channel 3400.0 MHz
    pdw_hop = make_pdw(101, 1000.0, 3400.05, pw_us=10.0)
    best = associator.find_best_track(pdw_hop, [track])

    assert best is not None
    assert best.track_id == 1


def test_track_associator_unlocked_hopper_does_not_false_merge():
    """Verify that a track without PRI lock will NOT associate pulses outside the standard gate."""
    associator = TrackAssociator()

    # Tentative track with only 2 pulses, no established PRI
    track = EmitterTrack(
        track_id=1,
        emitter_id="TRACK_0001",
        pulse_count=2,
        mean_frequency_mhz=3400.0,
        mean_pw_us=10.0,
        estimated_pri_us=0.0,
        first_toa_us=0.0,
        last_toa_us=100.0,
        track_confidence=0.5,
        status=TrackStatus.TENTATIVE,
        frequency_history=deque([3400.0, 3400.0], maxlen=128),
    )

    # Pulse 100 MHz away
    pdw_jump = make_pdw(3, 200.0, 3500.0, pw_us=10.0)
    best = associator.find_best_track(pdw_jump, [track])

    assert best is None

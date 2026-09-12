"""Unit tests for Track Manager and Lifecycle."""

from __future__ import annotations

import pytest
from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig, TrackStatus
from receiver_env.deinterleaver.track_manager import TrackManager


def make_pdw(pdw_id: int, toa_us: float, freq_mhz: float = 3000.0, pw_us: float = 10.0) -> PDW:
    return PDW(
        pdw_id=pdw_id,
        pulse_id=pdw_id,
        toa_us=toa_us,
        pulse_width_us=pw_us,
        frequency_mhz=freq_mhz,
        amplitude_db=-20.0,
        confidence=0.95,
        receiver_id="RX_01",
        generation_timestamp_us=toa_us + 1.0,
    )


def test_track_manager_lifecycle():
    manager = TrackManager()

    # 1. Create track with seed
    p1 = make_pdw(1, 10.0)
    t1 = manager.create_track(p1)
    assert t1.track_id == 1
    assert t1.status == TrackStatus.TENTATIVE
    assert t1.pulse_count == 1
    assert len(manager.active_tracks) == 1

    # 2. Update track
    p2 = make_pdw(2, 110.0)
    manager.update_track(t1, p2)
    assert t1.pulse_count == 2
    assert t1.last_toa_us == 110.0

    # 3. Third pulse -> promotion to CONFIRMED with PRI estimation
    p3 = make_pdw(3, 210.0)
    manager.update_track(t1, p3)
    assert t1.pulse_count == 3
    assert t1.status == TrackStatus.CONFIRMED
    assert abs(t1.estimated_pri_us - 100.0) < 1.0
    assert len(t1.history.recent_toas) == 3


def test_track_manager_merge():
    manager = TrackManager(DeinterleaverConfig(merge_freq_gate_mhz=0.5, merge_pw_gate_pct=0.2))

    # Create two split tracks with near-identical parameters
    t1 = manager.create_track(make_pdw(1, 10.0, freq_mhz=3000.0, pw_us=10.0))
    manager.update_track(t1, make_pdw(2, 110.0, freq_mhz=3000.0, pw_us=10.0))
    manager.update_track(t1, make_pdw(3, 210.0, freq_mhz=3000.0, pw_us=10.0))

    t2 = manager.create_track(make_pdw(4, 310.0, freq_mhz=3000.05, pw_us=10.05))
    manager.update_track(t2, make_pdw(5, 410.0, freq_mhz=3000.05, pw_us=10.05))
    manager.update_track(t2, make_pdw(6, 510.0, freq_mhz=3000.05, pw_us=10.05))

    assert len(manager.active_tracks) == 2
    merged = manager.merge_tracks()
    assert merged == 1
    assert len(manager.active_tracks) == 1
    assert manager.active_tracks[0].pulse_count == 6


def test_track_manager_expiry():
    manager = TrackManager(DeinterleaverConfig(track_expiry_pri_mult=4.0))

    t1 = manager.create_track(make_pdw(1, 10.0))
    manager.update_track(t1, make_pdw(2, 110.0))
    manager.update_track(t1, make_pdw(3, 210.0))  # PRI = 100 us

    assert len(manager.active_tracks) == 1

    # Advance time far beyond timeout (e.g. 60,000 us)
    expired = manager.expire_tracks(65_000.0)
    assert len(expired) == 1
    assert len(manager.active_tracks) == 0
    assert expired[0].status == TrackStatus.TERMINATED

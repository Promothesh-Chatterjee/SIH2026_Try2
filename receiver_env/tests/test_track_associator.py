"""Unit tests for Hungarian Track Associator."""

from __future__ import annotations

import pytest
from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig, EmitterTrack, TrackHistory, TrackStatus
from receiver_env.deinterleaver.track_associator import TrackAssociator


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


def test_track_associator_hungarian_assignment():
    associator = TrackAssociator()

    track1 = EmitterTrack(
        track_id=1,
        emitter_id="TRACK_0001",
        pulse_count=10,
        mean_frequency_mhz=3000.0,
        mean_pw_us=10.0,
        estimated_pri_us=100.0,
        first_toa_us=10.0,
        last_toa_us=910.0,
        track_confidence=0.95,
        status=TrackStatus.CONFIRMED,
    )
    track2 = EmitterTrack(
        track_id=2,
        emitter_id="TRACK_0002",
        pulse_count=10,
        mean_frequency_mhz=3050.0,
        mean_pw_us=5.0,
        estimated_pri_us=70.0,
        first_toa_us=10.0,
        last_toa_us=640.0,
        track_confidence=0.95,
        status=TrackStatus.CONFIRMED,
    )

    # Next expected pulse for track1: ~1010.0 us, track2: ~710.0 us
    pdw_a = make_pdw(101, 1010.0, 3000.02, pw_us=10.0)
    pdw_b = make_pdw(102, 710.0, 3049.98, pw_us=5.0)

    # Batch with both pulses
    assigned, unassigned = associator.associate_batch([pdw_a, pdw_b], [track1, track2])

    assert len(assigned) == 2
    assert len(unassigned) == 0

    mapping = dict(assigned)
    assert mapping[0] == 0  # pdw_a -> track1
    assert mapping[1] == 1  # pdw_b -> track2


def test_track_associator_gate_rejection():
    associator = TrackAssociator()
    track = EmitterTrack(
        track_id=1,
        emitter_id="TRACK_0001",
        pulse_count=10,
        mean_frequency_mhz=3000.0,
        mean_pw_us=10.0,
        estimated_pri_us=100.0,
        first_toa_us=10.0,
        last_toa_us=910.0,
        track_confidence=0.95,
    )

    # Far frequency
    pdw_bad_f = make_pdw(1, 1010.0, 3010.0, pw_us=10.0)
    # Bad pulse width
    pdw_bad_pw = make_pdw(2, 1010.0, 3000.0, pw_us=25.0)

    assigned, unassigned = associator.associate_batch([pdw_bad_f, pdw_bad_pw], [track])
    assert len(assigned) == 0
    assert len(unassigned) == 2

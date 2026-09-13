"""Unit tests for PulseTracker state machine and boundary detection."""

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.pulse_tracker import PulseTracker


def test_pulse_tracker_single_pulse():
    """Verify clean tracking of a single isolated pulse."""
    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=10)
    tracker = PulseTracker(config)

    # 100 samples total, pulse from index 20 to 50 (duration 31 samples)
    envelope = np.full(100, 0.1, dtype=np.float32)
    envelope[20:51] = 0.8
    t_high = 0.5
    t_low = 0.3

    pulses = tracker.track_chunk(envelope, t_high, t_low, global_sample_offset=1000)

    assert len(pulses) == 1
    p = pulses[0]
    assert p.pulse_id == 1
    assert p.global_start_sample == 1020
    assert p.global_end_sample == 1050
    assert p.peak_magnitude == pytest.approx(0.8, abs=1e-4)
    assert 0.0 <= p.confidence <= 1.0


def test_pulse_tracker_straddling_chunk_boundary():
    """Verify pulse that crosses chunk boundary is not split or duplicated."""
    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=10)
    tracker = PulseTracker(config)

    t_high = 0.5
    t_low = 0.3

    # Chunk 0: 50 samples. Pulse starts at sample 35 and remains high to the end (35 to 49 = 15 samples)
    chunk0 = np.full(50, 0.1, dtype=np.float32)
    chunk0[35:] = 0.85
    pulses0 = tracker.track_chunk(chunk0, t_high, t_low, global_sample_offset=0)
    assert len(pulses0) == 0, "Pulse should still be in-progress across chunk boundary!"

    # Chunk 1: 50 samples. Pulse continues until sample 15, then drops to 0.1
    chunk1 = np.full(50, 0.1, dtype=np.float32)
    chunk1[:16] = 0.85
    pulses1 = tracker.track_chunk(chunk1, t_high, t_low, global_sample_offset=50)

    # Pulse should close once gap threshold is satisfied
    assert len(pulses1) == 1, "Pulse should complete in chunk 1"
    p = pulses1[0]
    assert p.global_start_sample == 35
    assert p.global_end_sample == 50 + 15  # 65
    assert p.peak_magnitude == pytest.approx(0.85, abs=1e-4)


def test_pulse_tracker_anti_glitch_rejection():
    """Verify pulses shorter than min_pulse_samples are discarded as glitches."""
    config = ReceiverConfig(min_pulse_samples=15, min_gap_samples=5)
    tracker = PulseTracker(config)

    # Short glitch pulse of 5 samples (less than 15)
    envelope = np.full(100, 0.1, dtype=np.float32)
    envelope[30:35] = 0.9

    pulses = tracker.track_chunk(envelope, t_high=0.5, t_low=0.3, global_sample_offset=0)
    assert len(pulses) == 0, f"Glitch pulse of 5 samples should be rejected! Got: {pulses}"


def test_pulse_tracker_anti_splitting_noise_dip():
    """Verify intra-pulse noise dips shorter than min_gap_samples do not split the pulse."""
    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=10)
    tracker = PulseTracker(config)

    # Pulse from 20 to 80 with a 3-sample dip at 45:48
    envelope = np.full(120, 0.1, dtype=np.float32)
    envelope[20:81] = 0.8
    envelope[45:48] = 0.15  # brief noise drop below t_low for 3 samples

    pulses = tracker.track_chunk(envelope, t_high=0.5, t_low=0.3, global_sample_offset=0)

    assert len(pulses) == 1, f"Pulse was erroneously split into {len(pulses)} parts!"
    p = pulses[0]
    assert p.global_start_sample == 20
    assert p.global_end_sample == 80


def test_pulse_tracker_anti_merging():
    """Verify two pulses separated by more than min_gap_samples are resolved as distinct pulses."""
    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=10)
    tracker = PulseTracker(config)

    # Pulse 1: 20 to 50 (31 samples)
    # Gap: 51 to 70 (20 samples > min_gap_samples)
    # Pulse 2: 71 to 100 (30 samples)
    envelope = np.full(140, 0.1, dtype=np.float32)
    envelope[20:51] = 0.8
    envelope[71:101] = 0.85

    pulses = tracker.track_chunk(envelope, t_high=0.5, t_low=0.3, global_sample_offset=0)

    assert len(pulses) == 2, f"Expected 2 distinct pulses, got {len(pulses)}"
    assert pulses[0].global_start_sample == 20
    assert pulses[0].global_end_sample == 50
    assert pulses[1].global_start_sample == 71
    assert pulses[1].global_end_sample == 100

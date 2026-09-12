"""Tests for Phase 4.2 End-to-End Hopper Tracking and Scheduler Perception State.

Validates:
- Scenario 12: Hopper Alone (3400 -> 3500 -> 3600 -> 3700 MHz).
- Scenario 13: Mixed Environment (Stable 3000 MHz, Hopper 3400-3700 MHz, Stable 7750 MHz).
- Scenario 14: Boundary Hopper (3490 <-> 3510 MHz crossing Band 6 / Band 7 boundary).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.contracts import CANONICAL_N_BANDS
from src.operational.state_builder import OperationalStateBuilder
from src.perception.emitter_tracker import EmitterTracker, EmitterTrack
from src.receiver.models import DetectionObservation


def test_scenario_12_hopper_alone():
    """Scenario 12: Hopper Alone (3400 -> 3500 -> 3600 -> 3700 MHz).
    
    Verifies that OperationalStateBuilder populates BOTH Band 6 and Band 7
    in the 360-D observation state, resolving the single-band collapse bug.
    """
    hops = [3400.0, 3500.0, 3600.0, 3700.0]
    pri_us = 100.0
    pw_us = 10.0
    n_pulses = 40

    track = EmitterTrack(
        track_id=1,
        cluster_label=0,
        last_seen_time=n_pulses * pri_us,
        last_band=7,
    )

    for i in range(n_pulses):
        f = hops[i % len(hops)]
        t = i * pri_us
        track.frequency_history.append(f)
        track.toa_history.append(t)
        track.aoa_history.append(30.0)
        track.pw_history.append(pw_us)
        track.amplitude_history.append(-45.0)

    track._update_current_state()
    track._update_agility()

    assert track.frequency_hopping_detected, "Hopper should be detected as agile"
    assert track.frequency_span_mhz == 300.0, f"Expected 300 MHz span, got {track.frequency_span_mhz}"
    assert track.latest_frequency_mhz == hops[(n_pulses - 1) % len(hops)]

    # Build observation vector
    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS, recent_hop_window_us=10_000.0)
    current_time_us = n_pulses * pri_us
    obs = state_builder.build_state(current_time_us=current_time_us, active_tracks={1: track})
    state_matrix = obs.reshape(CANONICAL_N_BANDS, 10)

    # Feature 5: Emitter count in band
    # Feature 8: Frequency dispersion / agility
    band_6_emitters = state_matrix[6, 5]
    band_7_emitters = state_matrix[7, 5]

    assert band_6_emitters > 0.0, f"Band 6 was omitted! Feature 5 is {band_6_emitters}"
    assert band_7_emitters > 0.0, f"Band 7 was omitted! Feature 5 is {band_7_emitters}"
    assert state_matrix[6, 8] > 0.0, "Band 6 should reflect agility dispersion"
    assert state_matrix[7, 8] > 0.0, "Band 7 should reflect agility dispersion"

    # Other bands should have zero emitters
    for b in range(CANONICAL_N_BANDS):
        if b not in (6, 7):
            assert state_matrix[b, 5] == 0.0, f"Band {b} unexpectedly has emitters"


def test_scenario_13_mixed_environment():
    """Scenario 13: Mixed Environment.
    
    Emitter 1: Stable 3000 MHz (Band 6).
    Emitter 2: Hopper 3400-3700 MHz (Bands 6 and 7).
    Emitter 3: Stable 7750 MHz (Band 15).
    
    Verifies that stable emitters remain strictly confined to their single band,
    while the hopping emitter activates only its visited bands.
    """
    pri_us = 100.0
    current_time_us = 2000.0

    # 1. Stable emitter at 3000 MHz
    track_stable_1 = EmitterTrack(
        track_id=1,
        cluster_label=1,
        last_seen_time=current_time_us,
        last_band=6,
    )
    for i in range(20):
        t = current_time_us - (20 - i) * pri_us
        track_stable_1.frequency_history.append(3000.0 + 0.05 * (i % 2))
        track_stable_1.toa_history.append(t)
        track_stable_1.aoa_history.append(15.0)
        track_stable_1.pw_history.append(10.0)
        track_stable_1.amplitude_history.append(-40.0)
    track_stable_1._update_current_state()
    track_stable_1._update_agility()

    # 2. Hopping emitter 3400-3700 MHz
    hops = [3400.0, 3500.0, 3600.0, 3700.0]
    track_hopper = EmitterTrack(
        track_id=2,
        cluster_label=2,
        last_seen_time=current_time_us,
        last_band=7,
    )
    for i in range(20):
        t = current_time_us - (20 - i) * pri_us
        track_hopper.frequency_history.append(hops[i % len(hops)])
        track_hopper.toa_history.append(t)
        track_hopper.aoa_history.append(45.0)
        track_hopper.pw_history.append(12.0)
        track_hopper.amplitude_history.append(-50.0)
    track_hopper._update_current_state()
    track_hopper._update_agility()

    # 3. Stable emitter at 7750 MHz (Band 15: 7500-8000 MHz)
    track_stable_3 = EmitterTrack(
        track_id=3,
        cluster_label=3,
        last_seen_time=current_time_us,
        last_band=15,
    )
    for i in range(20):
        t = current_time_us - (20 - i) * pri_us
        track_stable_3.frequency_history.append(7750.0 + 0.05 * (i % 2))
        track_stable_3.toa_history.append(t)
        track_stable_3.aoa_history.append(-20.0)
        track_stable_3.pw_history.append(5.0)
        track_stable_3.amplitude_history.append(-35.0)
    track_stable_3._update_current_state()
    track_stable_3._update_agility()

    assert not track_stable_1.frequency_hopping_detected
    assert track_hopper.frequency_hopping_detected
    assert not track_stable_3.frequency_hopping_detected

    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS, recent_hop_window_us=10_000.0)
    active_tracks = {1: track_stable_1, 2: track_hopper, 3: track_stable_3}
    obs = state_builder.build_state(current_time_us=current_time_us, active_tracks=active_tracks)
    state_matrix = obs.reshape(CANONICAL_N_BANDS, 10)

    # Band 6 should contain both track 1 and track 2 -> count >= 2 / 5.0 = 0.4
    assert state_matrix[6, 5] >= 0.4, f"Band 6 emitter count should be >= 0.4, got {state_matrix[6, 5]}"
    # Band 7 should contain track 2 -> count >= 1 / 5.0 = 0.2
    assert state_matrix[7, 5] >= 0.2, f"Band 7 emitter count should be >= 0.2, got {state_matrix[7, 5]}"
    # Band 15 should contain ONLY track 3 -> count == 1 / 5.0 = 0.2
    assert math.isclose(state_matrix[15, 5], 0.2, abs_tol=1e-3), f"Band 15 emitter count should be 0.2, got {state_matrix[15, 5]}"

    # No other band should have emitters
    for b in range(CANONICAL_N_BANDS):
        if b not in (6, 7, 15):
            assert state_matrix[b, 5] == 0.0, f"Band {b} should have 0 emitters, got {state_matrix[b, 5]}"


def test_scenario_14_boundary_hopper():
    """Scenario 14: Boundary Hopper (3490 <-> 3510 MHz crossing Band 6 / Band 7 boundary).
    
    Verifies that rapid crossing of the 3500 MHz band partition boundary
    correctly activates both Band 6 and Band 7 without edge truncation.
    """
    pri_us = 50.0
    current_time_us = 1000.0
    boundary_hops = [3490.0, 3510.0]

    track = EmitterTrack(
        track_id=1,
        cluster_label=0,
        last_seen_time=current_time_us,
        last_band=7,
    )
    for i in range(20):
        t = current_time_us - (20 - i) * pri_us
        track.frequency_history.append(boundary_hops[i % 2])
        track.toa_history.append(t)
        track.aoa_history.append(0.0)
        track.pw_history.append(8.0)
        track.amplitude_history.append(-42.0)

    track._update_current_state()
    track._update_agility()

    assert track.frequency_hopping_detected
    assert track.frequency_span_mhz == 20.0

    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS, recent_hop_window_us=10_000.0)
    obs = state_builder.build_state(current_time_us=current_time_us, active_tracks={1: track})
    state_matrix = obs.reshape(CANONICAL_N_BANDS, 10)

    assert state_matrix[6, 5] > 0.0, "Band 6 (3490 MHz) must be populated"
    assert state_matrix[7, 5] > 0.0, "Band 7 (3510 MHz) must be populated"
    assert state_matrix[6, 8] > 0.0, "Band 6 should have agility dispersion > 0"
    assert state_matrix[7, 8] > 0.0, "Band 7 should have agility dispersion > 0"

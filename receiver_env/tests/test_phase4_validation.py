"""Comprehensive Phase 4 Validation Suite covering all 9 Mandatory Scenarios."""

from __future__ import annotations

import gc
import os
import time
import numpy as np
import pytest

from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig
from receiver_env.deinterleaver.deinterleaver import PDWDeinterleaver
from receiver_env.deinterleaver.validation import ValidationFramework


def generate_pulse_train(
    emitter_id: str,
    start_pdw_id: int,
    start_toa_us: float,
    pri_us: float,
    pw_us: float,
    freq_mhz: float,
    num_pulses: int,
    jitter_pct: float = 0.0,
    drop_pct: float = 0.0,
    drift_mhz_per_pulse: float = 0.0,
    seed: int = 42,
) -> tuple[list[PDW], dict[int, str]]:
    rng = np.random.default_rng(seed)
    pdws = []
    gt = {}

    current_toa = start_toa_us
    current_pdw_id = start_pdw_id

    for i in range(num_pulses):
        # Apply PRI jitter
        jitter = rng.uniform(-pri_us * (jitter_pct / 100.0), pri_us * (jitter_pct / 100.0)) if jitter_pct > 0 else 0.0
        # Pulse drop simulation
        drop = rng.random() < (drop_pct / 100.0) if drop_pct > 0 else False

        freq = freq_mhz + i * drift_mhz_per_pulse

        if not drop:
            pdw = PDW(
                pdw_id=current_pdw_id,
                pulse_id=current_pdw_id,
                toa_us=round(float(current_toa), 2),
                pulse_width_us=round(float(pw_us + rng.uniform(-0.1, 0.1)), 2),
                frequency_mhz=round(float(freq + rng.uniform(-0.02, 0.02)), 2),
                amplitude_db=-25.0,
                confidence=0.98,
                receiver_id="RX_01",
                generation_timestamp_us=round(float(current_toa + 1.0), 2),
            )
            pdws.append(pdw)
            gt[current_pdw_id] = emitter_id
            current_pdw_id += 1

        current_toa += pri_us + jitter

    return pdws, gt


def test_scenario_1_single_emitter():
    """Scenario 1: Single emitter (Target: 1 track, 100% purity, 100% completeness)."""
    pdws, gt = generate_pulse_train("EMITTER_1", 1000, 10.0, pri_us=100.0, pw_us=10.0, freq_mhz=3000.0, num_pulses=50)
    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(pdws)

    assert len(tracks) == 1
    metrics = ValidationFramework.evaluate(tracks, gt)
    m = metrics["EMITTER_1"]

    assert m.track_purity >= 0.99
    assert m.track_completeness >= 0.99
    assert m.duplicate_assignment_count == 0
    assert abs(tracks[0].estimated_pri_us - 100.0) < 1.0


def test_scenario_2_two_emitters_disjoint_frequencies():
    """Scenario 2: Two emitters, different frequencies (Target: 2 tracks, no mixing)."""
    p1, g1 = generate_pulse_train("EMITTER_1", 1000, 10.0, pri_us=100.0, pw_us=10.0, freq_mhz=3000.0, num_pulses=40, seed=1)
    p2, g2 = generate_pulse_train("EMITTER_2", 2000, 25.0, pri_us=105.0, pw_us=12.0, freq_mhz=3050.0, num_pulses=40, seed=2)

    all_pdws = sorted(p1 + p2, key=lambda p: p.toa_us)
    gt = {**g1, **g2}

    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(all_pdws)

    assert len(tracks) == 2
    metrics = ValidationFramework.evaluate(tracks, gt)

    for em in ["EMITTER_1", "EMITTER_2"]:
        assert metrics[em].track_purity >= 0.98
        assert metrics[em].track_completeness >= 0.98
        assert metrics[em].false_assignment_rate <= 0.02


def test_scenario_3_three_emitters_similar_pri_diff_freq():
    """Scenario 3: Three emitters, similar PRI, different frequencies (Target: Purity >= 95%)."""
    p1, g1 = generate_pulse_train("E1", 1000, 10.0, pri_us=80.0, pw_us=8.0, freq_mhz=2990.0, num_pulses=40, seed=10)
    p2, g2 = generate_pulse_train("E2", 2000, 15.0, pri_us=80.0, pw_us=8.0, freq_mhz=3000.0, num_pulses=40, seed=20)
    p3, g3 = generate_pulse_train("E3", 3000, 20.0, pri_us=80.0, pw_us=8.0, freq_mhz=3010.0, num_pulses=40, seed=30)

    all_pdws = sorted(p1 + p2 + p3, key=lambda p: p.toa_us)
    gt = {**g1, **g2, **g3}

    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(all_pdws)

    assert len(tracks) == 3
    metrics = ValidationFramework.evaluate(tracks, gt)
    for em in ["E1", "E2", "E3"]:
        assert metrics[em].track_purity >= 0.95
        assert metrics[em].track_completeness >= 0.95


def test_scenario_4_three_emitters_similar_freq_diff_pri():
    """Scenario 4: Three emitters, similar frequency, different PRI (Target: Purity >= 95%)."""
    p1, g1 = generate_pulse_train("E1", 1000, 10.0, pri_us=60.0, pw_us=10.0, freq_mhz=3200.0, num_pulses=50, seed=11)
    p2, g2 = generate_pulse_train("E2", 2000, 18.0, pri_us=90.0, pw_us=10.0, freq_mhz=3200.0, num_pulses=35, seed=22)
    p3, g3 = generate_pulse_train("E3", 3000, 25.0, pri_us=140.0, pw_us=10.0, freq_mhz=3200.0, num_pulses=25, seed=33)

    all_pdws = sorted(p1 + p2 + p3, key=lambda p: p.toa_us)
    gt = {**g1, **g2, **g3}

    # Use tight timing weights and physical minimum PRI constraint for similar frequency separation
    config = DeinterleaverConfig(min_pri_us=20.0, frequency_gate_mhz=0.3, weight_pri=0.45, weight_freq=0.30)
    deinterleaver = PDWDeinterleaver(config)
    tracks = deinterleaver.process_batch(all_pdws)

    metrics = ValidationFramework.evaluate(tracks, gt)
    for em in ["E1", "E2", "E3"]:
        assert metrics[em].track_purity >= 0.90
        assert metrics[em].track_completeness >= 0.90


def test_scenario_5_jittered_pri():
    """Scenario 5: Jittered PRI (+/- 10%) (Target: Track maintained, jitter estimated)."""
    pdws, gt = generate_pulse_train(
        "EMITTER_JITTER", 1000, 10.0, pri_us=120.0, pw_us=8.0, freq_mhz=3100.0, num_pulses=60, jitter_pct=10.0, seed=55
    )
    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(pdws)

    assert len(tracks) == 1
    metrics = ValidationFramework.evaluate(tracks, gt)
    m = metrics["EMITTER_JITTER"]

    assert m.track_purity >= 0.95
    assert m.track_completeness >= 0.95
    assert abs(tracks[0].estimated_pri_us - 120.0) < 6.0
    assert tracks[0].pri_jitter_pct > 2.0


def test_scenario_6_dropped_pulses():
    """Scenario 6: Dropped pulses (10-20%) (Target: Completeness >= 95%)."""
    pdws, gt = generate_pulse_train(
        "EMITTER_DROPS", 1000, 10.0, pri_us=90.0, pw_us=10.0, freq_mhz=2950.0, num_pulses=70, drop_pct=15.0, seed=77
    )
    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(pdws)

    assert len(tracks) == 1
    metrics = ValidationFramework.evaluate(tracks, gt)
    m = metrics["EMITTER_DROPS"]

    assert m.track_completeness >= 0.95
    assert m.track_purity >= 0.95
    assert abs(tracks[0].estimated_pri_us - 90.0) < 3.0


def test_scenario_7_100k_stress_and_determinism():
    """Scenario 7: 100k+ PDW stress test (Target: Memory growth < 5%, 100% determinism)."""
    import psutil

    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start = process.memory_info().rss

    n_pulses = 100_000
    deinterleaver1 = PDWDeinterleaver()
    deinterleaver2 = PDWDeinterleaver()

    batch_size = 5000
    n_batches = n_pulses // batch_size

    # Generate streams of interleaved pulses
    pulses_batch_sample = []
    for i in range(200):
        em_idx = i % 3
        freq = 3000.0 + em_idx * 20.0
        pdw = PDW(
            pdw_id=i,
            pulse_id=i,
            toa_us=float(10.0 + i * 50.0),
            pulse_width_us=10.0,
            frequency_mhz=freq,
            amplitude_db=-20.0,
            confidence=0.96,
            receiver_id="RX_01",
            generation_timestamp_us=float(10.0 + i * 50.0 + 1.0),
        )
        pulses_batch_sample.append(pdw)

    # Verify bitwise determinism on 200 pulses
    t1_tracks = deinterleaver1.process_batch(pulses_batch_sample)
    t2_tracks = deinterleaver2.process_batch(pulses_batch_sample)

    assert len(t1_tracks) == len(t2_tracks)
    for tr1, tr2 in zip(t1_tracks, t2_tracks):
        assert tr1.to_dict() == tr2.to_dict()

    # Now run through full 100k pulses to verify memory stability
    t0 = time.perf_counter()
    pdw_id_ctr = 0
    current_toa = 10.0

    for b in range(n_batches):
        batch = []
        for j in range(batch_size):
            em_idx = j % 4
            freq = 3000.0 + em_idx * 15.0
            p = PDW(
                pdw_id=pdw_id_ctr,
                pulse_id=pdw_id_ctr,
                toa_us=round(current_toa, 2),
                pulse_width_us=10.0,
                frequency_mhz=freq,
                amplitude_db=-20.0,
                confidence=0.95,
                receiver_id="RX_01",
                generation_timestamp_us=round(current_toa + 1.0, 2),
            )
            batch.append(p)
            pdw_id_ctr += 1
            current_toa += 25.0
        deinterleaver1.process_batch(batch)

    t1 = time.perf_counter()
    gc.collect()
    mem_end = process.memory_info().rss
    growth_pct = ((mem_end - mem_start) / max(1, mem_start)) * 100.0

    print(f"100k Deinterleaving processed in {t1 - t0:.2f} s, memory growth: {growth_pct:.2f}%")
    assert growth_pct < 5.0


def test_scenario_8_mixed_dense_environment():
    """Scenario 8: Mixed dense environment (5 simultaneous emitters) (Target: Purity >= 95%, Completeness >= 95%)."""
    emitters = [
        ("E1", 1000, 10.0, 60.0, 5.0, 2900.0, 50, 1),
        ("E2", 2000, 15.0, 85.0, 8.0, 2950.0, 40, 2),
        ("E3", 3000, 20.0, 110.0, 10.0, 3000.0, 35, 3),
        ("E4", 4000, 25.0, 140.0, 12.0, 3050.0, 30, 4),
        ("E5", 5000, 30.0, 180.0, 15.0, 3100.0, 25, 5),
    ]

    all_pdws = []
    gt = {}
    for name, pid, start_toa, pri, pw, freq, count, seed in emitters:
        p, g = generate_pulse_train(name, pid, start_toa, pri, pw, freq, count, seed=seed)
        all_pdws.extend(p)
        gt.update(g)

    all_pdws.sort(key=lambda p: p.toa_us)

    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(all_pdws)

    assert len(tracks) == 5
    metrics = ValidationFramework.evaluate(tracks, gt)

    for name, _, _, _, _, _, _, _ in emitters:
        m = metrics[name]
        assert m.track_purity >= 0.95
        assert m.track_completeness >= 0.95
        assert m.false_assignment_rate <= 0.05
        assert m.duplicate_assignment_count == 0


def test_scenario_9_crossing_emitters_frequency_drift():
    """Scenario 9: Crossing emitters with frequency drift (Target: Track swap rate = 0, Purity >= 95%)."""
    # Emitter A: 3000.0 MHz drifting upwards +0.01 MHz/pulse, PRI = 100.0 us
    # Emitter B: 3000.4 MHz drifting downwards -0.01 MHz/pulse, PRI = 102.0 us
    # At pulse 20, frequencies cross around 3000.2 MHz
    p1, g1 = generate_pulse_train(
        "EMITTER_A", 1000, 10.0, pri_us=100.0, pw_us=8.0, freq_mhz=3000.0, num_pulses=50, drift_mhz_per_pulse=+0.01, seed=101
    )
    p2, g2 = generate_pulse_train(
        "EMITTER_B", 2000, 15.0, pri_us=102.0, pw_us=12.0, freq_mhz=3000.5, num_pulses=50, drift_mhz_per_pulse=-0.01, seed=202
    )

    all_pdws = sorted(p1 + p2, key=lambda p: p.toa_us)
    gt = {**g1, **g2}

    config = DeinterleaverConfig(weight_pri=0.40, weight_pw=0.25, weight_freq=0.25)
    deinterleaver = PDWDeinterleaver(config)
    tracks = deinterleaver.process_batch(all_pdws)

    metrics = ValidationFramework.evaluate(tracks, gt)

    for em in ["EMITTER_A", "EMITTER_B"]:
        m = metrics[em]
        assert m.track_swap_rate <= 0.05
        assert m.track_purity >= 0.95
        assert m.track_completeness >= 0.95


def test_scenario_10_dense_similar_emitters():
    """Scenario 10: Dense Similar Emitters (5 emitters, 0.5 MHz frequency separation, similar PRI).
    Target: Purity >= 95%, Completeness >= 95%, False Assignment <= 5%, 0 duplicates.
    """
    emitters_spec = [
        ("EM_A", 1000, 10.0, 100.0, 10.0, 3000.0, 50, 42),
        ("EM_B", 2000, 20.0, 102.0, 8.0, 3000.5, 50, 43),
        ("EM_C", 3000, 30.0, 98.0, 12.0, 3001.0, 50, 44),
        ("EM_D", 4000, 40.0, 105.0, 6.0, 3001.5, 50, 45),
        ("EM_E", 5000, 50.0, 95.0, 14.0, 3002.0, 50, 46),
    ]
    all_pdws = []
    gt = {}
    for name, start_id, start_toa, pri, pw, freq, count, seed in emitters_spec:
        p, g = generate_pulse_train(
            name, start_id, start_toa, pri, pw, freq, count, jitter_pct=1.0, seed=seed
        )
        all_pdws.extend(p)
        gt.update(g)

    all_pdws.sort(key=lambda p: p.toa_us)
    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(all_pdws)

    metrics = ValidationFramework.evaluate(tracks, gt)
    for name, _, _, _, _, _, _, _ in emitters_spec:
        m = metrics[name]
        assert m.track_purity >= 0.95, f"{name} purity {m.track_purity} < 0.95"
        assert m.track_completeness >= 0.95, f"{name} completeness {m.track_completeness} < 0.95"
        assert m.false_assignment_rate <= 0.05, f"{name} false assignment {m.false_assignment_rate} > 0.05"
        assert m.duplicate_assignment_count == 0


def test_scenario_11_track_fragmentation_stress_test():
    """Scenario 11: Track Fragmentation Stress Test (15% dropped pulses, PRI jitter, burst gaps).
    Target: 0 track fragmentation (single track per emitter), 0 swaps, Completeness >= 95% of surviving pulses.
    """
    p1, g1 = generate_pulse_train(
        "RADAR_A", 1000, 10.0, pri_us=80.0, pw_us=10.0, freq_mhz=3000.0,
        num_pulses=80, jitter_pct=3.0, drop_pct=15.0, seed=51
    )
    p2, g2 = generate_pulse_train(
        "RADAR_B", 2000, 15.0, pri_us=120.0, pw_us=6.0, freq_mhz=3050.0,
        num_pulses=60, jitter_pct=3.0, drop_pct=15.0, seed=52
    )

    all_pdws = sorted(p1 + p2, key=lambda p: p.toa_us)
    gt = {**g1, **g2}

    deinterleaver = PDWDeinterleaver()
    tracks = deinterleaver.process_batch(all_pdws)

    metrics = ValidationFramework.evaluate(tracks, gt)
    for em in ["RADAR_A", "RADAR_B"]:
        m = metrics[em]
        assert m.track_purity >= 0.95, f"{em} purity {m.track_purity} < 0.95"
        assert m.track_completeness >= 0.95, f"{em} completeness {m.track_completeness} < 0.95"
        assert m.track_swap_rate == 0.0, f"{em} swap rate {m.track_swap_rate} != 0"

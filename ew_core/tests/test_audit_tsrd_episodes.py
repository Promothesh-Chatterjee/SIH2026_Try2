"""Regression tests for scripts/audit_tsrd_episodes.py.

Verifies:
1. Monotonically ordered ToA
2. Reverse-ordered ToA (chronological sorting, non-negative duration/PRI, raw_toa_monotonic==False)
3. Partially shuffled ToA
4. Duplicate ToA (adjacent identical timestamps)
5. All-identical ToA (pos_pris empty, zero positive deltas, pri metrics 0.0)
6. Empty file (0 pulses)
7. Single-pulse file (1 pulse)
8. Invariant: audit duration and PRI values can never become negative.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.audit_tsrd_episodes import audit_file


def _create_test_h5(
    file_path: Path,
    toas: np.ndarray,
    frequencies: np.ndarray | None = None,
    emitters: np.ndarray | None = None,
) -> None:
    n = len(toas)
    data = np.zeros((n, 5), dtype=np.float64)
    data[:, 0] = toas
    if frequencies is not None:
        data[:, 1] = frequencies
    else:
        data[:, 1] = np.linspace(1000.0, 5000.0, n) if n > 0 else np.array([])
    data[:, 2] = 2.0  # pulse width
    data[:, 3] = 15.0  # aoa
    data[:, 4] = -60.0  # amp

    labels = emitters if emitters is not None else np.zeros(n, dtype=np.int64)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)


def test_monotonically_ordered_toa(tmp_path: Path):
    h5_path = tmp_path / "mono.h5"
    toas = np.array([10.0, 25.0, 50.0, 100.0, 200.0])
    _create_test_h5(h5_path, toas)

    rep = audit_file(h5_path, max_pulses=5)
    assert rep["raw_toa_monotonic"] is True
    assert rep["raw_negative_delta_count"] == 0
    assert rep["raw_duplicate_toa_count"] == 0
    assert rep["total_duration_us"] == 190.0
    assert rep["retained_duration_us"] == 190.0
    assert rep["pri_mean_us"] == pytest.approx(47.5, 0.01)
    assert rep["pri_p50_us"] == pytest.approx(37.5, 0.01)
    assert rep["pri_sample_count"] == 4


def test_reverse_ordered_toa(tmp_path: Path):
    h5_path = tmp_path / "reverse.h5"
    toas = np.array([200.0, 100.0, 50.0, 25.0, 10.0])
    _create_test_h5(h5_path, toas)

    rep = audit_file(h5_path, max_pulses=3)
    # Raw ordering was reversed: 4 negative deltas
    assert rep["raw_toa_monotonic"] is False
    assert rep["raw_negative_delta_count"] == 4
    # Total duration must be computed on sorted copy: 200.0 - 10.0 = 190.0 (NEVER negative)
    assert rep["total_duration_us"] == 190.0
    assert rep["total_duration_us"] >= 0.0
    # Retained pulses must be earliest 3 chronologically: [10, 25, 50] -> span = 40.0
    assert rep["retained_duration_us"] == 40.0
    assert rep["retained_duration_us"] >= 0.0
    assert rep["pri_mean_us"] > 0.0
    assert rep["pri_p50_us"] > 0.0
    assert rep["pri_sample_count"] == 2


def test_partially_shuffled_toa(tmp_path: Path):
    h5_path = tmp_path / "shuffled.h5"
    toas = np.array([10.0, 100.0, 25.0, 200.0, 50.0])
    _create_test_h5(h5_path, toas)

    rep = audit_file(h5_path, max_pulses=5)
    assert rep["raw_toa_monotonic"] is False
    assert rep["raw_negative_delta_count"] == 2  # 100->25 and 200->50
    assert rep["total_duration_us"] == 190.0
    assert rep["retained_duration_us"] == 190.0
    assert rep["pri_mean_us"] >= 0.0
    assert rep["pri_p50_us"] >= 0.0


def test_duplicate_toa(tmp_path: Path):
    h5_path = tmp_path / "duplicates.h5"
    toas = np.array([10.0, 10.0, 25.0, 25.0, 50.0])
    _create_test_h5(h5_path, toas)

    rep = audit_file(h5_path, max_pulses=5)
    assert rep["raw_toa_monotonic"] is True
    assert rep["raw_negative_delta_count"] == 0
    assert rep["raw_duplicate_toa_count"] == 2
    assert rep["total_duration_us"] == 40.0
    # Positive PRIs only: (25-10)=15, (50-25)=25 -> positive deltas = [15, 25]
    assert rep["pri_sample_count"] == 2
    assert rep["pri_mean_us"] == 20.0
    assert rep["pri_p50_us"] == 20.0


def test_all_identical_toa(tmp_path: Path):
    h5_path = tmp_path / "identical.h5"
    toas = np.array([100.0, 100.0, 100.0, 100.0])
    _create_test_h5(h5_path, toas)

    rep = audit_file(h5_path, max_pulses=4)
    assert rep["raw_toa_monotonic"] is True
    assert rep["raw_negative_delta_count"] == 0
    assert rep["raw_duplicate_toa_count"] == 3
    assert rep["total_duration_us"] == 0.0
    assert rep["retained_duration_us"] == 0.0
    # No positive deltas: must emit 0.0 and sample count 0, without crashing
    assert rep["pri_sample_count"] == 0
    assert rep["pri_mean_us"] == 0.0
    assert rep["pri_p50_us"] == 0.0
    assert rep["pri_p90_us"] == 0.0


def test_empty_file(tmp_path: Path):
    h5_path = tmp_path / "empty.h5"
    _create_test_h5(h5_path, np.array([]))

    rep = audit_file(h5_path, max_pulses=50000)
    assert rep["total_pulses"] == 0
    assert rep["retained_pulses"] == 0
    assert rep["total_duration_us"] == 0.0
    assert rep["retained_duration_us"] == 0.0
    assert rep["pri_sample_count"] == 0
    assert rep["pri_mean_us"] == 0.0
    assert rep["raw_toa_monotonic"] is True


def test_single_pulse_file(tmp_path: Path):
    h5_path = tmp_path / "single.h5"
    _create_test_h5(h5_path, np.array([42.0]))

    rep = audit_file(h5_path, max_pulses=50000)
    assert rep["total_pulses"] == 1
    assert rep["retained_pulses"] == 1
    assert rep["total_duration_us"] == 0.0
    assert rep["retained_duration_us"] == 0.0
    assert rep["pri_sample_count"] == 0
    assert rep["pri_mean_us"] == 0.0
    assert rep["raw_toa_monotonic"] is True


def test_all_invariants_non_negative(tmp_path: Path):
    """Stress test with various irregular sequences proving metrics cannot become negative."""
    patterns = [
        np.array([500.0, 400.0, 300.0, 200.0, 100.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([0.0, 1000.0, 500.0, 2000.0, 1500.0]),
        np.array([-50.0, -100.0, 0.0, 50.0]),
    ]
    for idx, toas in enumerate(patterns):
        p = tmp_path / f"pattern_{idx}.h5"
        _create_test_h5(p, toas)
        rep = audit_file(p, max_pulses=3)
        assert rep["total_duration_us"] >= 0.0
        assert rep["retained_duration_us"] >= 0.0
        assert rep["pri_mean_us"] >= 0.0
        assert rep["pri_p50_us"] >= 0.0
        assert rep["pri_p90_us"] >= 0.0


def test_interleaved_multi_emitter_global_chronological_deltas(tmp_path: Path):
    """Verify that in interleaved multi-emitter scenarios, audit measures positive global chronological deltas."""
    h5_path = tmp_path / "interleaved.h5"
    # Interleaved pulse train from 2 emitters
    toas = np.array([10.0, 15.0, 30.0, 35.0])
    emitters = np.array([0, 1, 0, 1], dtype=np.int64)
    _create_test_h5(h5_path, toas=toas, emitters=emitters)

    rep = audit_file(h5_path, max_pulses=4)
    # Global chronological deltas: 15-10=5, 30-15=15, 35-30=5 -> mean = 25/3 = 8.33 us
    assert rep["pri_sample_count"] == 3
    assert rep["pri_mean_us"] == pytest.approx(25.0 / 3.0, 0.01)
    assert rep["pri_p50_us"] == pytest.approx(5.0, 0.01)
    assert rep["all_emitters_count"] == 2
    assert rep["retained_emitters_count"] == 2


"""Regression tests for canonical TSRD loader chronological invariant.

Verifies:
1. load_h5_records sorts returned PulseRecords so records[i].toa_us <= records[i+1].toa_us.
2. Shuffled raw H5 data produces strictly non-decreasing PulseRecord sequence starting at t=0.
3. Duration computed as records[-1].toa_us - records[0].toa_us is non-negative and accurate.
4. Chunking dependency: demonstrates why raw monotonicity (enforced by preflight Check 24)
   is necessary for max_pulses slicing to correspond to a genuine temporal chunk.
"""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pytest

from ew_core.environment.scenario_generator import load_h5_records


def _create_h5_scenario(file_path: Path, toas: np.ndarray, cfs: np.ndarray | None = None) -> None:
    n = len(toas)
    data = np.zeros((n, 5), dtype=np.float64)
    data[:, 0] = toas
    data[:, 1] = cfs if cfs is not None else np.full(n, 3000.0)
    data[:, 2] = 2.0  # pw
    data[:, 3] = 10.0  # aoa
    data[:, 4] = -60.0  # amp
    labels = np.zeros(n, dtype=np.int64)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)


def test_shuffled_raw_h5_returns_chronological_records(tmp_path: Path):
    """Raw H5 with out-of-order timestamps returns sorted PulseRecords."""
    h5_path = tmp_path / "shuffled_scen.h5"
    raw_toas = np.array([500.0, 100.0, 800.0, 50.0, 1000.0, 200.0])
    _create_h5_scenario(h5_path, raw_toas)

    records = load_h5_records(h5_path, max_pulses=50000)
    assert len(records) == 6

    # Verify normalization t0=0
    assert records[0].toa_us == 0.0

    # Verify strict non-decreasing ordering invariant
    for i in range(len(records) - 1):
        assert records[i].toa_us <= records[i + 1].toa_us

    # Verify exact relative spacings preserved: min raw was 50.0
    expected_sorted_relative = sorted(raw_toas - 50.0)
    actual_toas = [r.toa_us for r in records]
    np.testing.assert_allclose(actual_toas, expected_sorted_relative)

    # Duration calculation
    duration_us = records[-1].toa_us - records[0].toa_us
    assert duration_us == 950.0
    assert duration_us >= 0.0


def test_reverse_ordered_raw_h5_returns_chronological_records(tmp_path: Path):
    h5_path = tmp_path / "reverse_scen.h5"
    raw_toas = np.linspace(10000.0, 1000.0, 10)
    _create_h5_scenario(h5_path, raw_toas)

    records = load_h5_records(h5_path, max_pulses=50000)
    assert len(records) == 10
    assert records[0].toa_us == 0.0
    assert all(records[i].toa_us <= records[i + 1].toa_us for i in range(len(records) - 1))
    assert (records[-1].toa_us - records[0].toa_us) == pytest.approx(9000.0)


def test_chunking_requires_raw_monotonicity(tmp_path: Path):
    """Demonstrate why preflight Check 24 (raw non-decreasing ToA) is necessary.

    When max_pulses < total, chunk_mode='first' slices data[:max_pulses].
    If raw data were sorted, this slice is the earliest temporal window.
    If raw data were non-monotonic, the slice is arbitrary rows.
    """
    h5_monotonic = tmp_path / "mono.h5"
    h5_shuffled = tmp_path / "shuf.h5"

    mono_toas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    shuf_toas = np.array([60.0, 10.0, 50.0, 20.0, 40.0, 30.0])

    _create_h5_scenario(h5_monotonic, mono_toas)
    _create_h5_scenario(h5_shuffled, shuf_toas)

    # Monotonic chunk [:3] captures the true earliest window [10, 20, 30]
    rec_mono = load_h5_records(h5_monotonic, max_pulses=3, chunk_mode="first")
    assert len(rec_mono) == 3
    # Shifted to t0=0: [0, 10, 20]
    assert [r.toa_us for r in rec_mono] == [0.0, 10.0, 20.0]

    # Shuffled chunk [:3] takes rows [60, 10, 50], which then sorts to [10, 50, 60]
    # (missing the actual earliest pulses 20, 30, 40).
    # This proves why Check 24's FAIL CLOSED policy on raw non-monotonic data is essential.
    rec_shuf = load_h5_records(h5_shuffled, max_pulses=3, chunk_mode="first")
    assert len(rec_shuf) == 3
    # Both return internally sorted records (loader invariant holds)
    assert all(rec_shuf[i].toa_us <= rec_shuf[i + 1].toa_us for i in range(len(rec_shuf) - 1))

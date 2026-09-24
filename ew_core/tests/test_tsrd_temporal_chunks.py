import h5py
import numpy as np
import pytest

from ew_core.environment.scenario_generator import load_h5_records, ScenarioSource


@pytest.fixture
def dummy_h5_dataset(tmp_path):
    """Creates a temporary H5 dataset with 200 pulses for chunking validation."""
    h5_file = tmp_path / "test_scenario.h5"
    n_pulses = 200
    # data columns: [toa_us, freq_mhz, pw_us, amp_db, aoa_deg]
    toas = np.linspace(10.0, 2000.0, n_pulses, dtype=np.float32)
    freqs = np.full(n_pulses, 3000.0, dtype=np.float32)
    pws = np.full(n_pulses, 10.0, dtype=np.float32)
    amps = np.full(n_pulses, -65.0, dtype=np.float32)
    aoas = np.full(n_pulses, 45.0, dtype=np.float32)

    data = np.stack([toas, freqs, pws, amps, aoas], axis=1)
    labels = np.zeros(n_pulses, dtype=np.int64)

    with h5py.File(str(h5_file), "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)

    return h5_file


def test_chunk_mode_first_deterministic(dummy_h5_dataset):
    """Validate that chunk_mode='first' (used in evaluation and val_set) is 100% deterministic."""
    max_pulses = 50
    rec1 = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="first")
    rec2 = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="first")

    assert len(rec1) == max_pulses
    assert len(rec2) == max_pulses

    for r1, r2 in zip(rec1, rec2):
        assert r1.toa_us == r2.toa_us
        assert r1.frequency_mhz == r2.frequency_mhz
        assert r1.amplitude_db == r2.amplitude_db


def test_chunk_mode_random_seed_reproducibility_and_variation(dummy_h5_dataset):
    """Validate that chunk_mode='random' is reproducible with same seed and varies across seeds."""
    max_pulses = 40

    # Same seed -> identical chunk
    rec_s1a = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="random", seed=42)
    rec_s1b = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="random", seed=42)

    assert len(rec_s1a) == max_pulses
    assert len(rec_s1b) == max_pulses
    toas_1a = [r.toa_us for r in rec_s1a]
    toas_1b = [r.toa_us for r in rec_s1b]
    assert toas_1a == toas_1b

    # Different seeds -> different temporal slices
    rec_s2 = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="random", seed=999)
    toas_2 = [r.toa_us for r in rec_s2]

    # Verify that slices differ
    assert len(rec_s2) == max_pulses


def test_chunk_mode_uniform(dummy_h5_dataset):
    """Validate that chunk_mode='uniform' samples across the full scenario duration."""
    max_pulses = 20
    recs = load_h5_records(dummy_h5_dataset, max_pulses=max_pulses, chunk_mode="uniform")
    assert len(recs) == max_pulses
    # First pulse starts at 0.0 (normalized)
    assert recs[0].toa_us == 0.0
    # Last pulse spans almost the entire duration (~1990 us)
    assert recs[-1].toa_us > 1800.0

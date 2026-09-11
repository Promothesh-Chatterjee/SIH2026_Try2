import numpy as np
from cognitive_ew_smart_scan.src.training.train_scheduler import BandDiscoveryTracker

def test_band_discovery_tracker_basic():
    n_bands = 36
    quota = 2
    tracker = BandDiscoveryTracker(n_bands=n_bands, discovery_quota=quota)
    diag = tracker.get_diagnostics()
    assert diag['unique_bands'] == 0
    assert diag['min_band_visits'] == 0
    assert diag['underexplored_band_count'] == 36
    assert diag['underexplored_band_fraction'] == 1.0

    np.random.seed(42)
    b = tracker.sample_targeted_band()
    assert b is not None
    assert 0 <= b < n_bands

    for band in range(n_bands):
        if band != 7:
            for _ in range(quota):
                tracker.record_visit(band)

    for _ in range(10):
        assert tracker.sample_targeted_band() == 7

    tracker.record_visit(7)
    tracker.record_visit(7)

    assert tracker.sample_targeted_band() is None

    diag = tracker.get_diagnostics()
    assert diag['unique_bands'] == 36
    assert diag['min_band_visits'] >= 2
    assert diag['underexplored_band_count'] == 0
    assert diag['underexplored_band_fraction'] == 0.0

    tracker.reset()
    assert tracker.get_diagnostics()['unique_bands'] == 0
    assert tracker.sample_targeted_band() is not None

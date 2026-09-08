from src.cognitive.spatial_tracker import SpatialTracker

def test_circular_wrap_around():
    # Test wrap-around across 0 / 360 boundary
    angles = [358.0, 359.0, 1.0, 2.0]
    mean_deg, r = SpatialTracker.compute_circular_mean_and_r(angles)
    assert abs(mean_deg - 0.0) < 1e-4 or abs(mean_deg - 360.0) < 1e-4, f'Expected 0.0 deg, got {mean_deg}'
    assert r > 0.99, f'Expected high confidence R near 1.0, got {r}'

def test_circular_dispersion():
    # Test completely dispersed angles (opposite directions)
    angles = [0.0, 180.0]
    mean_deg, r = SpatialTracker.compute_circular_mean_and_r(angles)
    assert r < 1e-6, f'Expected zero confidence R for opposite angles, got {r}'

def test_spatial_tracker_update_and_priority():
    tracker = SpatialTracker(n_sectors=12)
    # Track 101 receives pulses near 359 deg
    tracker.update_from_track(101, 358.0, 1000.0)
    tracker.update_from_track(101, 1.0, 2000.0)
    sb = tracker.beliefs[101]
    assert abs(sb.mean_aoa_deg - 359.5) < 1.0 or abs(sb.mean_aoa_deg - 360.0) < 1.0 or abs(sb.mean_aoa_deg - 0.0) < 1.0
    assert sb.confidence > 0.95
    prio = tracker.get_spatial_priority(101, 3000.0)
    assert 0.0 <= prio <= 1.0
    assert prio > 0.90

import numpy as np
import pytest


def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Computes exact linear p95 and median latency stats consistent with smoke_test.py."""
    if not latencies_ms:
        raise ValueError("Cannot compute latency stats on empty list")
    lat_arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "count": len(lat_arr),
        "min": float(np.min(lat_arr)),
        "median": float(np.median(lat_arr)),
        "p95": float(np.percentile(lat_arr, 95, method="linear")),
        "max": float(np.max(lat_arr)),
    }


def test_20_sample_latency_linear_p95():
    """Verify linear p95 computation on 20 samples matches exact mathematical definition.

    For 20 sorted samples x_0, ..., x_19:
    percentile rank = 0.95 * (20 - 1) = 18.05
    linear interp = x_18 + 0.05 * (x_19 - x_18)
    """
    # 20 samples from 10.0 to 200.0 linearly spaced
    samples = np.linspace(10.0, 200.0, 20).tolist()
    stats = compute_latency_stats(samples)

    assert stats["count"] == 20
    assert stats["min"] == 10.0
    assert stats["max"] == 200.0

    # Manual verification of linear interpolation at index 18.05
    x18 = samples[18]
    x19 = samples[19]
    expected_p95 = x18 + 0.05 * (x19 - x18)

    assert abs(stats["p95"] - expected_p95) < 1e-9
    assert stats["median"] == float(np.median(samples))
    assert stats["p95"] < 500.0  # Pass SLA


def test_latency_sla_breach_detection():
    """Verify that p95 >= 500ms or median >= 500ms correctly fails the SLA."""
    # 19 samples well under SLA, but tail sample causes p95 to exceed 500ms
    breach_samples = [100.0] * 18 + [550.0, 600.0]
    stats = compute_latency_stats(breach_samples)
    assert stats["p95"] > 500.0

    # High median breach
    median_breach_samples = [510.0] * 20
    stats_med = compute_latency_stats(median_breach_samples)
    assert stats_med["median"] > 500.0


def test_dual_latency_telemetry_consistency():
    """Verify client round-trip latency strictly bounds server inference latency."""
    client_latencies = [15.0, 20.0, 25.0, 18.0, 22.0] * 4  # 20 samples
    server_latencies = [5.0, 8.0, 10.0, 6.0, 9.0] * 4      # 20 samples

    client_stats = compute_latency_stats(client_latencies)
    server_stats = compute_latency_stats(server_latencies)

    # In a causally sound system, server inference time must be strictly less than client round-trip
    assert server_stats["median"] < client_stats["median"]
    assert server_stats["p95"] < client_stats["p95"]
    assert np.all(np.array(server_latencies) < np.array(client_latencies))

"""EW Metrics package exposing Figures of Merit calculation functions."""

from ew_core.metrics.ew_metrics import (
    EWMetrics,
    compute_all_metrics,
    compute_avg_intercept_rate,
    compute_avg_intercept_time_error,
    compute_avg_reward,
    compute_pd,
    compute_pfa,
    compute_pct_correct_predictions,
)

__all__ = [
    "EWMetrics",
    "compute_all_metrics",
    "compute_avg_intercept_rate",
    "compute_avg_intercept_time_error",
    "compute_avg_reward",
    "compute_pd",
    "compute_pfa",
    "compute_pct_correct_predictions",
]

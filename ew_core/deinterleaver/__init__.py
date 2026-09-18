"""Signal Deinterleaver package for EW."""

from .windowed_deinterleaver import (
    PulseDescriptorWord,
    PDWFeatureExtractor,
    CrossWindowReconciler,
    WindowedDeinterleaver,
    DeinterleaverResult,
    compute_purity,
    make_synthetic_pdws,
    make_tight_synthetic_embeddings,
    make_ground_truth_labels,
    run_with_sklearn_fallback,
)

__all__ = [
    "PulseDescriptorWord",
    "PDWFeatureExtractor",
    "CrossWindowReconciler",
    "WindowedDeinterleaver",
    "DeinterleaverResult",
    "compute_purity",
    "make_synthetic_pdws",
    "make_tight_synthetic_embeddings",
    "make_ground_truth_labels",
    "run_with_sklearn_fallback",
]

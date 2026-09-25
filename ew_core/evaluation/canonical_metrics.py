"""
Phase 8 Canonical Telemetry & Metric Integrity.

Establishes a single authoritative mathematical implementation for every
evaluation metric across all evaluation entry points:
- Interception Rate (IR) = total_hits / total_steps (dwells evaluated)
- Decision-level Probability of Detection (Pd) = TP / (TP + FN)
- Probability of False Alarm (Pfa) = FP / (FP + TN)
- Average Interception Latency / Time Error = mean(|t_hit - t_target|) across hits
- Unique Emitter Discovery Rate = len(discovered_emitters) / max(1, len(all_active_emitters))
- Action Shannon Entropy = -sum(p * log2(p))
- Top-Band / Top-Action Fraction = max(counts) / sum(counts)
- Evaluation Manifest logging: exact scenario, seed, pulses, and action counts.

Ensures zero divergence across staged_gate_evaluator, agile_sparse_battery,
and standalone baseline benchmarks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationManifest:
    """Rigorous metadata manifest ensuring defensible SIH benchmark provenance."""
    scenario_id: str
    seed: int
    episode_steps: int
    checkpoint_file: str | None = None
    git_revision: str | None = None
    total_pulses: int = 0
    active_emitters: int = 0
    discovered_emitters: int = 0
    total_interceptions: int = 0
    total_opportunities: int = 0
    false_alarms: int = 0
    latency_samples_count: int = 0
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalMetrics:
    """Canonical representation of evaluated policy metrics with explicit numerators/denominators."""
    # 1. Primary Interception Metrics
    interception_rate: float
    hits: int
    steps: int
    # 2. Decision-level Signal Quality
    pd: float
    pfa: float
    tp: int
    fn: int
    fp: int
    tn: int
    active_opportunities: int
    # 3. Latency / Timing Fidelity
    avg_intercept_time_error_us: float
    first_detection_time_us: float | None
    # 4. Spectrum & Action Exploration
    distinct_bands: int
    distinct_actions: int
    top_band_fraction: float
    top_action_fraction: float
    band_entropy: float
    mode_entropy: float
    action_entropy: float
    # 5. Emitter Tracking & Operational Coverage
    discovery_rate: float
    discovered_emitter_count: int
    total_emitter_count: int
    missed_active_emitter_count: int
    operational_coverage: float
    # 6. Reward Accumulation
    avg_reward: float
    total_reward: float
    # 7. Manifest Sidecar
    manifest: EvaluationManifest | None = None
    # 8. Phase 6 Dwell-Mode Distribution Diagnostics (Task 6.1)
    short_fraction: float = 0.0
    normal_fraction: float = 0.0
    long_fraction: float = 0.0
    revisit_fraction: float = 0.0
    preemptive_fraction: float = 0.0
    # 9. Phase 6 Dwell- vs Physical Time-Normalized Throughput & Latency (Task 6.2)
    ir_per_dwell: float = 0.0
    ir_per_ms: float = 0.0
    reward_per_dwell: float = 0.0
    reward_per_ms: float = 0.0
    first_hit_latency_us: float | None = None
    mission_time_to_first_intercept_ms: float | None = None
    total_mission_time_ms: float = 0.0
    # 10. Phase 2 Independent Timing Metrics
    operational_intercept_latency_us: float = 0.0
    predictive_time_error_us: float | None = None
    prediction_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.manifest:
            d["manifest"] = self.manifest.to_dict()
        return d


def shannon_entropy(counts: Sequence[float | int] | np.ndarray) -> float:
    """Compute base-2 Shannon entropy over a non-negative count distribution."""
    arr = np.asarray(counts, dtype=np.float64)
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    p = arr / total
    p_nonzero = p[p > 0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def compute_canonical_metrics(
    *,
    steps_done: int,
    ep_hits: int,
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    band_counts: np.ndarray | Sequence[int],
    action_counts: np.ndarray | Sequence[int],
    mode_counts: np.ndarray | Sequence[int],
    time_errors_us: list[float] | None = None,
    operational_intercept_latencies_us: list[float] | None = None,
    predictive_time_errors_us: list[float] | None = None,
    prediction_coverage: float | None = None,
    first_detection_time_us: float | None = None,
    discovered_emitters: set[int] | list[int] | None = None,
    all_active_emitters: set[int] | list[int] | None = None,
    selected_active_opportunities: int = 0,
    spectrum_active_opportunities: int = 0,
    total_reward: float = 0.0,
    manifest: EvaluationManifest | None = None,
    total_mission_time_us: float | None = None,
    first_hit_latency_us: float | None = None,
    mission_time_to_first_intercept_ms: float | None = None,
) -> CanonicalMetrics:
    """Authoritative canonical calculation of all EW evaluation figures of merit.

    Mathematical Invariants:
      * Interception Rate (IR) = ep_hits / steps_done
      * Pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      * Pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
      * Top Band Fraction = max(band_counts) / sum(band_counts)
      * Top Action Fraction = max(action_counts) / sum(action_counts)
      * Operational Coverage = selected_active / spectrum_active
    """
    steps = max(1, int(steps_done))
    hits = int(ep_hits)
    ir = float(hits / steps)

    # Signal quality (Pd / Pfa)
    denom_pd = int(tp + fn)
    pd_val = float(tp / denom_pd) if denom_pd > 0 else 0.0
    denom_pfa = int(fp + tn)
    pfa_val = float(fp / denom_pfa) if denom_pfa > 0 else 0.0

    # Latency / Timing Error
    valid_latencies = [x for x in (time_errors_us or []) if x == x and np.isfinite(x)]
    avg_latency = float(np.mean(valid_latencies)) if valid_latencies else 0.0

    valid_ops = [x for x in (operational_intercept_latencies_us or []) if x == x and np.isfinite(x)]
    op_latency = float(np.mean(valid_ops)) if valid_ops else avg_latency

    valid_preds = [x for x in (predictive_time_errors_us or []) if x == x and np.isfinite(x)]
    pred_error = float(np.mean(valid_preds)) if valid_preds else None
    pred_cov = float(prediction_coverage) if prediction_coverage is not None else (float(len(valid_preds) / max(1, hits)) if hits > 0 else 0.0)

    # Counts and distributions
    b_counts = np.asarray(band_counts, dtype=np.int64)
    a_counts = np.asarray(action_counts, dtype=np.int64)
    m_counts = np.asarray(mode_counts, dtype=np.int64)

    tot_b = max(1, int(np.sum(b_counts)))
    tot_a = max(1, int(np.sum(a_counts)))

    top_band_frac = float(np.max(b_counts) / tot_b) if b_counts.size > 0 else 0.0
    top_act_frac = float(np.max(a_counts) / tot_a) if a_counts.size > 0 else 0.0

    distinct_bands = int(np.count_nonzero(b_counts))
    distinct_actions = int(np.count_nonzero(a_counts))

    b_entropy = float(shannon_entropy(b_counts))
    a_entropy = float(shannon_entropy(a_counts))
    m_entropy = float(shannon_entropy(m_counts))

    # Emitter Discovery
    disc_set = set(discovered_emitters or [])
    active_set = set(all_active_emitters or [])
    if active_set:
        discovery_rate = float(len(disc_set & active_set) / len(active_set))
        missed_count = len(active_set - disc_set)
    else:
        discovery_rate = 1.0 if disc_set else 0.0
        missed_count = 0

    # Operational Spectrum Coverage
    if spectrum_active_opportunities > 0:
        coverage = float(selected_active_opportunities / spectrum_active_opportunities)
    else:
        coverage = 0.0

    avg_rew = float(total_reward / steps)

    # Phase 6: Mode fractions (Task 6.1)
    tot_m = max(1, int(np.sum(m_counts)))
    short_f = float(m_counts[0] / tot_m) if len(m_counts) > 0 else 0.0
    normal_f = float(m_counts[1] / tot_m) if len(m_counts) > 1 else 0.0
    long_f = float(m_counts[2] / tot_m) if len(m_counts) > 2 else 0.0
    revisit_f = float(m_counts[3] / tot_m) if len(m_counts) > 3 else 0.0
    preempt_f = float(m_counts[4] / tot_m) if len(m_counts) > 4 else 0.0

    # Phase 6: Dwell- vs Physical Time-Normalized metrics (Task 6.2)
    tot_ms = max(1e-6, float(total_mission_time_us / 1000.0)) if total_mission_time_us is not None else float(steps * 0.5)
    ir_dwell = ir
    ir_ms = float(hits / tot_ms)
    rew_dwell = avg_rew
    rew_ms = float(total_reward / tot_ms)

    # First hit metrics (None if 0 hits per Phase 6 specification)
    if hits <= 0:
        f_hit_lat = None
        f_hit_ms = None
    else:
        f_hit_lat = first_hit_latency_us
        f_hit_ms = mission_time_to_first_intercept_ms

    return CanonicalMetrics(
        interception_rate=ir,
        hits=hits,
        steps=steps,
        pd=pd_val,
        pfa=pfa_val,
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        tn=int(tn),
        active_opportunities=denom_pd,
        avg_intercept_time_error_us=avg_latency,
        operational_intercept_latency_us=op_latency,
        predictive_time_error_us=pred_error,
        prediction_coverage=pred_cov,
        first_detection_time_us=first_detection_time_us,
        distinct_bands=distinct_bands,
        distinct_actions=distinct_actions,
        top_band_fraction=top_band_frac,
        top_action_fraction=top_act_frac,
        band_entropy=b_entropy,
        mode_entropy=m_entropy,
        action_entropy=a_entropy,
        discovery_rate=discovery_rate,
        discovered_emitter_count=len(disc_set),
        total_emitter_count=len(active_set),
        missed_active_emitter_count=missed_count,
        operational_coverage=coverage,
        avg_reward=avg_rew,
        total_reward=float(total_reward),
        manifest=manifest,
        short_fraction=short_f,
        normal_fraction=normal_f,
        long_fraction=long_f,
        revisit_fraction=revisit_f,
        preemptive_fraction=preempt_f,
        ir_per_dwell=ir_dwell,
        ir_per_ms=ir_ms,
        reward_per_dwell=rew_dwell,
        reward_per_ms=rew_ms,
        first_hit_latency_us=f_hit_lat,
        mission_time_to_first_intercept_ms=f_hit_ms,
        total_mission_time_ms=tot_ms,
    )

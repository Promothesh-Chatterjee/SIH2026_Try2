"""
Figures of Merit for Cognitive EW Evaluation.

Tracks Pd, Pfa, intercept rate/time-error, and ROC used by SIH judges.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _counts(comb: int) -> int:
    """Number of unordered pairs among ``comb`` items: n(n-1)/2.

    Args:
        comb: Number of items in a set.

    Returns:
        Pair count ``comb * (comb - 1) // 2`` (0 for comb < 2).
    """
    comb = int(comb)
    return comb * (comb - 1) // 2 if comb >= 2 else 0


pair_count = _counts  # public alias used by tests/consumers


def pairwise_cluster_counts(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ignore_noise: bool = True,
) -> dict[str, int]:
    """Compute permutation-invariant pairwise TP/FP/FN/TN for clustering.

    Pair definition over all unordered pulse pairs (i < j):
      TP: truth(same_emitter) AND pred(same_cluster)
      FP: truth(different_emitter) AND pred(same_cluster)
      FN: truth(same_emitter) AND pred(different_cluster)
      TN: truth(different_emitter) AND pred(different_cluster)

    These counts are independent of numeric label identity (permutation-
    invariant). Computed in O(N) from the contingency matrix using the
    identities TP = sum_c sum_e C(m_{c,e},2), FP = sum_c C(s_c,2) - TP,
    FN = sum_e C(t_e,2) - TP, avoiding any O(N^2) pair matrix.

    Args:
        true_labels: (N,) ground-truth emitter IDs.
        pred_labels: (N,) predicted cluster IDs (use -1/noise for non-clustered).
        ignore_noise: If True, drop pulses whose true label is -1 (noise)
            before counting (matches "non-noise only" convention).

    Returns:
        Dict {"tp","fp","fn","tn","n_pairs"}.

    Raises:
        ValueError: If label arrays differ in length.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    if true_labels.ndim != 1 or pred_labels.ndim != 1 or true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError(f"Expected equal-length 1-D label arrays, got {true_labels.shape} vs {pred_labels.shape}")

    if true_labels.size == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_pairs": 0}

    if ignore_noise:
        keep = true_labels != -1
        true_labels = true_labels[keep]
        pred_labels = pred_labels[keep]

    n = true_labels.size
    total_pairs = _counts(n)
    if n == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_pairs": 0}

    # Build contingency matrix counts keyed by (cluster, emitter).
    # Contingency: rows = predicted clusters, cols = true emitters.
    from collections import Counter

    bc = Counter(zip(pred_labels.tolist(), true_labels.tolist()))
    row_sums: dict[int, int] = Counter(pred_labels.tolist())
    col_sums: dict[int, int] = Counter(true_labels.tolist())

    tp = int(sum(_counts(c) for c in bc.values()))
    fp = int(sum(_counts(row_sums[c]) for c in row_sums) - tp)
    fn = int(sum(_counts(col_sums[e]) for e in col_sums) - tp)
    tn = int(total_pairs - tp - fp - fn)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pairs": int(total_pairs)}


def pairwise_clustering_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ignore_noise: bool = True,
) -> dict[str, float]:
    """Permutation-invariant pairwise MCC and F1 from clustering counts.

    Args:
        true_labels: (N,) ground-truth emitter IDs.
        pred_labels: (N,) predicted cluster IDs (-1 = noise).
        ignore_noise: Drop true-noise pulses before counting.

    Returns:
        Dict {"pairwise_mcc","pairwise_f1","tp","fp","fn","tn","n_pairs"}.
        MCC/F1 are 0.0 when the denominator is degenerate (all pairs one class).
    """
    c = pairwise_cluster_counts(true_labels, pred_labels, ignore_noise=ignore_noise)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom > 0:
        mcc = (tp * tn - fp * fn) / denom
    elif tp > 0 and fp == 0 and fn == 0:
        # Perfect clustering with no negative pairs (degenerate): MCC = +1.
        mcc = 1.0
    else:
        mcc = 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "pairwise_mcc": float(mcc),
        "pairwise_f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "n_pairs": float(c["n_pairs"]),
    }


def deinterleaver_train_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    with_pairwise: bool = True,
) -> dict[str, float]:
    """Aggregate the official deinterleaving metrics for a single pulse train.

    Computes V-measure, ARI, AMI, homogeneity, completeness (sklearn) plus the
    scaled permutation-invariant pairwise MCC/F1 and noise/cluster diagnostics.

    Args:
        labels_true: (N,) ground-truth emitter IDs (-1 = noise if present).
        labels_pred: (N,) predicted cluster IDs (-1 = noise).
        with_pairwise: Include pairwise MCC/F1 (scalable).

    Returns:
        Dict of float metrics: v_measure, ari, ami, homogeneity, completeness,
        pairwise_mcc, pairwise_f1, noise_fraction, n_clusters_predicted,
        n_emitters_true, num_pulses. Missing sklearn metrics are NaN.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        v_measure_score,
    )

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    lt = labels_true.astype(int)
    lp = labels_pred.astype(int)

    n = len(lt)
    non_noise_true = int(np.sum(lt != -1))
    noise_fraction = float(np.sum(lt == -1) / n) if n else 0.0

    n_emitters_true = int(len(set(lt.tolist())) - (1 if -1 in set(lt.tolist()) else 0))
    n_clusters_predicted = int(len(set(lp.tolist())) - (1 if -1 in set(lp.tolist()) else 0))

    # sklearn metrics: operate on non-noise truth pulses against non-noise preds.
    mask = (lt != -1) & (lp != -1)
    out: dict[str, float] = {
        "noise_fraction": noise_fraction,
        "n_clusters_predicted": float(n_clusters_predicted),
        "n_emitters_true": float(n_emitters_true),
        "num_pulses": float(n),
    }

    if np.sum(mask) > 1 and n_emitters_true > 1:
        lt_keep = lt[mask]
        lp_keep = lp[mask]
        try:
            out["v_measure"] = float(v_measure_score(lt_keep, lp_keep))
        except ValueError:
            out["v_measure"] = float("nan")
        try:
            out["ari"] = float(adjusted_rand_score(lt_keep, lp_keep))
        except ValueError:
            out["ari"] = float("nan")
        try:
            out["ami"] = float(adjusted_mutual_info_score(lt_keep, lp_keep))
        except ValueError:
            out["ami"] = float("nan")
        try:
            out["homogeneity"] = float(homogeneity_score(lt_keep, lp_keep))
        except ValueError:
            out["homogeneity"] = float("nan")
        try:
            out["completeness"] = float(completeness_score(lt_keep, lp_keep))
        except ValueError:
            out["completeness"] = float("nan")
    else:
        for k in ["v_measure", "ari", "ami", "homogeneity", "completeness"]:
            out[k] = float("nan")

    if with_pairwise:
        pw = pairwise_clustering_metrics(lt, lp, ignore_noise=True)
        out["pairwise_mcc"] = pw["pairwise_mcc"]
        out["pairwise_f1"] = pw["pairwise_f1"]

    return out


def aggregate_deinterleaver_metrics(per_train: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-train deinterleaver metrics across a dataset.

    Reports signed mean/median/percentiles per metric plus failure counts for
    the metrics the SIH contract care about (V-measure, ARI, AMI, homogeneity,
    completeness, pairwise MCC/F1).

    Args:
        per_train: List of dicts from :func:`deinterleaver_train_metrics`.

    Returns:
        Flat dict keyed ``<metric>_mean`` , ``<metric>_median`` , ``<metric>_p25`` ,
        ``<metric>_p75`` , ``<metric>_nan_count`` , plus ``n_trains``.
    """
    if not per_train:
        return {"n_trains": 0.0}

    keys = [
        "v_measure",
        "ari",
        "ami",
        "homogeneity",
        "completeness",
        "pairwise_mcc",
        "pairwise_f1",
    ]
    agg: dict[str, float] = {"n_trains": float(len(per_train))}
    for key in keys:
        vals = np.asarray([r.get(key, float("nan")) for r in per_train], dtype=np.float64)
        good = vals[np.isfinite(vals)]
        agg[f"{key}_nan_count"] = float(int(np.sum(~np.isfinite(vals))))
        if good.size > 0:
            agg[f"{key}_mean"] = float(np.mean(good))
            agg[f"{key}_median"] = float(np.median(good))
            p25, p75 = np.percentile(good, [25, 75])
            agg[f"{key}_p25"] = float(p25)
            agg[f"{key}_p75"] = float(p75)
        else:
            for suffix in ["_mean", "_median", "_p25", "_p75"]:
                agg[f"{key}{suffix}"] = float("nan")

    noise = [r.get("noise_fraction", float("nan")) for r in per_train]
    noise_good = [x for x in noise if x == x]
    agg["noise_fraction_mean"] = float(np.mean(noise_good)) if noise_good else float("nan")

    clusters = [r.get("n_clusters_predicted", float("nan")) for r in per_train]
    clusters_good = [x for x in clusters if x == x]
    agg["n_clusters_predicted_mean"] = float(np.mean(clusters_good)) if clusters_good else float("nan")

    return agg


class FiguresOfMerit:
    """Accumulates SIH-required evaluation metrics over an episode.

    Metrics:
        Pd — Probability of Detection (hits / active opportunities)
        Pfa — Probability of False Alarm (false tunes / total tunes)
        Avg Intercept Rate — hits / total steps
        Avg Intercept Time Error — mean |predicted - actual| ToA on hits
        Percentage Correct Predictions — alias for Pd*100
        Avg Reward — mean shaped reward per step
        Sensitivity proxy — same as Pd for ES receiver

    Attributes:
        n_steps: Total scheduling steps.
        n_hits: Steps where chosen band had ≥1 pulse.
        n_misses: Steps where active pulses existed elsewhere but not intercepted.
        n_false_alarms: Steps tuned to empty band (no active pulses anywhere or miss).
        n_active_opportunities: Steps where at least one band had pulses.
        intercept_time_errors: List of timing errors on hits.
        rewards: List of per-step rewards.
    """

    def __init__(self, n_bands: int = 36) -> None:
        """Initialise all counters to zero.

        Args:
            n_bands: Number of frequency bands in the scan grid. Used to size the
                scalar ground-truth path (full-band vector path is authoritative).
        """
        self.n_bands = int(n_bands)
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators for a new episode."""
        self.n_steps: int = 0
        self.n_hits: int = 0
        self.n_misses: int = 0
        self.n_false_alarms: int = 0
        self.n_active_opportunities: int = 0

        # Rigorous Confusion-Opportunity Counters (P0-7)
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

        self.intercept_time_errors: list[float] = []
        self.rewards: list[float] = []
        self._roc_points: list[tuple[float, float]] = []

        # Reward component accumulators (SIH eval contract: log terms separately).
        self.reward_hit_term: float = 0.0
        self.reward_novel_term: float = 0.0
        self.reward_timing_penalty: float = 0.0
        self.reward_miss_penalty: float = 0.0
        self._reward_count: int = 0

        # Revisit and Emitter Tracking
        self.last_visit_per_band: dict[int, int] = {}
        self.revisit_intervals: list[int] = []
        self.discovered_emitters: set[int] = set()
        self.all_active_emitters: set[int] = set()

    def record_reward_components(self, components: dict[str, float] | None) -> None:
        """Accumulate an optionally-provided per-component reward breakdown.

        Args:
            components: Dict with keys hit_term, novel_term, timing_penalty,
                miss_penalty (as returned by receiver_reward_components). Missing
                keys default to 0.0.
        """
        if not components:
            return
        self._reward_count += 1
        self.reward_hit_term += float(components.get("hit_term", 0.0))
        self.reward_novel_term += float(components.get("novel_term", 0.0))
        self.reward_timing_penalty += float(components.get("timing_penalty", 0.0))
        self.reward_miss_penalty += float(components.get("miss_penalty", 0.0))

    def record_emitters(self, active_now: set[int], intercepted_now: set[int]) -> None:
        """Track unique emitter discovery progress."""
        self.all_active_emitters.update(active_now)
        self.discovered_emitters.update(intercepted_now)

    def update(
        self,
        band_chosen: int,
        ground_truth_active: np.ndarray | list[int] | bool,
        pred_active: bool,
        intercept_time_error_us: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Update accumulators for one scheduling step (decision-level contract).

        The chosen band's dwell is the only evaluated opportunity (SIH Evaluation
        Contract): unselected active bands are not counted as misses, so Pd is
        decision-aligned and Pfa reflects documented false-alarm opportunities.

        Args:
            band_chosen: Index of tuned band.
            ground_truth_active: Boolean or array indicating active bands during dwell.
            pred_active: Whether receiver declared detection on the chosen band.
            intercept_time_error_us: Real receiver-clock timing error of first pulse
                on hit (NaN when no intercept).
            reward: Shaped reward.
        """
        self.n_steps += 1
        self.rewards.append(float(reward))

        # Revisit latency tracking
        b = int(band_chosen)
        if b in self.last_visit_per_band:
            interval = self.n_steps - self.last_visit_per_band[b]
            self.revisit_intervals.append(interval)
        self.last_visit_per_band[b] = self.n_steps

        # Standardize ground_truth_active
        if isinstance(ground_truth_active, (bool, int, np.bool_)):
            is_active = bool(ground_truth_active)
            gt_vec = np.zeros(self.n_bands, dtype=np.int8)
            if is_active and 0 <= b < self.n_bands:
                gt_vec[b] = 1
        else:
            gt_vec = np.asarray(ground_truth_active).astype(np.int8)
            is_active = bool(gt_vec[b]) if 0 <= b < len(gt_vec) else False

        # ------------------------------------------------------------------
        # Decision-level Confusion-Opportunity contract (P0-7, SIH "7.
        # Evaluation contract"):
        #   The scheduler makes ONE decision per step: dwell on band `b`.
        #   The ONLY opportunity it can intercept or miss is that dwell.
        #   - tuned band active + detected  -> TP  (hit / intercept)
        #   - tuned band active + undetected-> FN  (miss on a real opportunity)
        #   - tuned band inactive + detected -> FP  (false alarm)
        #   - tuned band inactive, empty dwell-> TN (correct reject)
        #   Unselected active bands elsewhere are NOT counted as misses
        #   (per the contract) — they are not opportunities the scheduler
        #   acted upon.
        # ------------------------------------------------------------------
        if is_active and pred_active:
            self.tp += 1
            self.n_hits += 1
            self.n_active_opportunities += 1
            self.intercept_time_errors.append(float(intercept_time_error_us))
        elif is_active and not pred_active:
            self.fn += 1
            self.n_misses += 1
            self.n_active_opportunities += 1
        elif not is_active and pred_active:
            self.fp += 1
            self.n_false_alarms += 1
        else:
            self.tn += 1

        # Incremental ROC point
        step_pd = self.pd
        step_pfa = self.pfa
        self._roc_points.append((step_pfa, step_pd))

    @property
    def pd(self) -> float:
        """Probability of Detection: intercepted opportunities / defined active opportunities.

        Defined opportunity = a dwell where the scheduler's chosen band was active
        (TP intercepted, FN missed). Unselected active bands are not counted.
        """
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def pfa(self) -> float:
        """Probability of False Alarm: false detections / false-alarm opportunities.

        False-alarm opportunity = a dwell where the chosen band carried no active
        signal (FP spurious detection, TN correctly empty).
        """
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom > 0 else 0.0

    @property
    def avg_intercept_rate(self) -> float:
        """Hits per scheduling step."""
        return float(self.n_hits / self.n_steps) if self.n_steps > 0 else 0.0

    @property
    def avg_intercept_time_error(self) -> float:
        """Mean real intercept-time error (µs) across actual intercepts (hits).

        NaN entries (no intercept) are excluded. This is a genuinely measured
        quantity from the receiver clock, never hard-coded to zero.
        """
        vals = [x for x in self.intercept_time_errors if x == x]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_reward(self) -> float:
        """Mean reward per step."""
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    def _avg_component(self, total: float) -> float:
        """Mean of an accumulated reward component over component-logged steps."""
        if self._reward_count <= 0:
            return 0.0
        return float(total / self._reward_count)

    @property
    def unique_emitter_discovery_rate(self) -> float:
        """Fraction of total active emitters discovered."""
        if not self.all_active_emitters:
            return 1.0 if self.discovered_emitters else 0.0
        return float(len(self.discovered_emitters) / len(self.all_active_emitters))

    def revisit_latency_percentiles(self) -> dict[str, float]:
        """Compute p50, p90, p99 revisit latencies across bands."""
        if not self.revisit_intervals:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        arr = np.asarray(self.revisit_intervals)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    def summary(self) -> dict[str, float]:
        """Return all scientific figures of merit."""
        rev = self.revisit_latency_percentiles()
        total_decisions = max(1, self.tp + self.tn + self.fp + self.fn)
        pct_correct = ((self.tp + self.tn) / total_decisions) * 100.0

        return {
            "Pd": float(self.pd),
            "Pfa": float(self.pfa),
            "sensitivity": float(self.pd),
            "avg_intercept_rate": float(self.avg_intercept_rate),
            "avg_intercept_time_error_us": float(self.avg_intercept_time_error),
            "pct_correct_predictions": float(pct_correct),
            "avg_reward": float(self.avg_reward),
            "avg_reward_hit_term": self._avg_component(self.reward_hit_term),
            "avg_reward_novel_term": self._avg_component(self.reward_novel_term),
            "avg_reward_timing_penalty": self._avg_component(self.reward_timing_penalty),
            "avg_reward_miss_penalty": self._avg_component(self.reward_miss_penalty),
            "discovery_rate": float(self.unique_emitter_discovery_rate),
            "revisit_p50": float(rev["p50"]),
            "revisit_p90": float(rev["p90"]),
            "revisit_p99": float(rev["p99"]),
            "n_steps": float(self.n_steps),
            "n_hits": float(self.n_hits),
            "n_misses": float(self.n_misses),
            "n_false_alarms": float(self.n_false_alarms),
            "n_active_opportunities": float(self.n_active_opportunities),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "tn": float(self.tn),
        }

    def plot_roc_curve(self, save_path: str | Path = "roc_curve.pdf") -> Path:
        """Save Pd vs Pfa operating curve as PDF.

        Args:
            save_path: Output file path (PDF).

        Returns:
            Path to saved file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._roc_points:
            logger.warning("No ROC points collected — saving empty plot")
            pfa = [0, 1]
            pd_vals = [0, 1]
        else:
            pfa, pd_vals = zip(*self._roc_points, strict=False)
            # Ensure curve starts at origin and ends sorted
            pfa = list(pfa)
            pd_vals = list(pd_vals)

        plt.figure(figsize=(6, 5), dpi=300)
        plt.plot(pfa, pd_vals, marker=".", linewidth=1.5, label=f"Pd={self.pd:.3f}, Pfa={self.pfa:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="chance")
        plt.xlabel("Pfa (False Alarm Rate)")
        plt.ylabel("Pd (Detection Rate)")
        plt.title("ROC — Cognitive EW SmartScan (Pd vs Pfa)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, format="pdf")
        plt.close()
        logger.info("ROC curve saved to %s", save_path)
        return save_path

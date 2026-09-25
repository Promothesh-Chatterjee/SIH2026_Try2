"""Electronic Warfare (EW) Figures of Merit (FoMs).

This module implements the 7 core Figures of Merit required by the DRDO
problem statement for Electronic Support (ES) receiver scheduling:

1. Probability of Detection (Pd):
   True Pd = TP / (TP + FN), where TP is successful detection when tuned
   to an active band, and FN is failure to detect on an active band (e.g.
   below SNR/detection threshold).
2. Probability of False Alarm (Pfa):
   Pfa = FP / N_receiver_dwells, where FP is detection reported on an inactive band.
3. Sensitivity (Minimum Detectable Signal, S_min):
   Configured receiver sensitivity floor in dBm (default -140.0 dBm).
4. Average Intercept Rate:
   Fraction of receiver dwell steps resulting in a signal interception
   (hits / N_dwells).
5. Average Reward:
   Mean reward earned per episode step.
6. Percentage of Correct Predictions:
   Accuracy across all dwell decisions: (TP + TN) / N_steps * 100.
7. Average Intercept Time Error:
   Mean absolute error (MAE) between predicted illumination times and
   actual illumination times for periodic/scanning emitters.

All metric calculation functions are strictly stateless and functional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import numpy as np


@dataclass(frozen=True)
class EWMetrics:
    """Immutable container for all 7 Electronic Warfare Figures of Merit.

    Attributes
    ----------
    pd : float
        Probability of Detection: TP / (TP + FN).
    pfa : float
        Probability of False Alarm: FP / N_receiver_dwells.
    sensitivity_dbm : float
        Minimum detectable signal threshold in dBm.
    avg_intercept_rate : float
        Fraction of receiver dwell steps resulting in an intercept (hits / dwells).
    avg_reward : float
        Mean episodic step reward.
    pct_correct_predictions : float
        Percentage of steps where receiver decision matched ground truth (0-100%).
    avg_intercept_time_error_us : float
        Mean absolute error between predicted and actual intercept times.
    n_intercepts : int
        Total true positive interceptions (TP).
    n_false_alarms : int
        Total false alarm events (FP).
    n_total_transmissions : int
        Total emitter transmission events across all bands during the episode.
    n_receiver_dwells : int
        Total receiver dwell steps executed.
    n_missed_dwells : int
        Dwells on active bands that failed to detect (FN).
    """

    pd: float
    pfa: float
    sensitivity_dbm: float
    avg_intercept_rate: float
    avg_reward: float
    pct_correct_predictions: float
    avg_intercept_time_error_us: float
    operational_intercept_latency_us: float = 0.0
    predictive_time_error_us: float | None = None
    prediction_coverage: float = 0.0
    n_intercepts: int = 0
    n_false_alarms: int = 0
    n_total_transmissions: int = 0
    n_receiver_dwells: int = 0
    n_missed_dwells: int = 0
    n_true_negatives: int = 0
    canonical_pfa: float = 0.0
    tp: int | None = None
    fn: int | None = None
    fp: int | None = None
    tn: int | None = None

    def __post_init__(self):
        # Harmonize TP <-> n_intercepts
        if self.tp is None:
            object.__setattr__(self, "tp", int(self.n_intercepts))
        elif int(self.tp) != int(self.n_intercepts):
            raise ValueError(
                f"Conflicting values provided for canonical tp ({self.tp}) and alias n_intercepts ({self.n_intercepts})"
            )
        object.__setattr__(self, "n_intercepts", int(self.tp))

        # Harmonize FN <-> n_missed_dwells
        if self.fn is None:
            object.__setattr__(self, "fn", int(self.n_missed_dwells))
        elif int(self.fn) != int(self.n_missed_dwells):
            raise ValueError(
                f"Conflicting values provided for canonical fn ({self.fn}) and alias n_missed_dwells ({self.n_missed_dwells})"
            )
        object.__setattr__(self, "n_missed_dwells", int(self.fn))

        # Harmonize FP <-> n_false_alarms
        if self.fp is None:
            object.__setattr__(self, "fp", int(self.n_false_alarms))
        elif int(self.fp) != int(self.n_false_alarms):
            raise ValueError(
                f"Conflicting values provided for canonical fp ({self.fp}) and alias n_false_alarms ({self.n_false_alarms})"
            )
        object.__setattr__(self, "n_false_alarms", int(self.fp))

        # Harmonize TN <-> n_true_negatives
        if self.tn is None:
            object.__setattr__(self, "tn", int(self.n_true_negatives))
        elif int(self.tn) != int(self.n_true_negatives):
            raise ValueError(
                f"Conflicting values provided for canonical tn ({self.tn}) and alias n_true_negatives ({self.n_true_negatives})"
            )
        object.__setattr__(self, "n_true_negatives", int(self.tn))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Figures of Merit to dictionary without truthiness-based fallbacks."""
        tp_val = int(self.tp)
        fn_val = int(self.fn)
        fp_val = int(self.fp)
        tn_val = int(self.tn)
        return {
            "pd": float(self.pd),
            "pfa": float(self.pfa),
            "sensitivity_dbm": float(self.sensitivity_dbm),
            "avg_intercept_rate": float(self.avg_intercept_rate),
            "mean_intercept_rate": float(self.avg_intercept_rate),
            "avg_reward": float(self.avg_reward),
            "pct_correct_predictions": float(self.pct_correct_predictions),
            "avg_intercept_time_error_us": float(self.avg_intercept_time_error_us),
            "operational_intercept_latency_us": float(self.operational_intercept_latency_us),
            "predictive_time_error_us": float(self.predictive_time_error_us) if self.predictive_time_error_us is not None else None,
            "prediction_coverage": float(self.prediction_coverage),
            "tp": tp_val,
            "fn": fn_val,
            "fp": fp_val,
            "tn": tn_val,
            "n_intercepts": tp_val,
            "n_false_alarms": fp_val,
            "n_total_transmissions": int(self.n_total_transmissions),
            "n_receiver_dwells": int(self.n_receiver_dwells),
            "n_missed_dwells": fn_val,
            "n_true_negatives": tn_val,
            "canonical_pfa": float(self.canonical_pfa),
        }


def compute_pd(n_true_positives: int, n_false_negatives: int = 0) -> float:
    """Compute Probability of Detection: Pd = TP / (TP + FN).

    Parameters
    ----------
    n_true_positives : int
        Number of times the receiver was tuned to an active band and successfully
        detected the emitter transmission.
    n_false_negatives : int, optional
        Number of times the receiver was tuned to an active band but failed to
        detect (e.g. signal power below sensitivity threshold). Default is 0.

    Returns
    -------
    float
        Probability of Detection in [0.0, 1.0]. Returns 0.0 if (TP + FN) == 0.

    Note
    ----
    In simulation environments where tuning to an active band guarantees detection,
    FN = 0 on monitored active bands, giving Pd = 1.0 for monitored active bands.
    This is conceptually distinct from `avg_intercept_rate` (which normalizes by
    all receiver dwell steps) and mission intercept ratio (which normalizes by
    all emitter transmissions across all bands).
    """
    tp = int(n_true_positives)
    fn = int(n_false_negatives)
    denom = tp + fn
    if denom <= 0:
        return 0.0
    return float(np.clip(float(tp) / float(denom), 0.0, 1.0))


def compute_pfa(
    n_false_alarms: int,
    n_receiver_dwells: int = 0,
    n_true_negatives: int | None = None,
) -> float:
    """Compute Probability of False Alarm.

    If n_true_negatives is provided, computes canonical decision-level
    Pfa = FP / (FP + TN), aligning with canonical_metrics.py.
    Otherwise computes dwell-normalized Pfa = FP / N_receiver_dwells.

    Parameters
    ----------
    n_false_alarms : int
        Number of false alarms (FP).
    n_receiver_dwells : int, optional
        Total number of receiver dwell steps.
    n_true_negatives : int | None, optional
        Number of true negatives (TN: inactive dwells without false alarm).

    Returns
    -------
    float
        Probability of False Alarm in [0.0, 1.0].
    """
    fp = int(n_false_alarms)
    if n_true_negatives is not None:
        denom = fp + int(n_true_negatives)
        if denom <= 0:
            return 0.0
        return float(np.clip(float(fp) / float(denom), 0.0, 1.0))

    dwells = int(n_receiver_dwells)
    if dwells <= 0:
        return 0.0
    return float(np.clip(float(fp) / float(dwells), 0.0, 1.0))


def compute_canonical_pfa(n_false_alarms: int, n_true_negatives: int) -> float:
    """Authoritative decision-level Pfa = FP / (FP + TN) matching canonical_metrics.py."""
    return compute_pfa(n_false_alarms=n_false_alarms, n_true_negatives=n_true_negatives)


def compute_canonical_decision_metrics(
    n_true_positives: int,
    n_false_negatives: int,
    n_false_alarms: int,
    n_true_negatives: int,
) -> dict[str, float]:
    """Compute canonical decision-level Pd and Pfa matching canonical_metrics.py."""
    pd = compute_pd(n_true_positives, n_false_negatives)
    pfa = compute_canonical_pfa(n_false_alarms, n_true_negatives)
    return {"pd": pd, "pfa": pfa}


def compute_avg_intercept_rate(hits: Sequence[bool] | np.ndarray) -> float:
    """Compute Average Intercept Rate: fraction of dwells that intercepted a signal.

    Parameters
    ----------
    hits : Sequence[bool] | np.ndarray
        Boolean sequence indicating whether each dwell resulted in an interception.

    Returns
    -------
    float
        Fraction in [0.0, 1.0]. Returns 0.0 if hits is empty.
    """
    arr = np.asarray(hits, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def compute_avg_reward(rewards: Sequence[float] | np.ndarray) -> float:
    """Compute Average Reward per episode step.

    Parameters
    ----------
    rewards : Sequence[float] | np.ndarray
        Sequence of rewards earned per step.

    Returns
    -------
    float
        Mean reward. Returns 0.0 if rewards is empty.
    """
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def compute_pct_correct_predictions(
    hits: Sequence[bool] | np.ndarray,
    active_mask: Sequence[bool] | np.ndarray,
    *,
    tp: int | None = None,
    tn: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
) -> float:
    """Compute Percentage of Correct Decisions across all dwell steps.

    When decision-level confusion-matrix counts (tp, tn, fp, fn) are
    provided, the authoritative selected-band formula is used:

        pct_correct = (TP + TN) / (TP + TN + FP + FN) × 100

    This is the CANONICAL definition aligned with the SIH evaluation
    contract: a decision is correct if the scheduler selected a band
    that was active AND detected it (TP), or selected an inactive band
    AND did not false-alarm (TN).

    Parameters
    ----------
    hits : Sequence[bool] | np.ndarray
        Boolean sequence indicating if each dwell resulted in an interception.
        Used only in the legacy fallback path.
    active_mask : Sequence[bool] | np.ndarray
        Boolean sequence indicating if any emitter was active during that step.
        Used only in the legacy fallback path (DEPRECATED — uses spectrum-wide
        mask rather than selected-band activity, producing incorrect results
        when active emissions exist on unselected bands).
    tp : int, optional
        True Positives from selected-band confusion matrix.
    tn : int, optional
        True Negatives from selected-band confusion matrix.
    fp : int, optional
        False Positives from selected-band confusion matrix.
    fn : int, optional
        False Negatives from selected-band confusion matrix.

    Returns
    -------
    float
        Percentage in [0.0, 100.0]. Returns 0.0 if all inputs are empty/zero.
    """
    # Authoritative path: use decision-level confusion matrix
    if tp is not None and tn is not None:
        _tp = int(tp)
        _tn = int(tn)
        _fp = int(fp) if fp is not None else 0
        _fn = int(fn) if fn is not None else 0
        total = _tp + _tn + _fp + _fn
        if total == 0:
            return 0.0
        return float((_tp + _tn) / total * 100.0)

    # Legacy fallback (DEPRECATED): spectrum-wide active mask comparison.
    # This path produces incorrect results when active emissions exist on
    # unselected bands (marking correct TN decisions as wrong).
    import warnings
    warnings.warn(
        "compute_pct_correct_predictions called without confusion matrix counts. "
        "Using deprecated spectrum-wide active_mask comparison which may produce "
        "incorrect results. Pass tp/tn/fp/fn for the canonical selected-band metric.",
        DeprecationWarning,
        stacklevel=2,
    )
    h = np.asarray(hits, dtype=bool)
    m = np.asarray(active_mask, dtype=bool)
    if h.size == 0 or m.size == 0:
        return 0.0
    min_len = min(h.size, m.size)
    correct = (h[:min_len] == m[:min_len])
    return float(np.mean(correct) * 100.0)


def compute_avg_intercept_time_error(
    predicted_times: Sequence[float] | np.ndarray,
    actual_times: Sequence[float] | np.ndarray,
) -> float:
    """Compute Mean Absolute Error (MAE) between predicted and actual intercept times.

    Parameters
    ----------
    predicted_times : Sequence[float] | np.ndarray
        Predicted arrival time steps or microseconds.
    actual_times : Sequence[float] | np.ndarray
        Actual arrival time steps or microseconds.

    Returns
    -------
    float
        Mean absolute error. Returns 0.0 if either sequence is empty.
    """
    pred = np.asarray(predicted_times, dtype=np.float64)
    actual = np.asarray(actual_times, dtype=np.float64)
    if pred.size == 0 or actual.size == 0:
        return 0.0
    k = min(pred.size, actual.size)
    return float(np.mean(np.abs(pred[:k] - actual[:k])))


def compute_all_metrics(
    episode_log: Dict[str, Any],
    min_detectable_signal_dbm: float = -140.0,
) -> EWMetrics:
    """Aggregate all 7 Figures of Merit from an episode log dictionary.

    Compatible with `SpectrumEnvironment.get_episode_log()`.

    Parameters
    ----------
    episode_log : dict
        Telemetry log dictionary containing at minimum:
        - "hits": list of bool
        - "chosen_bands": list of int
        - "active_bands_per_step": list of list of int
        - "rewards": list of float
        Optionally:
        - "predicted_times": list of float
        - "actual_times": list of float
        - "false_alarms": list of bool
    min_detectable_signal_dbm : float, optional
        Minimum detectable signal threshold in dBm. Default is -140.0 dBm.

    Returns
    -------
    EWMetrics
        Frozen dataclass holding all 7 FoMs and underlying raw counters.
    """
    hits = episode_log.get("hits", [])
    chosen_bands = episode_log.get("chosen_bands", [])
    active_bands_per_step = episode_log.get("active_bands_per_step", [])
    rewards = episode_log.get("rewards", episode_log.get("episode_rewards", []))

    n_dwells = len(hits)
    
    # Calculate TP, FN, FP, TN
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    # Single source of truth for decision-level confusion matrix accounting
    # For every receiver dwell:
    # Selected band active + detected -> TP
    # Selected band active + not detected -> FN
    # Selected band inactive + detected -> FP
    # Selected band inactive + not detected -> TN
    if chosen_bands and active_bands_per_step and len(chosen_bands) > 0:
        n_eval = min(n_dwells, len(chosen_bands), len(active_bands_per_step))
        for t in range(n_eval):
            b_chosen = chosen_bands[t]
            active_b = active_bands_per_step[t]
            hit = bool(hits[t]) if t < len(hits) else False
            if b_chosen in active_b:
                if hit:
                    tp += 1
                else:
                    fn += 1
            else:
                if hit:
                    fp += 1
                else:
                    tn += 1
        if n_eval < n_dwells:
            tn += (n_dwells - n_eval)
    elif "false_alarms" in episode_log or "missed_dwells" in episode_log:
        explicit_false_alarms = episode_log.get("false_alarms", [])
        if isinstance(explicit_false_alarms, (list, tuple)):
            fp = int(sum(1 for f in explicit_false_alarms if f))
        else:
            fp = int(explicit_false_alarms or 0)
        missed_raw = episode_log.get("missed_dwells", 0)
        if isinstance(missed_raw, (list, tuple)):
            fn = int(sum(1 for m in missed_raw if m))
        else:
            fn = int(missed_raw or 0)
        total_hits = int(sum(1 for h in hits if h))
        tp = max(0, total_hits - fp)
        tn = max(0, n_dwells - tp - fn - fp)
    else:
        # Fallback when only hits are logged:
        tp = int(sum(1 for h in hits if h))
        fn = 0
        fp = 0
        tn = max(0, n_dwells - tp)

    # Invariant enforcement
    if (tp + fn + fp + tn) != n_dwells:
        if (tp + fn + fp) <= n_dwells:
            tn = n_dwells - (tp + fn + fp)
        else:
            raise ValueError(
                f"Confusion matrix counters do not sum to total dwells: "
                f"TP={tp}, FN={fn}, FP={fp}, TN={tn}, sum={tp+fn+fp+tn} != n_dwells={n_dwells}"
            )
    assert tp >= 0 and fn >= 0 and fp >= 0 and tn >= 0, f"Negative counters: TP={tp}, FN={fn}, FP={fp}, TN={tn}"

    # Total emitter transmission opportunities across all bands
    if "n_total_transmissions" in episode_log:
        n_total_transmissions = int(episode_log["n_total_transmissions"])
    else:
        n_total_transmissions = int(sum(len(b) for b in active_bands_per_step))

    # Active mask per step (was at least one emitter active anywhere in the spectrum?)
    if "active_mask" in episode_log:
        active_mask = episode_log["active_mask"]
    else:
        active_mask = [len(b) > 0 for b in active_bands_per_step]

    # Time error
    pred_times = episode_log.get("predicted_times", [])
    act_times = episode_log.get("actual_times", [])

    pd = compute_pd(n_true_positives=tp, n_false_negatives=fn)
    canonical_pfa = compute_canonical_pfa(n_false_alarms=fp, n_true_negatives=tn)
    pfa = canonical_pfa  # Authoritative decision-level Pfa = FP / (FP + TN)
    avg_intercept_rate = compute_avg_intercept_rate(hits)
    avg_reward = compute_avg_reward(rewards)
    pct_correct = compute_pct_correct_predictions(hits, active_mask, tp=tp, tn=tn, fp=fp, fn=fn)
    avg_time_error = compute_avg_intercept_time_error(pred_times, act_times)

    # Independent operational latency and predictive error separation (Phase 2)
    raw_op_latencies = episode_log.get("operational_latencies", [])
    if raw_op_latencies:
        valid_ops = [x for x in raw_op_latencies if x is not None and np.isfinite(x)]
        operational_latency = float(np.mean(valid_ops)) if valid_ops else 0.0
    else:
        operational_latency = float(avg_time_error)

    raw_pred_errors = episode_log.get("genuine_predictive_time_errors", [])
    if raw_pred_errors:
        valid_preds = [x for x in raw_pred_errors if x is not None and np.isfinite(x)]
        predictive_error = float(np.mean(valid_preds)) if valid_preds else None
        prediction_coverage = float(len(valid_preds) / max(1, tp))
    else:
        predictive_error = None
        prediction_coverage = 0.0

    return EWMetrics(
        pd=pd,
        pfa=pfa,
        sensitivity_dbm=float(min_detectable_signal_dbm),
        avg_intercept_rate=avg_intercept_rate,
        avg_reward=avg_reward,
        pct_correct_predictions=pct_correct,
        avg_intercept_time_error_us=avg_time_error,
        operational_intercept_latency_us=operational_latency,
        predictive_time_error_us=predictive_error,
        prediction_coverage=prediction_coverage,
        n_intercepts=tp,
        n_false_alarms=fp,
        n_total_transmissions=n_total_transmissions,
        n_receiver_dwells=n_dwells,
        n_missed_dwells=fn,
        n_true_negatives=tn,
        canonical_pfa=canonical_pfa,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
    )

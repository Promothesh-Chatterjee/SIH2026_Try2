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
    n_intercepts: int
    n_false_alarms: int
    n_total_transmissions: int
    n_receiver_dwells: int
    n_missed_dwells: int = 0
    n_true_negatives: int = 0
    canonical_pfa: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Figures of Merit to dictionary."""
        return {
            "pd": float(self.pd),
            "pfa": float(self.pfa),
            "sensitivity_dbm": float(self.sensitivity_dbm),
            "avg_intercept_rate": float(self.avg_intercept_rate),
            "avg_reward": float(self.avg_reward),
            "pct_correct_predictions": float(self.pct_correct_predictions),
            "avg_intercept_time_error_us": float(self.avg_intercept_time_error_us),
            "n_intercepts": int(self.n_intercepts),
            "n_false_alarms": int(self.n_false_alarms),
            "n_total_transmissions": int(self.n_total_transmissions),
            "n_receiver_dwells": int(self.n_receiver_dwells),
            "n_missed_dwells": int(self.n_missed_dwells),
            "n_true_negatives": int(self.n_true_negatives),
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
) -> float:
    """Compute Percentage of Correct Predictions across all dwell steps.

    Parameters
    ----------
    hits : Sequence[bool] | np.ndarray
        Boolean sequence indicating if each dwell resulted in an interception.
    active_mask : Sequence[bool] | np.ndarray
        Boolean sequence indicating if any emitter was active during that step.

    Returns
    -------
    float
        Percentage in [0.0, 100.0]. Returns 0.0 if arrays are empty.

    Note
    ----
    A dwell step is counted as correct if:
    - Receiver intercepted an active emitter (True Positive: hit=True, active=True), OR
    - Spectrum was silent and receiver did not false-alarm (True Negative: hit=False, active=False).
    """
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
    
    # If explicit false alarms logged, use them; otherwise verify against active_bands_per_step
    explicit_false_alarms = episode_log.get("false_alarms", None)
    if explicit_false_alarms is not None:
        fp = int(sum(1 for f in explicit_false_alarms if f))
        tp = int(sum(1 for h in hits if h))
        fn = int(episode_log.get("missed_dwells", 0))
        tn = max(0, n_dwells - tp - fn - fp)
    else:
        for t in range(n_dwells):
            b_chosen = chosen_bands[t] if t < len(chosen_bands) else -1
            active_b = active_bands_per_step[t] if t < len(active_bands_per_step) else []
            hit = bool(hits[t])

            is_active_band = b_chosen in active_b
            if is_active_band:
                if hit:
                    tp += 1
                else:
                    fn += 1
            else:
                if hit:
                    fp += 1
                else:
                    tn += 1

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
    pfa = compute_pfa(n_false_alarms=fp, n_receiver_dwells=n_dwells)
    canonical_pfa = compute_canonical_pfa(n_false_alarms=fp, n_true_negatives=tn)
    avg_intercept_rate = compute_avg_intercept_rate(hits)
    avg_reward = compute_avg_reward(rewards)
    pct_correct = compute_pct_correct_predictions(hits, active_mask)
    avg_time_error = compute_avg_intercept_time_error(pred_times, act_times)

    return EWMetrics(
        pd=pd,
        pfa=pfa,
        sensitivity_dbm=float(min_detectable_signal_dbm),
        avg_intercept_rate=avg_intercept_rate,
        avg_reward=avg_reward,
        pct_correct_predictions=pct_correct,
        avg_intercept_time_error_us=avg_time_error,
        n_intercepts=tp,
        n_false_alarms=fp,
        n_total_transmissions=n_total_transmissions,
        n_receiver_dwells=n_dwells,
        n_missed_dwells=fn,
        n_true_negatives=tn,
        canonical_pfa=canonical_pfa,
    )

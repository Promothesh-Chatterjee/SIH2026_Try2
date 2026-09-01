"""
Domain-specific Reward Functions.

1. compute_reward: shaped reward for RFScanEnv (novel intercepts, timing, miss).
2. compute_receiver_reward: reward derived purely from ReceiverObservation,
   designed for the receiver-driven CognitiveRFScanEnv. Ground truth (emitter_id)
   is only used for the "novel emitter" bonus — it is stripped before the
   scheduler observation is built and must never reach the policy input.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_reward(
    hit: bool,
    labels_intercepted: np.ndarray,
    intercepted_emitters: set[int],
    missed_opportunity: bool,
    intercept_time_error_us: float,
    w1: float = 5.0,
    w2: float = 8.0,
    w3: float = 0.1,
    w4: float = 4.0,
) -> tuple[float, set[int]]:
    """Calculate shaped reward for one scheduling step.

    Args:
        hit: Whether chosen band had pulses during dwell.
        labels_intercepted: Array of emitter label IDs captured (may include -1).
        intercepted_emitters: Set of emitter IDs already seen this episode (mutated by caller).
        missed_opportunity: Whether any other band had active pulses while receiver was elsewhere.
        intercept_time_error_us: Absolute error (µs) between dwell start and first-pulse ToA.
        w1: Reward per novel emitter.
        w2: Reward per priority threat (reserved for threat classifier, currently unused).
        w3: Penalty per µs timing error (only on hit).
        w4: Penalty for missed opportunity (only on miss).

    Returns:
        Tuple (scalar_reward, set_of_newly_found_emitter_ids).
    """
    reward = 0.0
    new_emitters: set[int] = set()

    if hit:
        # CRITICAL: emitter labels are file-local — caller must never mix across files
        unique_labels = set(int(lbl) for lbl in labels_intercepted if int(lbl) >= 0)
        for lbl in unique_labels:
            if lbl not in intercepted_emitters:
                reward += w1
                new_emitters.add(lbl)
                logger.debug("Novel emitter %d: +%.1f", lbl, w1)
        reward -= w3 * float(intercept_time_error_us)
        if intercept_time_error_us > 0:
            logger.debug("Timing penalty: -%.3f (err=%.1fus)", w3 * intercept_time_error_us, intercept_time_error_us)
    else:
        if missed_opportunity:
            reward -= w4
            logger.debug("Missed opportunity penalty: -%.1f", w4)

    return float(reward), new_emitters


def compute_receiver_reward(
    observation,
    ground_truth_active: bool,
    novel_emitter: bool,
    had_any_opportunity: bool,
    w_hit: float = 1.0,
    w_novel: float = 2.0,
    w_miss: float = -1.0,
    w_timing: float = 0.001,
) -> float:
    """Reward derived from a ReceiverObservation + ground-truth summary.

    The ``observation`` carries ``detections`` (a list of DetectionObservation with
    ``detected=True/False``). Reward is:
        +w_hit                       if any detection
        +w_novel                     if a new emitter was intercepted
        +w_miss                      if there was an opportunity elsewhere but we missed it
        -w_timing * abs(peak_time - dwell_start)   small time-shape penalty on hits

    ``ground_truth_active`` and ``had_any_opportunity`` are evaluation-only signals
    used to shape the "missed opportunity" term. They are NOT part of the scheduler
    observation vector.

    Args:
        observation: ReceiverObservation from SieveReceiver.get_observation().
        ground_truth_active: Whether any emitter was active during this dwell (evaluation only).
        novel_emitter: Whether this dwell intercepted an emitter not seen before.
        had_any_opportunity: Whether any pulse was physically interceptable (elsewhere).
        w_hit: Per-hit reward.
        w_novel: Novel-emitter bonus.
        w_miss: Miss penalty (<=0).
        w_timing: Per-unit timing penalty magnitude.

    Returns:
        Scalar reward.
    """
    detections = getattr(observation, "detections", [])
    n_hits = len(detections)

    reward = 0.0
    novel_bonus = 0.0
    miss_penalty = 0.0

    if n_hits > 0:
        reward += w_hit
        # small timing penalty: how far the first detection is from dwell start
        first_time = getattr(detections[0], "time_us", 0.0)
        dwell = getattr(observation, "dwell_interval_us", [0.0, 0.0])
        start = float(dwell[0]) if len(dwell) >= 1 else 0.0
        reward -= w_timing * abs(first_time - start)
        # Novel bonus only on an actual interception (not mere opportunity)
        if novel_emitter:
            novel_bonus = w_novel
            reward += w_novel
    elif had_any_opportunity:
        # No detection but an opportunity existed elsewhere -> miss penalty (<=0)
        miss_penalty = w_miss
        reward += w_miss

    return float(reward)

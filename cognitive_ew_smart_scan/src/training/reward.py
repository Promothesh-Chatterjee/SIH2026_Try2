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

_LOG2 = np.log(2.0)


def bernoulli_entropy(p: float) -> float:
    """Shannon entropy (bits) of a Bernoulli belief probability ``p``.

    H(p) = -p·log2(p) - (1-p)·log2(1-p), with H(0) = H(1) = 0. This is the
    entropy of the belief that the selected band is active, given occupancy
    probability ``p`` (Phase 10: information gain = H_before - H_after).

    Args:
        p: Belief probability that the band is active, in [0, 1].

    Returns:
        Entropy in bits, >= 0.
    """
    p = float(np.clip(p, 0.0, 1.0))
    if not (0.0 < p < 1.0):
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / _LOG2)


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
    **_extra,
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
        Scalar reward (sum of component terms).
    """
    comps = receiver_reward_components(
        observation,
        ground_truth_active=ground_truth_active,
        novel_emitter=novel_emitter,
        had_any_opportunity=had_any_opportunity,
        w_hit=w_hit,
        w_novel=w_novel,
        w_miss=w_miss,
        w_timing=w_timing,
    )
    return comps["reward"]


def receiver_reward_components(
    observation,
    ground_truth_active: bool,
    novel_emitter: bool,
    had_any_opportunity: bool,
    w_hit: float = 1.0,
    w_novel: float = 2.0,
    w_miss: float = -1.0,
    w_timing: float = 0.001,
    w_priority: float = 0.5,
    w_information_gain: float = 0.2,
    w_false_alarm: float = -0.5,
    w_dwell_cost: float = -0.001,
    w_redundant_scan: float = -0.1,
    w_delay: float = 0.0,
    band: int | None = None,
    belief=None,
    intercepted_emitters: set[int] | None = None,
    novel_ids: set[int] | None = None,
    priority_weight_reference: float = 0.5,
    information_gain: float | None = None,
    entropy_before: float | None = None,
    entropy_after: float | None = None,
) -> dict[str, float]:
    """Per-component reward breakdown (SIH eval contract: log terms separately).

    Full config-driven shaping set (each term auditable via FoM):

      +hit_term          w_hit          if any detection this dwell
      +novel_term        w_novel        if a not-before-seen emitter intercepted
      +timing_penalty   -w_timing*Δt   fast interception is better (reduced dwell lag)
      +priority_term     w_priority*prio  reward for dwelling high-priority bands (observable
                                          priority reference, never GT)
      +info_gain_term    w_information_gain*ΔH  true belief-entropy reduction on the
                                          selected band (IG = H_before - H_after, Phase 10)
      -false_alarm_pen   w_false_alarm*P(fa)   penalise tuning an empty band
      -dwell_cost       -w_dwell_cost*dwell    scan-efficiency cost
      -redundant_pen    -w_redundant_scan      penalty for re-walking a just-intercepted band
      -delay_pen        -w_delay*dwell_lag     penalty for late preemptive intercept on a
                                               high-urgency band (overdue periodic emitter)

    Opportunity semantics (Phase 9): ONLY the selected dwell is a decision-level
    opportunity. ``ground_truth_active`` / ``had_any_opportunity`` here carry the
    **selected-band activity** (not "any band active anywhere"). Thus:
      * selected band active + no detection -> miss_penalty (decision-level FN)
      * selected band inactive + no detection -> false_alarm_pen (tuned an empty band),
        regardless of activity elsewhere (unselected active bands are coverage
        opportunities, NOT decision-level misses and must never emit a miss penalty).

    ``novel_emitter`` / ``novel_ids`` derive from ground-truth emitter IDs and are
    evaluation-only. ``priority_weight_reference`` and ``belief`` are derived from
    scheduler-observable fields only. ``information_gain`` / ``entropy_before`` /
    ``entropy_after`` are computed by the environment around the belief update and
    are scheduler-observable (occupancy belief).

    Returns:
        Dict with reward (total), one key per component term, plus
        entropy_before, entropy_after, information_gain for logging.
    """
    detections = getattr(observation, "detections", [])
    n_hits = len(detections)
    dwell = getattr(observation, "dwell_interval_us", [0.0, 0.0])
    start = float(dwell[0]) if len(dwell) >= 1 else 0.0

    hit_term = 0.0
    timing_penalty = 0.0
    novel_term = 0.0
    miss_penalty = 0.0
    priority_term = 0.0
    info_gain_term = 0.0
    false_alarm_pen = 0.0
    dwell_cost = 0.0
    redundant_pen = 0.0
    delay_pen = 0.0

    hit = n_hits > 0
    novel = bool(novel_emitter or (novel_ids is not None and len(novel_ids) > 0))

    if hit:
        hit_term = w_hit
        first_time = float(getattr(detections[0], "time_us", start))
        timing_penalty = -w_timing * max(0.0, first_time - start)
        if novel:
            novel_term = w_novel
        # Priority term rewards intercepting a high-observable-priority band.
        priority_term = w_priority * float(np.clip(priority_weight_reference, 0.0, 1.0))
        # Redundant-scan penalty when we re-walk a band we just intercepted.
        if band is not None and belief is not None:
            age = int(getattr(belief, "revisit_age", np.zeros(belief.n_bands))[band])
            if age <= 1:
                redundant_pen = w_redundant_scan
        # Delay penalty for overdue high-urgency band (late preemptive intercept).
        if band is not None and belief is not None:
            urgent = float(belief.periodic_urgency[band])
            if urgent > 0.3:
                delay_pen = -abs(w_delay) * urgent
    else:
        # Phase 9: the opportunity signal is about the SELECTED band only.
        #   selected active + undetected  -> FN/miss
        #   selected inactive             -> empty dwell (correct reject / false alarm)
        if had_any_opportunity:
            miss_penalty = w_miss
        elif ground_truth_active is False and float(observation.dwell_time_us) > 0:
            false_alarm_pen = w_false_alarm

    # Dwell cost (scan efficiency): penalise long dwell occupancy.
    dwell_cost = w_dwell_cost * max(0.0, float(observation.dwell_time_us))

    # True information gain (Phase 10): IG = H_before - H_after over the selected
    # band's occupancy belief. Only meaningful when the env supplies it.
    ig = float(information_gain) if information_gain is not None and information_gain == information_gain else 0.0
    info_gain_term = w_information_gain * ig

    total = (
        hit_term + novel_term + timing_penalty + priority_term + info_gain_term
        + false_alarm_pen + dwell_cost + redundant_pen + miss_penalty + delay_pen
    )
    return {
        "reward": float(total),
        "hit_term": float(hit_term),
        "novel_term": float(novel_term),
        "timing_penalty": float(timing_penalty),
        "miss_penalty": float(miss_penalty),
        "priority_term": float(priority_term),
        "info_gain_term": float(info_gain_term),
        "false_alarm_penalty": float(false_alarm_pen),
        "dwell_cost": float(dwell_cost),
        "redundant_penalty": float(redundant_pen),
        "delay_penalty": float(delay_pen),
        "entropy_before": float(entropy_before) if entropy_before is not None else 0.0,
        "entropy_after": float(entropy_after) if entropy_after is not None else 0.0,
        "information_gain": ig,
    }

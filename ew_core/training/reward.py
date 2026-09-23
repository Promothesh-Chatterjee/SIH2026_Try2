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
    w_staleness: float = 0.6,
    staleness_norm: float = 50.0,
    band_age: float | None = None,
    **_extra,
) -> float:
    """Reward derived from a ReceiverObservation + ground-truth summary.

    The ``observation`` carries ``detections`` (a list of DetectionObservation with
    ``detected=True/False``). Reward is:
        +w_hit                       if any detection
        +w_novel                     if a new emitter was intercepted
        +w_miss                      if there was an opportunity elsewhere but we missed it
        +w_staleness * min(age/50,1) intrinsic coverage bonus for visiting cold bands
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
        w_staleness: Intrinsic staleness/coverage bonus weight.
        staleness_norm: Normalization age horizon for revisit age.
        band_age: Dwell band's pre-touch revisit age from belief.

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
        w_staleness=w_staleness,
        staleness_norm=staleness_norm,
        band_age=band_age,
        **_extra,
    )
    return comps["reward"]


def receiver_reward_components(
    observation,
    ground_truth_active: bool = False,
    novel_emitter: bool = False,
    had_any_opportunity: bool = False,
    w_hit: float = 1.0,
    w_novel: float = 2.0,
    w_miss: float = -1.0,
    w_missed_coverage: float = -0.2,
    w_timing: float = 0.001,
    w_priority: float = 0.5,
    w_information_gain: float = 0.2,
    w_false_alarm: float = -0.5,
    w_dwell_cost: float = -0.001,
    w_redundant_scan: float = -0.1,
    w_delay: float = 0.0,
    w_staleness: float = 0.6,
    staleness_norm: float = 50.0,
    band: int | None = None,
    belief=None,
    intercepted_emitters: set[int] | None = None,
    novel_ids: set[int] | None = None,
    priority_weight_reference: float = 0.5,
    information_gain: float | None = None,
    entropy_before: float | None = None,
    entropy_after: float | None = None,
    band_age: float | None = None,
    selected_active: bool | None = None,
    detected: bool | None = None,
    other_bands_active: bool | None = None,
    false_detection: bool | None = None,
    penalize_empty_dwell: bool = False,
    context_scaled_miss: bool = False,
    conditional_dwell_cost: bool = False,
    w_active_track: float = 0.0,
    w_pulse_scale: float = 0.0,
    lull_tolerance: bool = False,
    w_latency: float = 1.0,
    tau_latency: float = 100.0,
    w_prediction: float = 0.5,
    is_predicted: bool = False,
    intercept_time_us: float | None = None,
    disable_latency_reward: bool = False,
) -> dict[str, float]:
    """Per-component reward breakdown implementing 5 explicit decision & coverage signals.

    Signals:
      selected_active: ground truth activity in tuned band
      detected: receiver declared detection in tuned band (n_hits > 0)
      other_bands_active: activity present in unselected spectrum
      false_detection: receiver false declaration on empty band (!selected_active && detected)

    Derived states:
      TP = selected_active && detected           -> hit_term (+ novel_term + latency_bonus + prediction_bonus + active_track_term)
      FN = selected_active && !detected          -> miss_penalty (decision-level miss, subject to lull tolerance)
      FP = !selected_active && detected          -> false_alarm_pen (false detection)
      TN = !selected_active && !detected         -> empty dwell (penalized via false_alarm_pen/dwell_cost)
      coverage_opportunity = !selected_active && other_bands_active -> missed_coverage_pen
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
    staleness_bonus = 0.0
    false_alarm_pen = 0.0
    dwell_cost = 0.0
    redundant_pen = 0.0
    delay_pen = 0.0
    missed_coverage_pen = 0.0
    active_track_term = 0.0
    pulse_bonus_term = 0.0
    latency_bonus_term = 0.0
    prediction_bonus_term = 0.0

    # Derive 5 signals cleanly with backward compatibility
    is_sel_active = bool(selected_active if selected_active is not None else ground_truth_active)
    is_detected = bool(detected if detected is not None else (n_hits > 0))
    is_other_active = bool(other_bands_active if other_bands_active is not None else had_any_opportunity)
    is_false_det = bool(false_detection if false_detection is not None else (not is_sel_active and is_detected))

    novel = bool(novel_emitter or (novel_ids is not None and len(novel_ids) > 0))

    # Determine pre-touch band revisit age for staleness bonus and redundant penalty.
    effective_age = 0.0
    if band_age is not None:
        effective_age = max(0.0, float(band_age))
    elif band is not None and belief is not None:
        rev_age = getattr(belief, "revisit_age", None)
        effective_age = float(rev_age[band]) if rev_age is not None and len(rev_age) > band else 0.0

    # Intrinsic staleness / coverage bonus (rewards exploring unvisited bands).
    if staleness_norm > 0.0 and w_staleness != 0.0:
        staleness_bonus = w_staleness * min(effective_age / staleness_norm, 1.0)

    # 1. True Positive: Hit on an active band
    if is_sel_active and is_detected:
        hit_term = w_hit
        # Pulse-count aware scaling: reward catching multiple pulses in longer dwells
        if w_pulse_scale > 0.0 and n_hits > 1:
            pulse_bonus_term = w_pulse_scale * min(n_hits - 1, 4)
            hit_term += pulse_bonus_term

        first_time = float(getattr(detections[0], "time_us", start)) if n_hits > 0 else start
        timing_penalty = -w_timing * max(0.0, first_time - start)
        if novel:
            novel_term = w_novel
        priority_term = w_priority * float(np.clip(priority_weight_reference, 0.0, 1.0))

        # Productive tracking incentive: reward consecutively confirming hits on an active emitter
        if w_active_track > 0.0 and effective_age <= 2.0 and belief is not None and float(belief.occupancy_prob[band]) >= 0.4:
            active_track_term = w_active_track

        # Stage 3 Step 7: Early-interception reward (w_latency * exp(-t_hit / tau_latency))
        # Strictly conditioned on hit == True. Zero false-early bonus.
        if not disable_latency_reward and w_latency > 0.0:
            if intercept_time_us is not None and np.isfinite(intercept_time_us):
                t_hit = max(0.0, float(intercept_time_us))
            else:
                first_time = float(getattr(detections[0], "time_us", start)) if n_hits > 0 else start
                t_hit = max(0.0, first_time - start)
            latency_bonus_term = float(w_latency * np.exp(-t_hit / max(1.0, float(tau_latency))))

        # Stage 3 Step 7: Prediction bonus for confirmed hit on predicted agile arrival
        if w_prediction > 0.0 and is_predicted:
            prediction_bonus_term = float(w_prediction)

        # Redundant scan penalty: only applies to unconfirmed / empty re-scans, NOT confirmed hits
        if w_redundant_scan != 0.0 and effective_age <= 1.0 and (band_age is not None or (band is not None and belief is not None)):
            # If hit was confirmed, this is productive tracking, not redundant waste
            pass

        if band is not None and belief is not None and hasattr(belief, "periodic_urgency"):
            urgent = float(belief.periodic_urgency[band])
            if urgent > 0.3:
                delay_pen = -abs(w_delay) * urgent

    # 2. False Negative: Decision-level miss (selected band was active, but receiver missed it)
    elif is_sel_active and not is_detected:
        occ = float(belief.occupancy_prob[band]) if (band is not None and belief is not None) else 0.5
        # Inter-pulse lull tolerance: if band is an active track (p_occ >= 0.6) and just visited, don't penalize lull
        if lull_tolerance and effective_age <= 1.0 and occ >= 0.6:
            miss_penalty = 0.0
        elif context_scaled_miss:
            miss_penalty = w_miss * max(0.2, occ)
        else:
            miss_penalty = w_miss

    # 3. Selected band was inactive (FP false alarm or TN empty dwell)
    else:
        # If receiver declared a false detection on an empty band (spurious detection / FP)
        if is_false_det:
            false_alarm_pen = w_false_alarm * 1.5
        elif is_detected:
            false_alarm_pen = w_false_alarm
        elif penalize_empty_dwell and float(observation.dwell_time_us) > 0:
            false_alarm_pen = w_false_alarm

        # Re-scanning an inactive band consecutively is a redundant scan
        if w_redundant_scan != 0.0 and effective_age <= 1.0 and (band_age is not None or (band is not None and belief is not None)):
            redundant_pen = w_redundant_scan

        # Operational coverage opportunity: other bands were active while we tuned an empty band
        if is_other_active:
            missed_coverage_pen = w_missed_coverage

    # Dwell cost: Conditional opportunity cost vs flat cost
    dwell_us = max(0.0, float(observation.dwell_time_us))
    if conditional_dwell_cost:
        # Productive hit -> zero dwell penalty
        if is_sel_active and is_detected:
            dwell_cost = 0.0
        # Likely-active band under tracking (p_occ >= 0.5) -> near-zero cost
        elif band is not None and belief is not None and float(belief.occupancy_prob[band]) >= 0.5:
            dwell_cost = 0.1 * w_dwell_cost * dwell_us
        # Empty / unproductive dwell on cold band -> full opportunity cost
        else:
            dwell_cost = w_dwell_cost * dwell_us
    else:
        dwell_cost = w_dwell_cost * dwell_us

    # True information gain (Phase 10): IG = H_before - H_after over the selected band belief.
    ig = float(information_gain) if information_gain is not None and information_gain == information_gain else 0.0
    info_gain_term = w_information_gain * ig

    total = (
        hit_term + novel_term + timing_penalty + priority_term + info_gain_term
        + staleness_bonus + false_alarm_pen + dwell_cost + redundant_pen + miss_penalty + missed_coverage_pen + delay_pen
        + active_track_term + latency_bonus_term + prediction_bonus_term
    )
    return {
        "reward": float(total),
        "hit_term": float(hit_term),
        "novel_term": float(novel_term),
        "timing_penalty": float(timing_penalty),
        "miss_penalty": float(miss_penalty),
        "missed_coverage_penalty": float(missed_coverage_pen),
        "priority_term": float(priority_term),
        "info_gain_term": float(info_gain_term),
        "staleness_bonus": float(staleness_bonus),
        "false_alarm_penalty": float(false_alarm_pen),
        "dwell_cost": float(dwell_cost),
        "redundant_penalty": float(redundant_pen),
        "delay_penalty": float(delay_pen),
        "active_track_bonus": float(active_track_term),
        "pulse_bonus": float(pulse_bonus_term),
        "latency_bonus": float(latency_bonus_term),
        "prediction_bonus": float(prediction_bonus_term),
        "entropy_before": float(entropy_before) if entropy_before is not None else 0.0,
        "entropy_after": float(entropy_after) if entropy_after is not None else 0.0,
        "information_gain": ig,
    }


def validate_reward_v2_dominance(
    w_hit_repeat: float = 8.0,
    w_latency: float = 5.0,
    w_agile_bonus: float = 2.0,
    w_prediction: float = 0.5,
    w_redundant: float = -0.25,
    w_dwell_cost: float = -0.01,
    dwell_durations_us: tuple[float, ...] = (125.0, 500.0, 1250.0),
) -> bool:
    """Validate that shaping terms can never overpower primary repeat interception reward.

    Enforces:
      max_hit_shaping = abs(w_latency) + abs(w_agile_bonus) + abs(w_prediction) < w_hit_repeat
    across all valid dwell durations.
    """
    max_hit_shaping = abs(w_latency) + abs(w_agile_bonus) + abs(w_prediction)
    if max_hit_shaping >= w_hit_repeat:
        raise ValueError(
            f"Reward v2 dominance violated: max_hit_shaping ({max_hit_shaping:.2f}) "
            f">= w_hit_repeat ({w_hit_repeat:.2f})"
        )
    for dwell_us in dwell_durations_us:
        cost = abs(w_dwell_cost) * (dwell_us / 500.0)
        total_shaping = max_hit_shaping + abs(w_redundant) + cost
        if total_shaping >= (w_hit_repeat + 1.0):  # margin check
            raise ValueError(
                f"Reward v2 dominance violated for dwell {dwell_us}us: "
                f"total_shaping ({total_shaping:.2f}) >= w_hit_repeat + margin"
            )
    return True


def receiver_reward_components_v2(
    observation=None,
    ground_truth_active: bool = False,
    novel_emitter: bool = False,
    had_any_opportunity: bool = False,
    selected_active: bool | None = None,
    detected: bool | None = None,
    other_bands_active: bool | None = None,
    false_detection: bool | None = None,
    intercept_time_us: float | None = None,
    is_agile: bool = False,
    is_predicted: bool = False,
    band_age: float | None = None,
    band: int | None = None,
    belief=None,
    # Clean v2 hierarchical weights
    w_hit_novel: float = 10.0,
    w_hit_repeat: float = 8.0,
    w_latency: float = 5.0,
    w_agile_bonus: float = 2.0,
    w_prediction: float = 0.5,
    w_miss: float = -4.0,
    w_false_alarm: float = -1.0,
    w_redundant: float = -0.25,
    w_dwell_cost: float = -0.01,
    lambda_pfa: float = 2.0,      # Lagrangian multiplier for Pfa constraint
    pfa_threshold: float = 0.05,  # Maximum allowable Pfa (5%)
    running_pfa: float = 0.0,     # Rolling Pfa estimate from the episode so far
    reward_variant: str = "baseline",
    strict_dominance: bool = False,
    **_extra,
) -> dict[str, float | bool]:
    """Phase 4 Clean Rescue Reward (reward_v2) with Phase 9C configurable variants.

    Hierarchical design aligned directly with operational EW smart scan objectives:
      1. Primary: Successful Interception (+10.0 novel, +8.0 repeat)
      2. Secondary: Early Interception latency bonus (0.0 to +5.0)
      3. Agile bonus: +2.0 for intercepting agile/hopping emitters
      4. Prediction bonus: +0.5 for intercepting a predicted arrival
      5. Miss penalty: -4.0 for failing to intercept active emitter
      6. False alarm penalty: -1.0 for tuning inactive spectrum
      7. Redundant revisit: -0.25 for immediate re-visit of empty spectrum
      8. Dwell cost: -0.01 * (dwell_us / 500.0) [baseline] OR flat -0.01 [dwell_cost_normalized]

    Variants:
      - "baseline": standard reward_v2.
      - "dwell_cost_normalized": dwell_norm is fixed to 1.0, eliminating the 2.5x penalty on unintercepted long dwells.
      - "time_normalized": scales the total reward by (500.0 / dwell_us) to represent reward per standard time slot.
    """
    is_sel_active = bool(selected_active if selected_active is not None else ground_truth_active)
    detections = getattr(observation, "detections", []) if observation is not None else []
    is_detected = bool(detected if detected is not None else (len(detections) > 0))
    dwell_us = max(0.0, float(getattr(observation, "dwell_time_us", 500.0))) if observation is not None else 500.0

    if reward_variant == "dwell_cost_normalized":
        dwell_norm = 1.0
    else:
        dwell_norm = dwell_us / 500.0 if dwell_us > 0.0 else 1.0

    interception_reward = 0.0
    latency_reward = 0.0
    agility_bonus = 0.0
    prediction_bonus = 0.0
    miss_penalty = 0.0
    false_alarm_pen = 0.0
    redundant_pen = 0.0
    dwell_cost = 0.0

    if is_sel_active and is_detected:
        # 1. Successful Interception (Primary Dominant Signal)
        interception_reward = float(w_hit_novel if novel_emitter else w_hit_repeat)

        # 2. Earlier Interception Latency Bonus (Secondary Signal in [0, w_latency])
        if intercept_time_us is not None and np.isfinite(intercept_time_us):
            t_hit = max(0.0, float(intercept_time_us))
        else:
            dwell = getattr(observation, "dwell_interval_us", [0.0, 0.0]) if observation is not None else [0.0, 0.0]
            start = float(dwell[0]) if len(dwell) >= 1 else 0.0
            first_time = float(getattr(detections[0], "time_us", start)) if detections else start
            t_hit = max(0.0, first_time - start)
        latency_fraction = max(0.0, 1.0 - (t_hit / max(1.0, dwell_us)))
        latency_reward = float(w_latency * latency_fraction)

        # 3. Frequency-Agile Interception Bonus
        if is_agile:
            agility_bonus = float(w_agile_bonus)

        # 4. Predicted Arrival Interception Bonus
        if is_predicted:
            prediction_bonus = float(w_prediction)

        # Dwell cost on confirmed hit is 0 (hits are productive; dwell cost never penalizes detection)
        dwell_cost = 0.0

    elif is_sel_active and not is_detected:
        # 5. Missed Active Emitter
        miss_penalty = float(w_miss)
        dwell_cost = float(w_dwell_cost * dwell_norm)

    else:
        # 6. Inactive Band (False Alarm / Empty Dwell)
        false_alarm_pen = float(w_false_alarm)
        effective_age = float(band_age) if band_age is not None else (
            float(belief.revisit_age[band]) if (band is not None and belief is not None and hasattr(belief, "revisit_age")) else 10.0
        )
        if effective_age <= 1.0:
            redundant_pen = float(w_redundant)
        dwell_cost = float(w_dwell_cost * dwell_norm)

    # Missed-coverage penalty: fires when we visited an empty/wrong band
    # while active emitters existed elsewhere in the spectrum.
    # This is the key credit-assignment fix for the miss_rate gradient problem.
    missed_coverage_penalty = 0.0
    if not is_sel_active and other_bands_active:
        # How many active opportunities were missed?
        n_missed = max(0, int(getattr(belief, 'n_bands', 36)) - 1) if belief else 1
        opportunity_scale = min(1.0, float(n_missed) / 8.0)  # saturate at 8+ active bands
        missed_coverage_penalty = float(w_miss) * 0.25 * opportunity_scale
        # Scale by belief miss_rate of chosen band: if we had high miss_rate here,
        # penalise more (we should have known this band was poor)
        if belief is not None and hasattr(belief, 'detection_rate') and band is not None and 0 <= band < len(belief.detection_rate):
            band_miss_rate = 1.0 - float(belief.detection_rate[band])
            missed_coverage_penalty *= (0.5 + 0.5 * band_miss_rate)

    # Lagrangian Pfa Constraint (Audit Item 10)
    # Penalises the policy whenever the running false alarm rate exceeds the target.
    # Soft constraint: progressive penalty when running_pfa > pfa_threshold.
    lagrangian_pfa_penalty = 0.0
    if running_pfa > pfa_threshold:
        pfa_excess = running_pfa - pfa_threshold
        lagrangian_pfa_penalty = -lambda_pfa * pfa_excess
        # Cap penalty: never larger than 50% of the primary miss penalty
        lagrangian_pfa_penalty = max(lagrangian_pfa_penalty, w_miss * 0.5)

    total = (
        interception_reward
        + latency_reward
        + agility_bonus
        + prediction_bonus
        + miss_penalty
        + missed_coverage_penalty
        + false_alarm_pen
        + redundant_pen
        + dwell_cost
        + lagrangian_pfa_penalty
    )

    if reward_variant == "time_normalized" and dwell_us > 0.0:
        total = total * (500.0 / dwell_us)

    # Dominance Check: detect if secondary shaping exceeds primary interception signal
    shaping_mag = abs(latency_reward) + abs(agility_bonus) + abs(prediction_bonus) + abs(redundant_pen) + abs(dwell_cost) + abs(lagrangian_pfa_penalty)
    dominance_violated = bool(is_sel_active and is_detected and (shaping_mag >= abs(interception_reward)))
    if dominance_violated:
        if strict_dominance:
            raise ValueError(
                f"REWARD_OBJECTIVE_DOMINANCE_VIOLATION: shaping terms ({shaping_mag:.2f}) "
                f">= primary interception reward ({interception_reward:.2f})"
            )
        logger.warning(
            "REWARD_OBJECTIVE_DOMINANCE_WARNING: shaping terms (%.2f) exceed primary interception reward (%.2f)",
            shaping_mag,
            interception_reward,
        )

    # Diagnostic per-ms telemetry calculations
    dwell_ms = max(1e-6, dwell_us / 1000.0)
    reward_per_ms = float(total / dwell_ms)
    hit_reward_total = float(interception_reward + latency_reward + agility_bonus + prediction_bonus)
    hit_reward_per_ms = float(hit_reward_total / dwell_ms)
    # Preserved signed negative penalty so: reward_per_ms = hit_reward_per_ms + penalty_per_ms (or 0)
    penalty_total = float(miss_penalty + missed_coverage_penalty + false_alarm_pen + redundant_pen + dwell_cost + lagrangian_pfa_penalty)
    penalty_per_ms = float(penalty_total / dwell_ms)

    return {
        "reward": float(total),
        "interception_reward": float(interception_reward),
        "latency_reward": float(latency_reward),
        "agility_bonus": float(agility_bonus),
        "prediction_bonus": float(prediction_bonus),
        "frequency_agility_bonus": float(agility_bonus),
        "miss_penalty": float(miss_penalty),
        "missed_coverage_penalty": float(missed_coverage_penalty),
        "false_alarm_penalty": float(false_alarm_pen),
        "redundant_penalty": float(redundant_pen),
        "dwell_cost": float(dwell_cost),
        "lagrangian_pfa_penalty": float(lagrangian_pfa_penalty),
        "running_pfa": float(running_pfa),
        "pfa_constraint_active": bool(running_pfa > pfa_threshold),
        "dominance_warning": dominance_violated,
        "reward_component_dominance": dominance_violated,
        # Time-aware diagnostic telemetry fields
        "dwell_time_us": float(dwell_us),
        "reward_per_dwell": float(total),
        "reward_per_ms": reward_per_ms,
        "hit_reward_per_ms": hit_reward_per_ms,
        "penalty_per_ms": penalty_per_ms,
        # Backward-compatible aliases for legacy FiguresOfMerit accumulators
        "hit_term": float(interception_reward),
        "novel_term": float(w_hit_novel - w_hit_repeat if novel_emitter else 0.0),
        "timing_penalty": 0.0,
        "missed_coverage_penalty": float(missed_coverage_penalty),
        "priority_term": 0.0,
        "info_gain_term": 0.0,
        "staleness_bonus": 0.0,
        "active_track_bonus": 0.0,
        "pulse_bonus": 0.0,
        "latency_bonus": float(latency_reward),
        "delay_penalty": 0.0,
        "entropy_before": 0.0,
        "entropy_after": 0.0,
        "information_gain": 0.0,
    }

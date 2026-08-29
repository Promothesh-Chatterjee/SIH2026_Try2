"""
Domain-specific Reward Function for the RF Scanning Scheduler.

Computes shaped rewards based on intercept quality, novelty, timing accuracy,
and penalty for missed opportunities.
"""

import numpy as np


def compute_reward(
    hit: bool,
    labels_intercepted: np.ndarray,
    intercepted_emitters: set,
    missed_opportunity: bool,
    intercept_time_error_us: float,
    w1: float = 5.0,
    w2: float = 8.0,
    w3: float = 0.1,
    w4: float = 4.0,
) -> tuple[float, set]:
    """
    Calculates the shaped reward for a single scheduling step.

    Reward components:
        +w1 per novel emitter intercepted (never seen this episode)
        +w2 per high-priority emitter intercepted (future: threat classification)
        -w3 * intercept_time_error_us for timing inaccuracy
        -w4 if a non-empty band was missed entirely

    Args:
        hit: Whether the chosen band contained any pulses during the dwell.
        labels_intercepted: Array of emitter label IDs captured this dwell.
        intercepted_emitters: Set of emitter IDs already seen this episode.
        missed_opportunity: Whether any other band had active pulses while
                            the receiver was tuned to the chosen band.
        intercept_time_error_us: Absolute error (µs) between predicted and
                                 actual first-pulse ToA in the band.
        w1: Reward weight for novel emitter discovery.
        w2: Reward weight for priority emitter (stubbed; extends to threat model).
        w3: Penalty weight per µs of intercept timing error.
        w4: Penalty weight for a missed opportunity.

    Returns:
        Tuple of (scalar_reward, set_of_newly_found_emitter_ids).
    """
    reward = 0.0
    new_emitters: set = set()

    if hit:
        unique_labels = set(int(lbl) for lbl in labels_intercepted if lbl >= 0)
        for lbl in unique_labels:
            if lbl not in intercepted_emitters:
                reward += w1
                new_emitters.add(lbl)
            # Placeholder for priority threat bonus: extend when classifier is active
            # if threat_classifier.is_priority(lbl):
            #     reward += w2

        # Timing accuracy penalty: only charged on a hit
        reward -= w3 * intercept_time_error_us
    else:
        # Penalise tuning to an empty band when something was happening elsewhere
        if missed_opportunity:
            reward -= w4

    return reward, new_emitters

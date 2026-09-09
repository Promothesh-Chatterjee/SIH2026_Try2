"""
Adaptive Behavior Manager (ABM) for Cognitive EW Smart Scan.

Classifies emitter tracks into operational behavior categories and provides
behavior-specific scheduling policies, dwell mode recommendations, and
expected utility parameters without modifying frozen neural network weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.contracts import (
    NORMAL_DWELL,
    SHORT_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    CANONICAL_N_BANDS,
)


class EmitterBehavior(str, Enum):
    """Behavioral classification taxonomy for emitter tracks."""
    UNKNOWN = "UNKNOWN"
    FIXED = "FIXED"
    PERIODIC = "PERIODIC"
    CYCLIC_HOPPER = "CYCLIC_HOPPER"
    MARKOV_HOPPER = "MARKOV_HOPPER"
    FAST_HOPPER = "FAST_HOPPER"
    SLOW_HOPPER = "SLOW_HOPPER"
    BURSTY_JITTERED = "BURSTY_JITTERED"


@dataclass
class BehaviorProfile:
    """Behavior classification and scheduling policy recommendations for a track."""
    track_id: int
    behavior: EmitterBehavior
    confidence: float
    recommended_mode: int
    use_expected_utility: bool
    requires_reservation: bool
    entropy: float
    pri_us: float
    distinct_bands: int
    agility_score: float
    reason: str


class AdaptiveBehaviorManager:
    """Classifies emitter tracks and provides adaptive operational scheduling directives."""

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        fast_pri_threshold_us: float = 220.0,
        slow_pri_threshold_us: float = 500.0,
        jitter_threshold: float = 0.12,
        markov_entropy_threshold: float = 0.65,
    ) -> None:
        self.n_bands: int = n_bands
        self.fast_pri_threshold_us: float = float(fast_pri_threshold_us)
        self.slow_pri_threshold_us: float = float(slow_pri_threshold_us)
        self.jitter_threshold: float = float(jitter_threshold)
        self.markov_entropy_threshold: float = float(markov_entropy_threshold)

        self.track_profiles: Dict[int, BehaviorProfile] = {}

    def reset(self) -> None:
        """Clear all track behavioral history."""
        self.track_profiles.clear()

    def classify_track(self, track_state: Any) -> BehaviorProfile:
        """Classify track behavioral profile from TrackTemporalState.

        Args:
            track_state: TrackTemporalState instance with observation history.

        Returns:
            BehaviorProfile with behavior class and scheduling guidance.
        """
        track_id = int(track_state.track_id)
        n_pulses = len(track_state.toa_history)
        pri = float(track_state.pri_estimate)
        pri_var = float(track_state.pri_variance)
        rel_jitter = math.sqrt(max(0.0, pri_var)) / max(1.0, pri) if pri > 0.0 else 0.0

        distinct_bands = int(np.count_nonzero(track_state.band_counts))
        agility = float(getattr(track_state, "agility_score", 0.0))

        # Insufficient data
        if n_pulses < 3 or pri <= 0.0:
            profile = BehaviorProfile(
                track_id=track_id,
                behavior=EmitterBehavior.UNKNOWN,
                confidence=0.1,
                recommended_mode=NORMAL_DWELL,
                use_expected_utility=False,
                requires_reservation=False,
                entropy=0.0,
                pri_us=pri,
                distinct_bands=distinct_bands,
                agility_score=agility,
                reason="insufficient_pulses",
            )
            self.track_profiles[track_id] = profile
            return profile

        # Calculate transition entropy if multiple bands observed
        entropy = 0.0
        if distinct_bands > 1:
            total_transitions = sum(
                sum(next_map.values()) for next_map in track_state.unigram_counts.values()
            )
            if total_transitions > 0:
                ent_sum = 0.0
                for next_map in track_state.unigram_counts.values():
                    map_total = sum(next_map.values())
                    if map_total > 0:
                        for count in next_map.values():
                            p = count / map_total
                            if p > 0:
                                ent_sum -= p * math.log2(p)
                entropy = ent_sum / max(1, len(track_state.unigram_counts))

        # 1. Bursty / Jittered: High timing variation
        if rel_jitter > self.jitter_threshold and n_pulses >= 5:
            behavior = EmitterBehavior.BURSTY_JITTERED
            recommended_mode = LONG_DWELL  # Longer look to absorb jitter
            use_eu = False
            req_res = False
            conf = min(0.9, 0.5 + rel_jitter)
            reason = f"high_jitter_{rel_jitter:.2f}"

        # 2. Fixed Carrier: Exactly 1 band, zero agility
        elif distinct_bands <= 1 and agility < 0.1:
            behavior = EmitterBehavior.FIXED
            recommended_mode = NORMAL_DWELL
            use_eu = False
            req_res = False
            conf = min(1.0, 0.4 + 0.1 * n_pulses)
            reason = "single_carrier_zero_agility"

        # 3. Markov Stochastic Hopper: High transition entropy across carrier set
        elif agility >= 0.25 and (entropy > self.markov_entropy_threshold or len(track_state.bigram_counts) > 4):
            behavior = EmitterBehavior.MARKOV_HOPPER
            # If PRI is fast (<=220 us), use SHORT_DWELL (125 us) to avoid lock-out, else NORMAL_DWELL
            recommended_mode = SHORT_DWELL if pri <= self.fast_pri_threshold_us else NORMAL_DWELL
            use_eu = True  # True stochastic expected utility over all next states
            req_res = False
            conf = min(0.90, 0.5 + 0.2 * entropy)
            reason = f"stochastic_transitions_entropy_{entropy:.2f}"

        # 4. Slow Agile Hopper: PRI > base dwell (500 µs)
        elif agility >= 0.2 and pri >= self.slow_pri_threshold_us:
            behavior = EmitterBehavior.SLOW_HOPPER
            recommended_mode = NORMAL_DWELL
            use_eu = False
            req_res = True  # Must use temporal reservation
            conf = min(0.95, 0.6 + 0.05 * n_pulses)
            reason = f"slow_pri_{pri:.1f}us_agile"

        # 5. Fast Agile Hopper: PRI <= fast threshold (220 µs)
        elif agility >= 0.2 and pri <= self.fast_pri_threshold_us:
            behavior = EmitterBehavior.FAST_HOPPER
            recommended_mode = SHORT_DWELL  # 125 µs canonical dwell to match rapid cycle
            use_eu = False
            req_res = False
            conf = min(0.95, 0.6 + 0.05 * n_pulses)
            reason = f"fast_pri_{pri:.1f}us_short_dwell"

        # 6. Cyclic Deterministic Hopper: Repeating permutation of carrier bands
        elif agility >= 0.25 and distinct_bands >= 2:
            behavior = EmitterBehavior.CYCLIC_HOPPER
            # If PRI is below normal dwell, recommend SHORT_DWELL (125 µs), otherwise NORMAL
            recommended_mode = SHORT_DWELL if pri < 400.0 else NORMAL_DWELL
            use_eu = False
            req_res = False
            conf = min(0.95, 0.5 + 0.1 * distinct_bands)
            reason = f"cyclic_permutation_{distinct_bands}_bands"

        # 7. Stable Periodic Carrier
        elif rel_jitter < 0.05 and n_pulses >= 4:
            behavior = EmitterBehavior.PERIODIC
            recommended_mode = NORMAL_DWELL
            use_eu = False
            req_res = (pri > 400.0)
            conf = 0.85
            reason = "stable_pri_periodic"

        else:
            behavior = EmitterBehavior.UNKNOWN
            recommended_mode = NORMAL_DWELL
            use_eu = False
            req_res = False
            conf = 0.4
            reason = "unclassified_default"

        profile = BehaviorProfile(
            track_id=track_id,
            behavior=behavior,
            confidence=float(conf),
            recommended_mode=recommended_mode,
            use_expected_utility=use_eu,
            requires_reservation=req_res,
            entropy=float(entropy),
            pri_us=float(pri),
            distinct_bands=distinct_bands,
            agility_score=float(agility),
            reason=reason,
        )
        self.track_profiles[track_id] = profile
        return profile

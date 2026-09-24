"""Reward dominance unit tests.

Enforces the operational invariant that successful RF interception ALWAYS dominates
all penalties (miss penalty, missed-coverage penalty, false alarm penalty, dwell cost,
redundant revisit penalty, and Lagrangian Pfa constraint penalties) under all valid
configurations and dwell durations.
"""

import itertools
import unittest
from types import SimpleNamespace
from ew_core.training.reward import receiver_reward_components_v2


class TestRewardDominance(unittest.TestCase):
    """Rigorous qualification tests verifying primary interception reward dominance."""

    def test_hit_dominates_miss(self):
        """Interception reward must strictly dominate miss penalty across all flag combinations."""
        boolean_options = [False, True]
        novel_options = [False, True]
        agile_options = [False, True]
        pred_options = [False, True]
        other_active_options = [False, True]
        dwell_durations = [125.0, 500.0, 1250.0]

        for dwell_us in dwell_durations:
            dummy_obs = SimpleNamespace(dwell_time_us=dwell_us, detections=[SimpleNamespace(time_us=0.0)])
            dummy_obs_miss = SimpleNamespace(dwell_time_us=dwell_us, detections=[])

            for novel, agile, pred, other_act in itertools.product(
                novel_options, agile_options, pred_options, other_active_options
            ):
                r_hit = receiver_reward_components_v2(
                    observation=dummy_obs,
                    hit=True,
                    selected_active=True,
                    novel_emitter=novel,
                    is_agile=agile,
                    is_predicted=pred,
                    other_bands_active=other_act,
                    running_pfa=0.0,
                    pfa_threshold=0.05,
                )
                r_miss = receiver_reward_components_v2(
                    observation=dummy_obs_miss,
                    hit=False,
                    selected_active=True,
                    novel_emitter=novel,
                    is_agile=agile,
                    is_predicted=pred,
                    other_bands_active=other_act,
                    running_pfa=0.0,
                    pfa_threshold=0.05,
                )
                self.assertGreater(
                    r_hit["total"],
                    r_miss["total"],
                    f"Dominance failed for dwell={dwell_us}, novel={novel}, agile={agile}, pred={pred}: "
                    f"hit={r_hit['total']} <= miss={r_miss['total']}",
                )

    def test_hit_dominates_empty_band(self):
        """Interception reward must strictly dominate empty band (false alarm) dwell across durations."""
        dwell_durations = [125.0, 500.0, 1250.0]
        other_active_options = [False, True]

        for dwell_us in dwell_durations:
            dummy_obs_hit = SimpleNamespace(dwell_time_us=dwell_us, detections=[SimpleNamespace(time_us=0.0)])
            dummy_obs_empty = SimpleNamespace(dwell_time_us=dwell_us, detections=[])

            for other_act in other_active_options:
                r_hit = receiver_reward_components_v2(
                    observation=dummy_obs_hit,
                    hit=True,
                    selected_active=True,
                    other_bands_active=other_act,
                    running_pfa=0.0,
                )
                r_empty = receiver_reward_components_v2(
                    observation=dummy_obs_empty,
                    hit=False,
                    selected_active=False,
                    other_bands_active=other_act,
                    running_pfa=0.0,
                )
                self.assertGreater(
                    r_hit["total"],
                    r_empty["total"],
                    f"Interception ({r_hit['total']}) does not dominate empty band ({r_empty['total']}) at dwell={dwell_us}us",
                )

    def test_hit_is_positive(self):
        """Interception reward must be strictly positive under all valid operational conditions."""
        dwell_durations = [125.0, 500.0, 1250.0]
        novel_options = [False, True]
        pfa_rates = [0.0, 0.05, 0.10, 0.50, 1.0]

        for dwell_us in dwell_durations:
            dummy_obs = SimpleNamespace(dwell_time_us=dwell_us, detections=[SimpleNamespace(time_us=0.0)])
            for novel in novel_options:
                for pfa in pfa_rates:
                    r_hit = receiver_reward_components_v2(
                        observation=dummy_obs,
                        hit=True,
                        selected_active=True,
                        novel_emitter=novel,
                        running_pfa=pfa,
                        lambda_pfa=2.0,
                        pfa_threshold=0.05,
                    )
                    self.assertGreater(
                        r_hit["total"],
                        0.0,
                        f"Hit reward {r_hit['total']} is not positive for dwell={dwell_us}, novel={novel}, pfa={pfa}",
                    )

    def test_pfa_penalty_does_not_invert_dominance(self):
        """Even under maximum Pfa penalty excess, interception reward must dominate miss penalty."""
        excess_pfas = [0.10, 0.20, 0.50, 1.0]

        for pfa in excess_pfas:
            r_hit = receiver_reward_components_v2(
                hit=True,
                selected_active=True,
                running_pfa=pfa,
                lambda_pfa=2.0,
                pfa_threshold=0.05,
            )
            r_miss = receiver_reward_components_v2(
                hit=False,
                selected_active=True,
                running_pfa=pfa,
                lambda_pfa=2.0,
                pfa_threshold=0.05,
            )
            self.assertGreater(
                r_hit["total"],
                r_miss["total"],
                f"Pfa excess ({pfa}) inverted dominance: hit={r_hit['total']}, miss={r_miss['total']}",
            )


if __name__ == "__main__":
    unittest.main()

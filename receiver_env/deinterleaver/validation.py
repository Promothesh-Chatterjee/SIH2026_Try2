"""Phase 4 Ground-Truth Metrics and Validation Framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
from receiver_env.deinterleaver.models import EmitterTrack


@dataclass(frozen=True)
class DeinterleavingMetrics:
    """Quantitative performance metrics comparing separated tracks against ground truth."""
    track_purity: float
    track_completeness: float
    false_assignment_rate: float
    track_fragmentation: int
    duplicate_assignment_count: int
    track_swap_rate: float


class ValidationFramework:
    """Computes verification metrics given ground truth pulse emitter associations."""

    @staticmethod
    def evaluate(
        tracks: List[EmitterTrack],
        ground_truth_map: Dict[int, str],  # pdw_id -> emitter_label (e.g. "EMITTER_A")
    ) -> Dict[str, DeinterleavingMetrics]:
        """Compute metrics for each identified emitter track.

        Parameters
        ----------
        tracks : List[EmitterTrack]
            List of generated emitter tracks.
        ground_truth_map : Dict[int, str]
            True emitter label for each generated PDW ID.

        Returns
        -------
        Dict[str, DeinterleavingMetrics]
            Metrics keyed by emitter label.
        """
        # Invert ground truth: emitter_label -> Set[pdw_id]
        true_emitter_pdws: Dict[str, Set[int]] = {}
        for pdw_id, label in ground_truth_map.items():
            if label not in true_emitter_pdws:
                true_emitter_pdws[label] = set()
            true_emitter_pdws[label].add(pdw_id)

        # Check for duplicate assignments across tracks
        seen_pdws: Set[int] = set()
        duplicate_count = 0
        for t in tracks:
            for pid in t.assigned_pdw_ids:
                if pid in seen_pdws:
                    duplicate_count += 1
                seen_pdws.add(pid)

        results: Dict[str, DeinterleavingMetrics] = {}

        for emitter_label, true_pids in true_emitter_pdws.items():
            total_true = len(true_pids)
            if total_true == 0:
                continue

            # Find matching tracks having pulses from this emitter
            matching_tracks = []
            for t in tracks:
                overlap = t.assigned_pdw_ids.intersection(true_pids)
                if len(overlap) > 0:
                    matching_tracks.append((t, overlap))

            if not matching_tracks:
                results[emitter_label] = DeinterleavingMetrics(
                    track_purity=0.0,
                    track_completeness=0.0,
                    false_assignment_rate=1.0,
                    track_fragmentation=0,
                    duplicate_assignment_count=duplicate_count,
                    track_swap_rate=0.0,
                )
                continue

            # Primary track is the one with the maximum overlap
            matching_tracks.sort(key=lambda x: len(x[1]), reverse=True)
            primary_track, primary_overlap = matching_tracks[0]

            recovered_pulses = len(primary_overlap)
            total_pulses_in_track = len(primary_track.assigned_pdw_ids)

            purity = recovered_pulses / max(1, total_pulses_in_track)
            completeness = recovered_pulses / max(1, total_true)
            false_assignment = 1.0 - purity
            fragmentation = len(matching_tracks) - 1

            # Check track swap rate (pulses from other emitters in primary track)
            other_pulses = total_pulses_in_track - recovered_pulses
            swap_rate = other_pulses / max(1, total_pulses_in_track)

            results[emitter_label] = DeinterleavingMetrics(
                track_purity=float(purity),
                track_completeness=float(completeness),
                false_assignment_rate=float(false_assignment),
                track_fragmentation=max(0, fragmentation),
                duplicate_assignment_count=duplicate_count,
                track_swap_rate=float(swap_rate),
            )

        return results

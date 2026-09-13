"""Track Integrity Analyzer: Swaps, Fragmentation, and False Merges."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
from receiver_env.deinterleaver.models import EmitterTrack


@dataclass
class TrackIntegrityReport:
    """Comprehensive track integrity audit report."""
    total_emitters: int
    total_tracks: int
    swap_count: int
    swap_rate: float
    fragmentation_count: int
    fragmentation_rate: float
    false_merge_count: int
    false_merge_rate: float
    details_by_emitter: Dict[str, Dict]


class TrackIntegrityAnalyzer:
    """Analyzes track assignments for identity swaps, fragmentation, and false merges."""

    @staticmethod
    def analyze(
        tracks: List[EmitterTrack],
        ground_truth_map: Dict[int, str],  # pdw_id -> emitter_label
    ) -> TrackIntegrityReport:
        # Map each emitter to its set of true PDW IDs
        emitter_true_pdws: Dict[str, Set[int]] = {}
        for pid, label in ground_truth_map.items():
            if label not in emitter_true_pdws:
                emitter_true_pdws[label] = set()
            emitter_true_pdws[label].add(pid)

        total_emitters = len(emitter_true_pdws)
        total_tracks = len(tracks)

        # Map each track to the distribution of true emitters it captured
        track_emitter_distribution: Dict[int, Dict[str, int]] = {}
        for t in tracks:
            track_emitter_distribution[t.track_id] = {}
            for pid in t.assigned_pdw_ids:
                label = ground_truth_map.get(pid, "UNKNOWN")
                track_emitter_distribution[t.track_id][label] = (
                    track_emitter_distribution[t.track_id].get(label, 0) + 1
                )

        # 1. Detect False Merges: A track captures pulses from more than one known emitter (> 10% mix)
        false_merge_count = 0
        for tid, dist in track_emitter_distribution.items():
            valid_emitters = [em for em in dist.keys() if em != "UNKNOWN"]
            if len(valid_emitters) > 1:
                # check if secondary emitter has significant representation
                total_p = sum(dist.values())
                secondary = sorted(dist.values())[-2]
                if secondary / total_p >= 0.05:
                    false_merge_count += 1

        false_merge_rate = false_merge_count / max(1, total_tracks)

        # 2. Detect Fragmentation & Swaps by Emitter
        total_fragmentation = 0
        total_swaps = 0
        emitter_details = {}

        for em_name, true_pids in emitter_true_pdws.items():
            # Tracks that contain at least one pulse from this emitter
            capturing_tracks: List[Tuple[int, int]] = []
            for t in tracks:
                overlap = len(t.assigned_pdw_ids.intersection(true_pids))
                if overlap > 0:
                    capturing_tracks.append((t.track_id, overlap))

            capturing_tracks.sort(key=lambda x: x[1], reverse=True)
            frag_for_em = max(0, len(capturing_tracks) - 1)
            total_fragmentation += frag_for_em

            # Track Swaps: Chronological sequence of assigned tracks for this emitter's pulses
            # Check if an emitter was in track A, hopped to track B, then back to track A
            swaps_for_em = 0
            if len(capturing_tracks) > 1:
                # Find track assignments ordered by ToA
                pdw_track_map = {}
                for t in tracks:
                    for pid in t.assigned_pdw_ids:
                        if pid in true_pids:
                            pdw_track_map[pid] = t.track_id
                
                sorted_pids = sorted(list(true_pids))
                track_seq = [pdw_track_map[pid] for pid in sorted_pids if pid in pdw_track_map]
                for idx in range(1, len(track_seq)):
                    if track_seq[idx] != track_seq[idx - 1]:
                        swaps_for_em += 1

            total_swaps += swaps_for_em

            emitter_details[em_name] = {
                "capturing_tracks": [t[0] for t in capturing_tracks],
                "pulse_counts": [t[1] for t in capturing_tracks],
                "fragmentation": frag_for_em,
                "swaps": swaps_for_em,
            }

        total_pulses = len(ground_truth_map)
        swap_rate = total_swaps / max(1, total_pulses)
        fragmentation_rate = total_fragmentation / max(1, total_emitters)

        return TrackIntegrityReport(
            total_emitters=total_emitters,
            total_tracks=total_tracks,
            swap_count=total_swaps,
            swap_rate=round(swap_rate, 4),
            fragmentation_count=total_fragmentation,
            fragmentation_rate=round(fragmentation_rate, 4),
            false_merge_count=false_merge_count,
            false_merge_rate=round(false_merge_rate, 4),
            details_by_emitter=emitter_details,
        )

    @staticmethod
    def export_json(report: TrackIntegrityReport, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

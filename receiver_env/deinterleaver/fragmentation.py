"""Track Fragmentation Auditor and Report Generator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set
from receiver_env.deinterleaver.models import EmitterTrack


@dataclass
class EmitterFragmentation:
    """Fragmentation assessment for a single ground truth emitter."""
    emitter_name: str
    associated_tracks: List[str]
    pulse_count: int
    fragmentation_count: int


@dataclass
class FragmentationReport:
    """Full fragmentation audit report across all emitters."""
    emitters: List[EmitterFragmentation]
    total_fragmentation: int
    is_acceptable: bool


class FragmentationAuditor:
    """Computes track fragmentation metrics against ground truth."""

    @staticmethod
    def audit(
        tracks: List[EmitterTrack],
        ground_truth_map: Dict[int, str],
    ) -> FragmentationReport:
        emitter_pids: Dict[str, Set[int]] = {}
        for pid, label in ground_truth_map.items():
            emitter_pids.setdefault(label, set()).add(pid)

        records: List[EmitterFragmentation] = []
        total_frag = 0

        for em_name, true_pids in emitter_pids.items():
            capturing = []
            for t in tracks:
                overlap = t.assigned_pdw_ids.intersection(true_pids)
                if len(overlap) > 0:
                    capturing.append((t.emitter_id, len(overlap)))

            capturing.sort(key=lambda x: x[1], reverse=True)
            track_ids = [c[0] for c in capturing]
            frag_count = max(0, len(track_ids) - 1)
            total_frag += frag_count

            records.append(
                EmitterFragmentation(
                    emitter_name=em_name,
                    associated_tracks=track_ids,
                    pulse_count=len(true_pids),
                    fragmentation_count=frag_count,
                )
            )

        return FragmentationReport(
            emitters=records,
            total_fragmentation=total_frag,
            is_acceptable=(total_frag == 0),
        )

    @staticmethod
    def export_json(report: FragmentationReport, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

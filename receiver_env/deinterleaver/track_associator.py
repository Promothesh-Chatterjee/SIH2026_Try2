"""Multi-Attribute Association and Conflict Resolution Module with Audit Logging."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np

from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig, EmitterTrack
from receiver_env.deinterleaver.frequency_grouper import FrequencyGrouper
from receiver_env.deinterleaver.association_audit import AssociationAudit


class TrackAssociator:
    """Evaluates multi-attribute match scores and resolves conflicts between competing tracks."""

    def __init__(
        self,
        config: DeinterleaverConfig | None = None,
        audit: AssociationAudit | None = None,
    ) -> None:
        self.config = config or DeinterleaverConfig()
        self.freq_grouper = FrequencyGrouper(self.config)
        self.audit = audit

    def find_best_track(
        self,
        pdw: PDW,
        tracks: Sequence[EmitterTrack],
    ) -> Optional[EmitterTrack]:
        """Find the best matching active track for an incoming PDW with conflict resolution."""
        if not tracks:
            if self.audit:
                self.audit.record(
                    pulse_id=pdw.pulse_id,
                    pdw_id=pdw.pdw_id,
                    assigned_track_id=None,
                    candidate_track_id=None,
                    frequency_score=0.0,
                    pri_score=0.0,
                    pw_score=0.0,
                    confidence_score=pdw.confidence,
                    total_score=0.0,
                    decision_threshold=self.config.association_threshold,
                    accepted=False,
                    rejection_reason="No active tracks",
                )
            return None

        candidates: List[Tuple[float, float, bool, EmitterTrack]] = []

        for track in tracks:
            rejection_reason = ""
            # 1. Physical Minimum Interval Constraint
            delta_toa = pdw.toa_us - track.last_toa_us
            if delta_toa < self.config.min_pri_us:
                rejection_reason = f"Delta-ToA ({delta_toa:.1f} us) < min_pri ({self.config.min_pri_us:.1f} us)"
                if self.audit:
                    self._record_audit(pdw, track, 0.0, 0.0, 0.0, 0.0, False, rejection_reason)
                continue

            # 2. Pulse Width Gate Check (with collision tolerance)
            pw_gate = self.config.pw_gate_pct * track.mean_pw_us
            delta_pw = abs(pdw.pulse_width_us - track.mean_pw_us)
            if delta_pw > pw_gate:
                rejection_reason = f"PW delta ({delta_pw:.2f} us) > gate ({pw_gate:.2f} us)"
                if self.audit:
                    self._record_audit(pdw, track, 0.0, 0.0, 0.0, 0.0, False, rejection_reason)
                continue
            score_pw = max(0.0, 1.0 - (delta_pw / max(1e-6, pw_gate)))

            # 3. PRI Timing & Consistency Check
            is_locked = (track.estimated_pri_us > 0.0 and track.pulse_count >= 2)
            timing_residual = 0.0

            if is_locked:
                k = max(1, int(round(delta_toa / track.estimated_pri_us)))
                expected = k * track.estimated_pri_us
                timing_residual = abs(delta_toa - expected)
                tol = max(3.0, (0.12 + track.pri_jitter_pct / 100.0) * expected)
                if timing_residual > tol:
                    rejection_reason = f"Timing residual ({timing_residual:.1f} us) > tol ({tol:.1f} us)"
                    if self.audit:
                        self._record_audit(pdw, track, 0.0, 0.0, score_pw, 0.0, False, rejection_reason)
                    continue
                score_pri = math.exp(-0.5 * (timing_residual / (tol / 2.0)) ** 2)
                pri_bonus = 0.25
            else:
                # Tentative track interval consistency check
                if len(track.history.recent_toas) == 2:
                    prev_interval = track.history.recent_toas[1] - track.history.recent_toas[0]
                    k = max(1, int(round(delta_toa / prev_interval)))
                    res = abs(delta_toa - k * prev_interval)
                    if res > max(3.0, 0.15 * prev_interval):
                        rejection_reason = f"Tentative interval residual ({res:.1f} us) inconsistent"
                        if self.audit:
                            self._record_audit(pdw, track, 0.0, 0.0, score_pw, 0.0, False, rejection_reason)
                        continue
                    score_pri = 0.80
                else:
                    # Single-pulse seed track: no timing periodicity observed yet
                    score_pri = 0.0
                pri_bonus = 0.0

            # 4. Adaptive Frequency Gate Check (with PRI-locked known hop channel guard)
            score_freq = 0.0
            freq_compat = self.freq_grouper.is_compatible(pdw.frequency_mhz, track.mean_frequency_mhz, track.track_confidence)
            if freq_compat:
                score_freq = self.freq_grouper.compute_match_score(
                    pdw.frequency_mhz, track.mean_frequency_mhz, track.track_confidence
                )
            elif is_locked and track.pulse_count >= 4 and score_pri >= 0.85:
                # PRI lock established: allow known hop channels from frequency history
                known_freqs = list(track.frequency_history) if hasattr(track, "frequency_history") and track.frequency_history else list(track.history.recent_frequency_mhz)
                min_delta = min((abs(pdw.frequency_mhz - f) for f in known_freqs), default=999.0)
                if min_delta <= self.config.frequency_gate_mhz:
                    score_freq = 0.90
                else:
                    rejection_reason = f"Freq delta ({abs(pdw.frequency_mhz - track.mean_frequency_mhz):.2f} MHz) outside gate"
                    if self.audit:
                        self._record_audit(pdw, track, 0.0, score_pri, score_pw, 0.0, False, rejection_reason)
                    continue
            else:
                rejection_reason = f"Freq delta ({abs(pdw.frequency_mhz - track.mean_frequency_mhz):.2f} MHz) outside gate"
                if self.audit:
                    self._record_audit(pdw, track, 0.0, score_pri, score_pw, 0.0, False, rejection_reason)
                continue

            score_conf = max(0.0, min(1.0, pdw.confidence))

            # Adaptive weights: shift weight to PRI if PRI confidence is high
            wf = self.config.weight_freq
            wpri = self.config.weight_pri
            if is_locked and track.pri_confidence > 0.80:
                wpri += 0.05
                wf -= 0.05

            if track.pulse_count == 1:
                # Single-pulse tentative track: evaluate on frequency, pulse width, and confidence
                non_pri_w = wf + self.config.weight_pw + self.config.weight_conf
                total_score = (
                    wf * score_freq
                    + self.config.weight_pw * score_pw
                    + self.config.weight_conf * score_conf
                ) / max(1e-6, non_pri_w)
            else:
                total_score = (
                    wf * score_freq
                    + wpri * score_pri
                    + self.config.weight_pw * score_pw
                    + self.config.weight_conf * score_conf
                    + pri_bonus
                )

            accepted = (total_score >= self.config.association_threshold)
            if not accepted:
                rejection_reason = f"Total score ({total_score:.3f}) < threshold ({self.config.association_threshold:.2f})"

            if self.audit:
                self._record_audit(pdw, track, score_freq, score_pri, score_pw, total_score, accepted, rejection_reason)

            if accepted:
                candidates.append((total_score, -timing_residual, is_locked, track))

        if not candidates:
            return None

        # Conflict resolution: sort prioritizing locked PRIs, then score, then lower residual
        candidates.sort(key=lambda c: (c[2], c[0], c[1]), reverse=True)
        return candidates[0][3]

    def associate_batch(
        self,
        pdws: Sequence[PDW],
        tracks: Sequence[EmitterTrack],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Batch association of incoming PDWs to active tracks.

        Returns:
            Tuple of (assigned_pairs, unassigned_indices)
            where assigned_pairs is a list of (pdw_idx, track_idx)
            and unassigned_indices is a list of pdw indices that were not associated.
        """
        if not tracks or not pdws:
            return [], list(range(len(pdws)))

        track_to_idx = {t.track_id: i for i, t in enumerate(tracks)}
        assigned: List[Tuple[int, int]] = []
        unassigned: List[int] = []

        for p_idx, pdw in enumerate(pdws):
            matched = self.find_best_track(pdw, tracks)
            if matched is not None and matched.track_id in track_to_idx:
                assigned.append((p_idx, track_to_idx[matched.track_id]))
            else:
                unassigned.append(p_idx)

        return assigned, unassigned

    def _record_audit(
        self,
        pdw: PDW,
        track: EmitterTrack,
        sf: float,
        spri: float,
        spw: float,
        total: float,
        accepted: bool,
        reason: str,
    ) -> None:
        if self.audit:
            self.audit.record(
                pulse_id=pdw.pulse_id,
                pdw_id=pdw.pdw_id,
                assigned_track_id=track.track_id if accepted else None,
                candidate_track_id=track.track_id,
                frequency_score=sf,
                pri_score=spri,
                pw_score=spw,
                confidence_score=pdw.confidence,
                total_score=total,
                decision_threshold=self.config.association_threshold,
                accepted=accepted,
                rejection_reason=reason,
            )

"""Unified Phase 4.1 PDW Deinterleaving & Emitter Separation Subsystem."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig, EmitterTrack
from receiver_env.deinterleaver.track_associator import TrackAssociator
from receiver_env.deinterleaver.track_manager import TrackManager
from receiver_env.deinterleaver.association_audit import AssociationAudit


class PDWDeinterleaver:
    """Production Phase 4.1 PDW Deinterleaver and Emitter Track Manager."""

    def __init__(
        self,
        config: DeinterleaverConfig | None = None,
        audit: AssociationAudit | None = None,
    ) -> None:
        self.config = config or DeinterleaverConfig()
        self.audit = audit
        self.track_manager = TrackManager(self.config)
        self.associator = TrackAssociator(self.config, audit=self.audit)

    def process_batch(self, pdws: Sequence[PDW]) -> List[EmitterTrack]:
        """Process a batch of incoming PDW objects in chronological order."""
        if not pdws:
            return self.track_manager.active_tracks

        sorted_pdws = sorted(pdws, key=lambda p: p.toa_us)

        for pdw in sorted_pdws:
            active_tracks = self.track_manager.active_tracks
            matched_track = self.associator.find_best_track(pdw, active_tracks)

            if matched_track is not None:
                self.track_manager.update_track(matched_track, pdw)
            else:
                self.track_manager.create_track(pdw)

        self.track_manager.merge_tracks()
        max_toa = sorted_pdws[-1].toa_us
        self.track_manager.expire_tracks(max_toa)

        return self.track_manager.active_tracks

    def process_pdw(self, pdw: PDW) -> List[EmitterTrack]:
        return self.process_batch([pdw])

    def get_active_tracks(self) -> List[EmitterTrack]:
        return self.track_manager.active_tracks

    def get_all_tracks(self) -> List[EmitterTrack]:
        return self.track_manager.all_tracks

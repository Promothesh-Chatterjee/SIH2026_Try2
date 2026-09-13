"""Emitter Track Lifecycle and State Management with Hardened Merging."""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional
import numpy as np

from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import (
    DeinterleaverConfig,
    EmitterStatistics,
    EmitterTrack,
    TrackHistory,
    TrackStatus,
)
from receiver_env.deinterleaver.pri_estimator import PRIEstimator


class TrackManager:
    """Manages creation, recursive updates, merging, and expiry of emitter tracks."""

    def __init__(self, config: DeinterleaverConfig | None = None) -> None:
        self.config = config or DeinterleaverConfig()
        self.pri_estimator = PRIEstimator(
            min_pri_us=self.config.min_pri_us,
            max_pri_us=self.config.max_pri_us,
            bin_width_us=self.config.pri_bin_width_us,
            harmonic_tolerance_pct=self.config.harmonic_tolerance_pct,
        )
        self._next_track_id: int = 1
        self._active_tracks: Dict[int, EmitterTrack] = {}
        self._terminated_tracks: List[EmitterTrack] = []

    @property
    def active_tracks(self) -> List[EmitterTrack]:
        return list(self._active_tracks.values())

    @property
    def all_tracks(self) -> List[EmitterTrack]:
        return list(self._active_tracks.values()) + self._terminated_tracks

    def create_track(self, seed_pdw: PDW) -> EmitterTrack:
        track_id = self._next_track_id
        self._next_track_id += 1

        history = TrackHistory(max_history=self.config.max_history_len)
        history.append(seed_pdw.toa_us, seed_pdw.pulse_width_us, seed_pdw.frequency_mhz)

        freq_hist = deque([seed_pdw.frequency_mhz], maxlen=self.config.max_history_len)

        track = EmitterTrack(
            track_id=track_id,
            emitter_id=f"TRACK_{track_id:04d}",
            pulse_count=1,
            mean_frequency_mhz=seed_pdw.frequency_mhz,
            mean_pw_us=seed_pdw.pulse_width_us,
            estimated_pri_us=0.0,
            first_toa_us=seed_pdw.toa_us,
            last_toa_us=seed_pdw.toa_us,
            track_confidence=float(seed_pdw.confidence * 0.5),
            status=TrackStatus.TENTATIVE,
            history=history,
            stats=EmitterStatistics(frequency_std_mhz=0.0, pw_std_us=0.0, pri_std_us=0.0),
            assigned_pdw_ids={seed_pdw.pdw_id},
            frequency_history=freq_hist,
            latest_frequency_mhz=seed_pdw.frequency_mhz,
            frequency_span_mhz=0.0,
            frequency_hopping_detected=False,
            hop_rate_hz=0.0,
        )

        self._active_tracks[track_id] = track
        return track

    def update_track(self, track: EmitterTrack, pdw: PDW) -> None:
        n = track.pulse_count + 1
        track.pulse_count = n

        delta_f = pdw.frequency_mhz - track.mean_frequency_mhz
        track.mean_frequency_mhz += delta_f / n

        delta_pw = pdw.pulse_width_us - track.mean_pw_us
        track.mean_pw_us += delta_pw / n

        track.last_toa_us = max(track.last_toa_us, pdw.toa_us)
        track.first_toa_us = min(track.first_toa_us, pdw.toa_us)
        track.assigned_pdw_ids.add(pdw.pdw_id)

        track.history.append(pdw.toa_us, pdw.pulse_width_us, pdw.frequency_mhz)

        if len(track.history.recent_frequency_mhz) > 1:
            track.stats.frequency_std_mhz = float(np.std(track.history.recent_frequency_mhz, ddof=1))
            track.stats.pw_std_us = float(np.std(track.history.recent_pw_us, ddof=1))

        track.latest_frequency_mhz = pdw.frequency_mhz
        track.frequency_history.append(pdw.frequency_mhz)
        if len(track.frequency_history) > 1:
            track.frequency_span_mhz = float(max(track.frequency_history) - min(track.frequency_history))

        if track.frequency_span_mhz > 2.0 and n >= 4:
            track.frequency_hopping_detected = True
            fh = list(track.frequency_history)
            hops = sum(1 for i in range(1, len(fh)) if abs(fh[i] - fh[i - 1]) > 2.0)
            dt_s = (track.last_toa_us - track.first_toa_us) * 1e-6
            track.hop_rate_hz = float(hops / dt_s) if dt_s > 0 else 0.0
        else:
            track.frequency_hopping_detected = False
            track.hop_rate_hz = 0.0

        should_estimate = (
            len(track.history.recent_toas) >= 3
            and (track.estimated_pri_us <= 0.0 or n <= 15 or n % 10 == 0)
        )
        if should_estimate:
            pri, jitter, pri_std, pri_conf = self.pri_estimator.estimate(list(track.history.recent_toas))
            if pri > 0.0:
                track.estimated_pri_us = pri
                track.pri_jitter_pct = jitter
                track.stats.pri_std_us = pri_std
                track.pri_confidence = pri_conf

        if track.status == TrackStatus.TENTATIVE and n >= self.config.min_confirm_pulses:
            if track.estimated_pri_us > 0.0 or n >= 5:
                track.status = TrackStatus.CONFIRMED

        pri_factor = track.pri_confidence if track.estimated_pri_us > 0.0 else 0.5
        count_factor = min(1.0, n / 10.0)
        pdw_factor = pdw.confidence
        track.track_confidence = float(np.clip(0.4 * pri_factor + 0.3 * count_factor + 0.3 * pdw_factor, 0.1, 0.99))

    def merge_tracks(self) -> int:
        """Merge fragmented tracks with hardened false-merge protection."""
        active_list = list(self._active_tracks.values())
        merged_count = 0
        to_remove = set()

        for i in range(len(active_list)):
            t1 = active_list[i]
            if t1.track_id in to_remove:
                continue

            for j in range(i + 1, len(active_list)):
                t2 = active_list[j]
                if t2.track_id in to_remove:
                    continue

                if self._can_merge(t1, t2):
                    self._merge_into(t1, t2)
                    to_remove.add(t2.track_id)
                    merged_count += 1

        for track_id in to_remove:
            if track_id in self._active_tracks:
                del self._active_tracks[track_id]

        return merged_count

    def _can_merge(self, t1: EmitterTrack, t2: EmitterTrack) -> bool:
        """Evaluate whether two tracks can be merged safely."""
        # 1. Frequency delta check
        if abs(t1.mean_frequency_mhz - t2.mean_frequency_mhz) > self.config.merge_freq_gate_mhz:
            return False

        # 2. Pulse width delta check
        delta_pw_pct = abs(t1.mean_pw_us - t2.mean_pw_us) / max(1e-6, t1.mean_pw_us)
        if delta_pw_pct > self.config.merge_pw_gate_pct:
            return False

        # 3. PRI Compatibility Check: Require both tracks to have established PRIs before merging
        if t1.estimated_pri_us <= 0.0 or t2.estimated_pri_us <= 0.0:
            # Cannot safely verify periodicity similarity without established PRIs
            return False

        delta_pri_pct = abs(t1.estimated_pri_us - t2.estimated_pri_us) / max(1e-6, t1.estimated_pri_us)
        if delta_pri_pct > self.config.merge_pri_gate_pct:
            # Check for 2x harmonic multiple
            ratio = t1.estimated_pri_us / t2.estimated_pri_us
            is_harmonic = abs(ratio - 2.0) < 0.08 or abs(ratio - 0.5) < 0.08
            if not is_harmonic:
                return False

        # 4. Temporal non-collision check: tracks cannot emit two overlapping pulses
        # Ensure ToA spacing between recent pulses is consistent
        if len(t1.history.recent_toas) > 0 and len(t2.history.recent_toas) > 0:
            t1_last = t1.last_toa_us
            t2_last = t2.last_toa_us
            if abs(t1_last - t2_last) < self.config.min_pri_us and t1.pulse_count > 1 and t2.pulse_count > 1:
                return False

        return True

    def _merge_into(self, target: EmitterTrack, source: EmitterTrack) -> None:
        total_pulses = target.pulse_count + source.pulse_count
        target.mean_frequency_mhz = (
            target.mean_frequency_mhz * target.pulse_count + source.mean_frequency_mhz * source.pulse_count
        ) / total_pulses

        target.mean_pw_us = (
            target.mean_pw_us * target.pulse_count + source.mean_pw_us * source.pulse_count
        ) / total_pulses

        target.pulse_count = total_pulses
        target.first_toa_us = min(target.first_toa_us, source.first_toa_us)
        target.last_toa_us = max(target.last_toa_us, source.last_toa_us)
        target.assigned_pdw_ids.update(source.assigned_pdw_ids)

        combined_history = list(zip(target.history.recent_toas, target.history.recent_pw_us, target.history.recent_frequency_mhz)) +                            list(zip(source.history.recent_toas, source.history.recent_pw_us, source.history.recent_frequency_mhz))
        combined_history.sort(key=lambda x: x[0])
        combined_history = combined_history[-self.config.max_history_len:]

        target.history.clear()
        for toa, pw, freq in combined_history:
            target.history.append(toa, pw, freq)

        if len(target.history.recent_toas) >= 3:
            pri, jitter, pri_std, pri_conf = self.pri_estimator.estimate(list(target.history.recent_toas))
            if pri > 0.0:
                target.estimated_pri_us = pri
                target.pri_jitter_pct = jitter
                target.stats.pri_std_us = pri_std
                target.pri_confidence = pri_conf

        target.latest_frequency_mhz = target.history.recent_frequency_mhz[-1] if target.history.recent_frequency_mhz else target.mean_frequency_mhz
        target.frequency_history = deque(list(target.history.recent_frequency_mhz), maxlen=self.config.max_history_len)
        if len(target.frequency_history) > 1:
            target.frequency_span_mhz = float(max(target.frequency_history) - min(target.frequency_history))
        if target.frequency_span_mhz > 2.0 and target.pulse_count >= 4:
            target.frequency_hopping_detected = True
            fh = list(target.frequency_history)
            hops = sum(1 for i in range(1, len(fh)) if abs(fh[i] - fh[i - 1]) > 2.0)
            dt_s = (target.last_toa_us - target.first_toa_us) * 1e-6
            target.hop_rate_hz = float(hops / dt_s) if dt_s > 0 else 0.0
        else:
            target.frequency_hopping_detected = False
            target.hop_rate_hz = 0.0

        target.track_confidence = max(target.track_confidence, source.track_confidence)
        source.status = TrackStatus.TERMINATED

    def expire_tracks(self, current_toa_us: float) -> List[EmitterTrack]:
        expired = []
        for track_id, track in list(self._active_tracks.items()):
            timeout = (
                self.config.track_expiry_pri_mult * track.estimated_pri_us
                if track.estimated_pri_us > 0.0
                else self.config.track_expiry_timeout_us
            )
            timeout = max(timeout, 50_000.0)
            if current_toa_us - track.last_toa_us > timeout:
                track.status = TrackStatus.TERMINATED
                expired.append(track)
                del self._active_tracks[track_id]
                self._terminated_tracks.append(track)
        return expired

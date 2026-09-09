"""
Reservation Manager for Cognitive EW Smart Scan.

Maintains projected future arrival reservations and deadlines for slow hoppers,
periodic emitters, and long-PRI targets without blind early tuning or premature exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.contracts import NORMAL_DWELL, LONG_DWELL, PREEMPTIVE_INTERCEPT


class ReservationStatus(str, Enum):
    PENDING = "PENDING"      # Arrival is far in the future; scan other bands freely
    ACTIVE = "ACTIVE"        # Imminent arrival: receiver must tune NOW to capture pulse
    EXECUTED = "EXECUTED"    # Dwell completed
    EXPIRED = "EXPIRED"      # Window passed without interception


@dataclass
class TemporalReservation:
    """A scheduled future interception reservation."""
    reservation_id: int
    track_id: int
    target_band: int
    target_mode: int
    expected_toa_us: float
    window_start_us: float
    window_end_us: float
    deadline_us: float
    confidence: float
    priority: float
    status: ReservationStatus = ReservationStatus.PENDING

    @property
    def is_actionable(self) -> bool:
        return self.status == ReservationStatus.ACTIVE


class ReservationManager:
    """Manages future arrival windows and deadlines for temporal scheduling."""

    def __init__(
        self,
        retune_latency_us: float = 15.0,
        default_window_half_width_us: float = 60.0,
        pre_arrival_lead_us: float = 40.0,
    ) -> None:
        self.retune_latency_us: float = float(retune_latency_us)
        self.default_window_half_width_us: float = float(default_window_half_width_us)
        self.pre_arrival_lead_us: float = float(pre_arrival_lead_us)

        self._reservations: Dict[int, TemporalReservation] = {}
        self._next_id: int = 1

    def reset(self) -> None:
        """Clear all active reservations."""
        self._reservations.clear()
        self._next_id = 1

    def create_or_update_reservation(
        self,
        track_id: int,
        target_band: int,
        expected_toa_us: float,
        confidence: float,
        priority: float = 1.0,
        target_mode: int = NORMAL_DWELL,
        window_half_width_us: float | None = None,
    ) -> TemporalReservation:
        """Register or update a future arrival reservation for a track."""
        track_id = int(track_id)
        half_w = float(window_half_width_us if window_half_width_us is not None else self.default_window_half_width_us)
        win_start = max(0.0, float(expected_toa_us) - half_w)
        win_end = float(expected_toa_us) + half_w
        # Deadline: receiver must initiate retune before arrival window starts
        deadline = max(0.0, win_start - self.retune_latency_us - self.pre_arrival_lead_us)

        # Look for existing reservation for this track
        existing_res_id = None
        for r_id, res in self._reservations.items():
            if res.track_id == track_id and res.status in (ReservationStatus.PENDING, ReservationStatus.ACTIVE):
                existing_res_id = r_id
                break

        if existing_res_id is not None:
            res = self._reservations[existing_res_id]
            res.target_band = int(target_band)
            res.target_mode = int(target_mode)
            res.expected_toa_us = float(expected_toa_us)
            res.window_start_us = win_start
            res.window_end_us = win_end
            res.deadline_us = deadline
            res.confidence = float(confidence)
            res.priority = float(priority)
            return res

        r_id = self._next_id
        self._next_id += 1
        res = TemporalReservation(
            reservation_id=r_id,
            track_id=track_id,
            target_band=int(target_band),
            target_mode=int(target_mode),
            expected_toa_us=float(expected_toa_us),
            window_start_us=win_start,
            window_end_us=win_end,
            deadline_us=deadline,
            confidence=float(confidence),
            priority=float(priority),
            status=ReservationStatus.PENDING,
        )
        self._reservations[r_id] = res
        return res

    def get_actionable_reservation(
        self,
        current_time_us: float,
        dwell_duration_us: float = 500.0,
    ) -> Optional[TemporalReservation]:
        """Check if any pending reservation has arrived at its execution deadline.

        A reservation is actionable if:
          1. Current time has reached the retune deadline: t >= deadline
          2. The dwell would cover the arrival: t + retune <= window_end
          3. It has high confidence and hasn't expired.
        """
        curr_t = float(current_time_us)
        candidates: List[TemporalReservation] = []

        for r_id, res in list(self._reservations.items()):
            if res.status == ReservationStatus.EXECUTED:
                continue

            # Expired check
            if curr_t > res.window_end_us:
                res.status = ReservationStatus.EXPIRED
                continue

            # Check if execution deadline is reached:
            # If we start retuning now (taking retune_latency_us), will the dwell window overlap the pulse arrival?
            dwell_start = curr_t + self.retune_latency_us
            dwell_end = dwell_start + float(dwell_duration_us)

            # Overlap condition: dwell_start <= res.expected_toa <= dwell_end
            # or deadline reached and not yet past window_end
            if res.deadline_us <= curr_t <= res.window_end_us:
                if res.confidence >= 0.40:
                    res.status = ReservationStatus.ACTIVE
                    candidates.append(res)
            else:
                res.status = ReservationStatus.PENDING

        if not candidates:
            return None

        # Sort by priority desc, then ETA asc
        candidates.sort(key=lambda r: (-r.priority, abs(r.expected_toa_us - curr_t)))
        return candidates[0]

    def mark_executed(self, reservation_id: int) -> None:
        """Mark reservation as executed."""
        if reservation_id in self._reservations:
            self._reservations[reservation_id].status = ReservationStatus.EXECUTED

    def prune(self, current_time_us: float) -> None:
        """Prune expired or completed reservations older than 5,000 µs."""
        curr_t = float(current_time_us)
        to_delete = [
            r_id for r_id, r in self._reservations.items()
            if r.status in (ReservationStatus.EXPIRED, ReservationStatus.EXECUTED)
            and curr_t - r.window_end_us > 5000.0
        ]
        for r_id in to_delete:
            del self._reservations[r_id]

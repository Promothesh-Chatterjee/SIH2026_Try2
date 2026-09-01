"""RadioEnvironment with ENTRY/EXIT event lifecycle and time-sorted event queue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class PulseRecord:
    toa_us: float
    frequency_mhz: float
    pulse_width_us: float
    amplitude_db: float
    aoa_deg: float
    emitter_id: int = 0
    source_id: str = "synthetic"

    @property
    def exit_us(self) -> float:
        return self.toa_us + self.pulse_width_us


@dataclass
class ActivePulse:
    pulse_id: int
    toa_us: float
    frequency_mhz: float
    pulse_width_us: float
    amplitude_db: float
    aoa_deg: float
    emitter_id: int
    exit_us: float
    source_id: str = ""


@dataclass
class SimulationEvent:
    event_type: str
    time_us: float
    pulse_id: Optional[int] = None
    pulse: Optional[ActivePulse] = None
    active_count: int = 0


class RadioEnvironment:
    """Event-driven RF simulation. Generates ENTRY and EXIT events in time order.

    Each PulseRecord produces two events:
        ENTRY at toa_us  (pulse becomes active)
        EXIT  at exit_us (pulse leaves)
    """

    def __init__(
        self,
        records: Optional[Sequence[PulseRecord]] = None,
        on_event: Optional[Callable[[SimulationEvent], None]] = None,
    ):
        self.records = sorted(list(records or []), key=lambda r: r.toa_us)
        self._callbacks: List[Callable[[SimulationEvent], None]] = []
        if on_event is not None:
            if callable(on_event):
                self._callbacks.append(on_event)
            else:
                self._callbacks.extend(on_event)

        self.active: Dict[int, ActivePulse] = {}
        self.time_us: float = 0.0
        self.done = False
        self._pulse_seq = 0

        # Build a sorted event queue: (time, priority, event_type, pulse_record)
        # priority: 0=EXIT before ENTRY at same time (pulse ends before new one at same toa)
        self._event_queue: list[tuple[float, int, str, Optional[PulseRecord], Optional[int]]] = []
        self._queue_idx = 0

        for rec in self.records:
            pid = self._pulse_seq
            self._pulse_seq += 1
            pulse = ActivePulse(
                pulse_id=pid,
                toa_us=rec.toa_us,
                frequency_mhz=rec.frequency_mhz,
                pulse_width_us=rec.pulse_width_us,
                amplitude_db=rec.amplitude_db,
                aoa_deg=rec.aoa_deg,
                emitter_id=rec.emitter_id,
                exit_us=rec.exit_us,
                source_id=rec.source_id,
            )
            self._event_queue.append((rec.toa_us, 1, "entry", rec, pid))
            self._event_queue.append((rec.exit_us, 0, "exit", rec, pid))

        self._event_queue.sort(key=lambda e: (e[0], e[1]))

    def add_callback(self, cb: Callable[[SimulationEvent], None]) -> None:
        self._callbacks.append(cb)

    def _emit(self, event: SimulationEvent) -> None:
        for cb in self._callbacks:
            cb(event)

    @property
    def total_events(self) -> int:
        return len(self._event_queue)

    @property
    def remaining_events(self) -> int:
        return len(self._event_queue) - self._queue_idx

    def peek_time(self) -> Optional[float]:
        """Return the time of the next event without consuming it, or None."""
        if self._queue_idx >= len(self._event_queue):
            return None
        return self._event_queue[self._queue_idx][0]

    def step(self) -> Optional[SimulationEvent]:
        """Pop the next event from the sorted queue. Returns None when exhausted."""
        if self.done or self._queue_idx >= len(self._event_queue):
            self.done = True
            return None

        time_us, _pri, event_type, rec, pulse_id = self._event_queue[self._queue_idx]
        self._queue_idx += 1
        self.time_us = time_us

        if event_type == "entry":
            # Build and store the active pulse
            pulse = ActivePulse(
                pulse_id=pulse_id,
                toa_us=rec.toa_us,
                frequency_mhz=rec.frequency_mhz,
                pulse_width_us=rec.pulse_width_us,
                amplitude_db=rec.amplitude_db,
                aoa_deg=rec.aoa_deg,
                emitter_id=rec.emitter_id,
                exit_us=rec.exit_us,
                source_id=rec.source_id,
            )
            self.active[pulse_id] = pulse
            event = SimulationEvent(
                event_type="entry",
                time_us=time_us,
                pulse_id=pulse_id,
                pulse=pulse,
                active_count=len(self.active),
            )
            self._emit(event)
            return event

        if event_type == "exit":
            pulse = self.active.pop(pulse_id, None)
            event = SimulationEvent(
                event_type="exit",
                time_us=time_us,
                pulse_id=pulse_id,
                pulse=pulse,
                active_count=len(self.active),
            )
            self._emit(event)
            return event

        return None

    def run(self) -> None:
        """Drain all events."""
        while not self.done:
            self.step()

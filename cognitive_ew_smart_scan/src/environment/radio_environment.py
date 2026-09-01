"""Minimal RadioEnvironment for the receiver integration path."""

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
    def __init__(self, records: Optional[Sequence[PulseRecord]] = None, on_event: Optional[Callable[[SimulationEvent], None]] = None):
        self.records = list(records or [])
        self._callbacks: List[Callable[[SimulationEvent], None]] = []
        if on_event is not None:
            if callable(on_event):
                self._callbacks.append(on_event)
            else:
                self._callbacks.extend(on_event)
        self.active: Dict[int, ActivePulse] = {}
        self.time_us: float = 0.0
        self.done = False
        self._index = 0
        self._pulse_seq = 0

    def add_callback(self, cb: Callable[[SimulationEvent], None]) -> None:
        self._callbacks.append(cb)

    def _emit(self, event: SimulationEvent) -> None:
        for cb in self._callbacks:
            cb(event)

    def step(self) -> Optional[SimulationEvent]:
        if self.done:
            return None
        if self._index >= len(self.records):
            self.done = True
            return None
        rec = self.records[self._index]
        self._index += 1
        self.time_us = rec.toa_us
        pulse = ActivePulse(
            pulse_id=self._pulse_seq,
            toa_us=rec.toa_us,
            frequency_mhz=rec.frequency_mhz,
            pulse_width_us=rec.pulse_width_us,
            amplitude_db=rec.amplitude_db,
            aoa_deg=rec.aoa_deg,
            emitter_id=rec.emitter_id,
            exit_us=rec.exit_us,
            source_id=rec.source_id,
        )
        self._pulse_seq += 1
        self.active[pulse.pulse_id] = pulse
        event = SimulationEvent(event_type="entry", time_us=rec.toa_us, pulse_id=pulse.pulse_id, pulse=pulse, active_count=len(self.active))
        self._emit(event)
        return event

    def run(self) -> None:
        while not self.done:
            self.step()

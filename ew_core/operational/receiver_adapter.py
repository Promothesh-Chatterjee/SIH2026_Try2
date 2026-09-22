"""
Physical Receiver Hardware & RF Stream Adapter.

Decouples the cognitive controller from physical SDR or simulated receiver backends.
Enforces:
- Physical frequency windowing (IBW 500 MHz)
- Temporal aperture visibility ([dwell_start, dwell_end])
- Hardware detection sensitivity threshold (-140 dBm)
- Causal RF feed: zero access to future pulses
- Stripping of ground-truth emitter labels: live detections contain only measured physical attributes
- Hardware health / disconnect fail-safe states
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    RF_BASE_DWELL_TIME_US,
    RF_FREQ_MAX_MHZ,
    RF_IBW_MHZ,
)
from ew_core.receiver.models import DetectionObservation, ReceiverObservation
from ew_core.receiver.sieve_receiver import SieveReceiver

logger = logging.getLogger(__name__)


class ReceiverHardwareError(RuntimeError):
    """Raised on receiver hardware communication failure or disconnect."""
    pass


class ReceiverAdapter:
    """Production adapter interfacing with physical RF receiver or SieveReceiver model."""

    def __init__(
        self,
        receiver: Optional[SieveReceiver] = None,
        total_bandwidth_mhz: float = RF_FREQ_MAX_MHZ,
        ibw_mhz: float = RF_IBW_MHZ,
        dwell_time_us: float = RF_BASE_DWELL_TIME_US,
        sensitivity_dbm: float = -140.0,
    ) -> None:
        self.receiver = receiver or SieveReceiver(
            total_bandwidth=total_bandwidth_mhz,
            ibw=ibw_mhz,
            frequency_step=ibw_mhz,
            dwell_time=dwell_time_us,
            detection_threshold_db=sensitivity_dbm,
        )
        self.ibw_mhz = float(ibw_mhz)
        self.sensitivity_dbm = float(sensitivity_dbm)
        self._is_connected: bool = True
        self._pulse_cursor: int = 0
        self._last_stream_id: Optional[int] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def set_connected(self, connected: bool) -> None:
        """Simulate hardware disconnect or reconnect for testing."""
        self._is_connected = bool(connected)

    def reset(self) -> None:
        """Reset receiver hardware state."""
        self.receiver.reset()
        self._is_connected = True
        self._pulse_cursor = 0
        self._last_stream_id = None

    def tune(self, center_freq_mhz: float) -> float:
        """Retune receiver local oscillator to center_freq_mhz."""
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot tune: RF Receiver is disconnected")
        return self.receiver.tune(center_freq_mhz)

    def set_dwell_time(self, duration_us: float) -> None:
        """Configure dwell aperture duration."""
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot configure dwell: RF Receiver is disconnected")
        self.receiver.set_dwell_time(duration_us)

    def feed_incident_rf(
        self,
        pulses: Sequence[Any],
        max_time_us: Optional[float] = None,
    ) -> int:
        """Causally ingest incident RF pulses into the physical receiver buffer.
        
        Only pulses with ToA <= max_time_us are ingested. Future pulses are ignored.
        """
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot feed RF: RF Receiver is disconnected")

        # Support RadioEnvironment directly if passed
        if hasattr(pulses, "peek_time") and hasattr(pulses, "step"):
            ingested = 0
            while pulses.remaining_events > 0:
                next_time = pulses.peek_time()
                if next_time is not None and max_time_us is not None and next_time > float(max_time_us):
                    break
                event = pulses.step()
                if event and getattr(event, "event_type", None) == "entry" and event.pulse is not None:
                    self.receiver.add_pulse(event.pulse)
                    ingested += 1
            return ingested

        if id(pulses) != self._last_stream_id:
            self._last_stream_id = id(pulses)
            self._pulse_cursor = 0

        ingested = 0
        n = len(pulses)
        while self._pulse_cursor < n:
            p = pulses[self._pulse_cursor]
            # Extract time
            t = getattr(p, "time_us", getattr(p, "toa_us", None))
            if t is None and isinstance(p, dict):
                t = p.get("time_us", p.get("toa_us"))
            if t is None:
                self._pulse_cursor += 1
                continue

            t = float(t)
            if max_time_us is not None and t > float(max_time_us):
                # Strictly enforce causality: no future pulse entering buffer
                break

            # Ensure pulse dictionary has standard field names expected by SieveReceiver
            pulse_to_add = p
            if isinstance(p, dict):
                pw = float(p.get("pulse_width_us", p.get("pw_us", 1.0)))
                pulse_to_add = {
                    "toa_us": t,
                    "time_us": t,
                    "exit_us": t + pw,
                    "frequency_mhz": p.get("frequency_mhz", p.get("freq_mhz")),
                    "pulse_width_us": pw,
                    "amplitude_db": float(p.get("amplitude_db", p.get("amp_db", -50.0))),
                    "aoa_deg": float(p.get("aoa_deg", p.get("aoa", 0.0))),
                    "pulse_id": p.get("pulse_id"),
                }

            self.receiver.add_pulse(pulse_to_add)
            self._pulse_cursor += 1
            ingested += 1

        return ingested

    def execute_dwell(
        self,
        dwell_start_us: float,
        dwell_end_us: float,
    ) -> List[Dict[str, Any]]:
        """Physically execute the receiver dwell aperture.
        
        Filters buffered pulses through the tuned RF passband and sensitivity gate.
        Returns intercepted PDWs with NO privileged ground-truth emitter labels.
        """
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot execute dwell: RF Receiver is disconnected")

        raw_detections: List[DetectionObservation] = self.receiver._detect_buffered_interval(
            dwell_start_us, dwell_end_us
        )

        # Update receiver internal clock and prune past pulses
        self.receiver.current_time_us = dwell_end_us
        self.receiver.dwell_start_us = dwell_start_us
        self.receiver.dwell_end_us = dwell_end_us
        self.receiver._prune(dwell_end_us)

        # Sanitize detections into pure operational PDWs (stripping ground truth)
        clean_pdws: List[Dict[str, Any]] = []
        for det in raw_detections:
            pdw = {
                "time_us": float(det.time_us),
                "frequency_mhz": float(det.frequency_mhz),
                "pulse_width_us": float(det.pulse_width_us),
                "amplitude_db": float(det.amplitude_db),
                "aoa_deg": float(det.aoa_deg),
                "pulse_id": det.pulse_id,
                "center_frequency_mhz": float(det.center_frequency_mhz),
            }
            clean_pdws.append(pdw)

        return clean_pdws


# ── Hardware Abstraction Layer (Audit Item 12) ──────────────────────────────
from dataclasses import dataclass, field
import abc
import os


@dataclass
class ReceiverDetection:
    """A single detection from the receiver during one dwell."""
    band_idx: int
    freq_mhz: float
    power_dbm: float
    time_us: float
    confidence: float = 1.0


@dataclass
class ReceiverObservationHW:
    """Observation returned by the hardware backend after one dwell."""
    band_idx: int
    dwell_time_us: float
    detections: list[ReceiverDetection] = field(default_factory=list)
    noise_floor_dbm: float = -140.0
    timestamp_us: float = 0.0

    @property
    def hit(self) -> bool:
        return len(self.detections) > 0


class ReceiverBackend(abc.ABC):
    """Abstract base class for all receiver backends."""

    @abc.abstractmethod
    def tune(self, band_idx: int, mode_idx: int, dwell_time_us: float) -> None:
        """Tune the receiver to the given band and set dwell time."""

    @abc.abstractmethod
    def read_power(self, band_idx: int) -> float:
        """Return current received power in dBm for the tuned band."""

    @abc.abstractmethod
    def get_observation(self) -> ReceiverObservationHW:
        """Execute the current dwell and return detections."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset receiver state between episodes."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend identifier string."""


class SimulatedBackend(ReceiverBackend):
    """Simulated backend — wraps CognitiveRFScanEnv or SieveReceiver for software operation.

    Used in training, evaluation, and Azure deployment runs.
    """

    def __init__(self, env: Any = None):
        self._env = env
        self._current_band = 0
        self._current_mode = 0
        self._current_dwell_us = 500.0

    def tune(self, band_idx: int, mode_idx: int, dwell_time_us: float) -> None:
        self._current_band = int(band_idx)
        self._current_mode = int(mode_idx)
        self._current_dwell_us = float(dwell_time_us)

    def read_power(self, band_idx: int) -> float:
        if self._env is not None and hasattr(self._env, "get_band_power_dbm"):
            return float(self._env.get_band_power_dbm(band_idx))
        return -120.0

    def get_observation(self) -> ReceiverObservationHW:
        if self._env is not None and hasattr(self._env, "step"):
            action = self._current_band * 5 + self._current_mode
            obs, reward, done, truncated, info = self._env.step(action)
            detections = []
            if info.get("hit", False):
                detections.append(ReceiverDetection(
                    band_idx=self._current_band,
                    freq_mhz=self._current_band * 500.0 + 250.0,
                    power_dbm=info.get("signal_power_dbm", -80.0),
                    time_us=info.get("first_pulse_toa_us", getattr(self._env, "simulated_clock_us", 0.0)),
                ))
            return ReceiverObservationHW(
                band_idx=self._current_band,
                dwell_time_us=self._current_dwell_us,
                detections=detections,
            )
        return ReceiverObservationHW(
            band_idx=self._current_band,
            dwell_time_us=self._current_dwell_us,
            detections=[],
        )

    def reset(self) -> None:
        if self._env is not None and hasattr(self._env, "reset"):
            self._env.reset()

    @property
    def name(self) -> str:
        return "simulated"


class ZMQBackend(ReceiverBackend):
    """ZeroMQ backend for real SDR / GNU Radio hardware integration.

    Sends tune commands over ZMQ REQ socket and receives detection results.
    Compatible with GNU Radio ZMQ sink/source blocks.

    Set env var: ZMQ_RECEIVER_ADDR=tcp://192.168.1.100:5555
    """

    def __init__(self, address: Optional[str] = None):
        addr = address or os.environ.get("ZMQ_RECEIVER_ADDR", "tcp://localhost:5555")
        try:
            import zmq
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.REQ)
            self._socket.connect(addr)
            self._socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
            logger.info("ZMQBackend connected to %s", addr)
        except ImportError:
            raise RuntimeError("pyzmq not installed. Run: pip install pyzmq")
        self._current_band = 0
        self._current_dwell_us = 500.0

    def tune(self, band_idx: int, mode_idx: int, dwell_time_us: float) -> None:
        self._current_band = int(band_idx)
        self._current_dwell_us = float(dwell_time_us)
        import json
        cmd = json.dumps({
            "cmd": "tune",
            "band": band_idx,
            "mode": mode_idx,
            "dwell_us": dwell_time_us,
        })
        self._socket.send_string(cmd)
        self._socket.recv_string()  # ACK

    def read_power(self, band_idx: int) -> float:
        import json
        cmd = json.dumps({"cmd": "power", "band": band_idx})
        self._socket.send_string(cmd)
        reply = json.loads(self._socket.recv_string())
        return float(reply.get("power_dbm", -140.0))

    def get_observation(self) -> ReceiverObservationHW:
        import json
        cmd = json.dumps({
            "cmd": "dwell",
            "band": self._current_band,
            "dwell_us": self._current_dwell_us,
        })
        self._socket.send_string(cmd)
        reply = json.loads(self._socket.recv_string())
        detections = [
            ReceiverDetection(
                band_idx=self._current_band,
                freq_mhz=float(d["freq_mhz"]),
                power_dbm=float(d["power_dbm"]),
                time_us=float(d["time_us"]),
            )
            for d in reply.get("detections", [])
        ]
        return ReceiverObservationHW(
            band_idx=self._current_band,
            dwell_time_us=self._current_dwell_us,
            detections=detections,
            noise_floor_dbm=float(reply.get("noise_floor_dbm", -140.0)),
        )

    def reset(self) -> None:
        import json
        self._socket.send_string(json.dumps({"cmd": "reset"}))
        self._socket.recv_string()

    @property
    def name(self) -> str:
        return "zmq"


def get_backend(backend_type: str = "simulated", **kwargs) -> ReceiverBackend:
    """Factory function: return the appropriate ReceiverBackend by name.

    backend_type: "simulated" | "zmq"
    Set RECEIVER_BACKEND env var to control in deployment.
    """
    backends = {
        "simulated": SimulatedBackend,
        "zmq": ZMQBackend,
    }
    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}"
        )
    logger.info("ReceiverBackend initialised: %s", backend_type)
    return backends[backend_type](**kwargs)

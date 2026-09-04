"""IQ-to-PDW Extractor — Phase 3A.

Subscribes to the GNU Radio ZMQ IQ stream, detects pulses using magnitude
thresholding, estimates pulse parameters, and emits NDJSON PDW records.

The core detection logic (PDWDetector) operates on numpy arrays only and is
testable without GNU Radio.  The ZMQ subscriber is a thin wrapper around it.

Units
-----
- Sample rate:  2 MS/s  →  1 sample = 0.5 µs
- Frequency:    kHz (local/baseband, NOT logical RF)
- Time:         µs (sample stream simulation time, NOT wall-clock)
- Amplitude:    relative linear magnitude (RMS of complex IQ during pulse)
  NOT calibrated dBm.

Architecture
------------
GNU Radio IQ (complex64 over ZMQ tcp://127.0.0.1:55555)
    │
    ▼
PDWDetector.detect_iq(iq_samples, sample_offset)
    │
    ▼
list of PDW dicts (NDJSON to stdout)

Ground truth (emitter_id, logical RF) is NEVER used by the detector.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ================================================================
# PDW Detector (pure numpy — no GNU Radio dependency)
# ================================================================


class PDWDetector:
    """Magnitude-based pulse detector with frequency estimation.

    Parameters
    ----------
    sample_rate : float
        Sample rate in S/s.  Default 2e6 (2 MS/s).
    threshold_db_above_noise : float
        Detection threshold in dB above estimated noise floor.
        Default 10 dB.
    noise_estimation_samples : int
        Number of initial samples used to estimate noise floor.
        Default 1000.
    min_pulse_samples : int
        Minimum pulse width in samples to be considered a valid pulse.
        Default 4 (2 µs at 2 MS/s).
    max_pulse_samples : int
        Maximum allowed pulse width in samples before splitting.
        Default 200 (100 µs).
    hysteresis_db : float
        Hysteresis in dB for falling-edge detection (drop below
        threshold minus hysteresis).  Default 3 dB.
    """

    def __init__(
        self,
        sample_rate: float = 2e6,
        threshold_db_above_noise: float = 10.0,
        noise_estimation_samples: int = 1000,
        min_pulse_samples: int = 4,
        max_pulse_samples: int = 200,
        hysteresis_db: float = 3.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if noise_estimation_samples < 1:
            raise ValueError("noise_estimation_samples must be >= 1")
        if min_pulse_samples < 1:
            raise ValueError("min_pulse_samples must be >= 1")

        self.sample_rate = float(sample_rate)
        self.samples_per_us = self.sample_rate / 1e6
        self.threshold_db_above_noise = float(threshold_db_above_noise)
        self.noise_estimation_samples = int(noise_estimation_samples)
        self.min_pulse_samples = int(min_pulse_samples)
        self.max_pulse_samples = int(max_pulse_samples)
        self.hysteresis_db = float(hysteresis_db)

        self._noise_floor: Optional[float] = None

    # ------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------

    def estimate_noise_floor(self, iq: np.ndarray) -> float:
        """Estimate noise floor from the first N samples.

        Uses the 25th percentile of magnitude (robust against signal
        contamination in the estimation window).
        """
        n = min(self.noise_estimation_samples, len(iq))
        if n == 0:
            return 0.0
        mag = np.abs(iq[:n])
        self._noise_floor = float(np.percentile(mag, 25))
        return self._noise_floor

    @property
    def noise_floor(self) -> float:
        if self._noise_floor is None:
            return 0.0
        return self._noise_floor

    @property
    def detection_threshold(self) -> float:
        """Magnitude threshold = noise_floor * 10^(dB/20)."""
        return self.noise_floor * (10.0 ** (self.threshold_db_above_noise / 20.0))

    @property
    def hysteresis_threshold(self) -> float:
        """Magnitude threshold minus hysteresis."""
        return self.noise_floor * (
            10.0 ** ((self.threshold_db_above_noise - self.hysteresis_db) / 20.0)
        )

    # ------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------

    def detect_iq(
        self,
        iq: np.ndarray,
        sample_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Detect pulses in a complex IQ array and return PDW dicts.

        Parameters
        ----------
        iq : np.ndarray
            Complex IQ samples (complex64 or complex128).
        sample_offset : int
            Global sample index of ``iq[0]`` (for ToA calculation).

        Returns
        -------
        list of dict
            Each dict contains PDW fields.
        """
        iq = np.asarray(iq, dtype=np.complex64)

        if len(iq) == 0:
            return []

        # Estimate noise floor if not yet done.
        if self._noise_floor is None:
            self.estimate_noise_floor(iq)

        # Fallback: if noise floor is still 0 (e.g. all-zero onset), use
        # a peak-relative threshold — set noise floor so that the
        # detection threshold lands at 50% of the peak magnitude.
        if self.noise_floor <= 0:
            mag = np.abs(iq)
            peak = float(np.max(mag))
            if peak <= 0:
                return []
            self._noise_floor = peak / (10.0 ** (self.threshold_db_above_noise / 20.0)) * 0.5

        # Magnitude envelope.
        mag = np.abs(iq)

        # Thresholds.
        thresh = self.detection_threshold
        hyst = self.hysteresis_threshold

        # State machine: scan for pulses.
        pdws: List[Dict[str, Any]] = []
        i = 0
        n = len(mag)

        while i < n:
            # --- Rising edge: find start of pulse ---
            if mag[i] < thresh:
                i += 1
                continue

            # Pulse detected above threshold.
            pulse_start = i

            # --- Falling edge: find end of pulse ---
            while i < n and mag[i] >= hyst:
                i += 1

            pulse_end = i  # exclusive

            pulse_len = pulse_end - pulse_start

            # Discard undersized pulses.
            if pulse_len < self.min_pulse_samples:
                continue

            # Split oversized pulses (take first max_pulse_samples).
            while pulse_len > self.max_pulse_samples:
                pdw = self._make_pdw(
                    iq,
                    pulse_start,
                    pulse_start + self.max_pulse_samples,
                    sample_offset,
                )
                pdws.append(pdw)
                pulse_start += self.max_pulse_samples
                pulse_len = pulse_end - pulse_start

            if pulse_len >= self.min_pulse_samples:
                pdw = self._make_pdw(iq, pulse_start, pulse_end, sample_offset)
                pdws.append(pdw)

        return pdws

    # ------------------------------------------------------------
    # PDW field estimation
    # ------------------------------------------------------------

    def _make_pdw(
        self,
        iq: np.ndarray,
        start: int,
        end: int,
        sample_offset: int,
    ) -> Dict[str, Any]:
        """Build a PDW dict from detected pulse boundaries."""
        pulse_iq = iq[start:end]

        # ToA in µs.
        toa_us = (start + sample_offset) / self.samples_per_us

        # Pulse width in µs.
        pw_us = (end - start) / self.samples_per_us

        # Amplitude: RMS of complex IQ magnitude during pulse.
        amplitude = float(np.sqrt(np.mean(np.abs(pulse_iq) ** 2)))

        # Local/baseband frequency estimation via mean instantaneous frequency.
        freq_khz = self._estimate_frequency(pulse_iq)

        return {
            "type": "pdw",
            "toa_us": round(toa_us, 3),
            "frequency_local_khz": round(freq_khz, 3),
            "pulse_width_us": round(pw_us, 3),
            "amplitude": round(amplitude, 6),
            "source": "gnu_radio",
        }

    def _estimate_frequency(self, pulse_iq: np.ndarray) -> float:
        """Estimate local/baseband frequency of a pulse (kHz).

        Uses the mean of the instantaneous frequency estimated from the
        phase difference between consecutive samples.

        Returns frequency in kHz.
        """
        if len(pulse_iq) < 2:
            return 0.0

        # Phase difference between consecutive samples.
        phase_diff = np.angle(pulse_iq[1:] * np.conj(pulse_iq[:-1]))

        # Mean instantaneous frequency in radians per sample.
        mean_phase_diff = float(np.mean(phase_diff))

        # Convert to Hz: freq_hz = mean_phase_diff * sample_rate / (2 * pi)
        freq_hz = mean_phase_diff * self.sample_rate / (2.0 * np.pi)

        return freq_hz / 1e3  # to kHz


# ================================================================
# ZMQ Live Reader (thin GNU Radio wrapper)
# ================================================================


def run_live(
    address: str = "tcp://127.0.0.1:55555",
    sample_rate: float = 2e6,
    chunk_samples: int = 4096,
    duration_s: Optional[float] = None,
    **detector_kwargs: Any,
) -> None:
    """Subscribe to GNU Radio ZMQ IQ stream and emit PDWs as NDJSON.

    This function requires GNU Radio.  Import gnuradio only here so that
    the PDWDetector class can be used/tested without GNU Radio.
    """
    from gnuradio import gr, zeromq, blocks

    detector = PDWDetector(sample_rate=sample_rate, **detector_kwargs)

    # GNU Radio flowgraph: ZMQ SUB → vector sink (we drain periodically).
    tb = gr.top_block("iq_to_pdw_reader")

    sub = zeromq.sub_source(
        gr.sizeof_gr_complex,
        1,
        address,
        100,
        False,
        -1,
        "",
        False,
    )

    sink = blocks.vector_sink_c()

    tb.connect(sub, sink)

    print(
        json.dumps(
            {
                "type": "info",
                "message": "iq_to_pdw: connected to ZMQ",
                "address": address,
                "sample_rate": sample_rate,
                "chunk_samples": chunk_samples,
            }
        ),
        flush=True,
    )

    tb.start()

    sample_offset = 0
    start_time = time.time()

    try:
        while True:
            if duration_s is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration_s:
                    break

            # Collect one chunk.
            time.sleep(chunk_samples / sample_rate * 0.8)

            data = np.array(sink.data(), dtype=np.complex64)
            if len(data) == 0:
                continue

            # Detect pulses.
            pdws = detector.detect_iq(data, sample_offset=sample_offset)

            for pdw in pdws:
                print(json.dumps(pdw), flush=True)

            sample_offset += len(data)

            # Reset sink for next chunk.
            sink.reset()

    except KeyboardInterrupt:
        pass

    tb.stop()
    tb.wait()

    print(
        json.dumps(
            {
                "type": "info",
                "message": "iq_to_pdw: stopped",
                "total_samples": sample_offset,
            }
        ),
        flush=True,
    )


# ================================================================
# NDJSON File Reader (for offline testing with recorded IQ)
# ================================================================


def run_file(
    file_path: str,
    sample_rate: float = 2e6,
    chunk_samples: int = 4096,
    **detector_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Read IQ from a raw complex64 binary file and emit PDWs.

    The file must contain interleaved float32 I/Q samples
    (i.e. complex64 native endianness).

    Returns all detected PDWs.
    """
    raw = np.fromfile(file_path, dtype=np.complex64)
    detector = PDWDetector(sample_rate=sample_rate, **detector_kwargs)

    all_pdws: List[Dict[str, Any]] = []
    offset = 0

    while offset < len(raw):
        chunk = raw[offset : offset + chunk_samples]
        pdws = detector.detect_iq(chunk, sample_offset=offset)
        all_pdws.extend(pdws)
        offset += chunk_samples

    return all_pdws


# ================================================================
# CLI
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQ-to-PDW Extractor (Phase 3A)",
    )
    parser.add_argument(
        "--address",
        default="tcp://127.0.0.1:55555",
        help="ZMQ SUB address (default: tcp://127.0.0.1:55555)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2e6,
        help="Sample rate in S/s (default: 2e6)",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=4096,
        help="Samples per processing chunk (default: 4096)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then stop (default: infinite)",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=10.0,
        help="Detection threshold dB above noise floor (default: 10)",
    )
    parser.add_argument(
        "--min-pw-samples",
        type=int,
        default=4,
        help="Minimum pulse width in samples (default: 4)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Read from raw complex64 binary file instead of ZMQ",
    )

    args = parser.parse_args()

    kwargs = dict(
        threshold_db_above_noise=args.threshold_db,
        min_pulse_samples=args.min_pw_samples,
    )

    if args.file:
        pdws = run_file(
            args.file,
            sample_rate=args.sample_rate,
            chunk_samples=args.chunk_samples,
            **kwargs,
        )
        for pdw in pdws:
            print(json.dumps(pdw))
    else:
        run_live(
            address=args.address,
            sample_rate=args.sample_rate,
            chunk_samples=args.chunk_samples,
            duration_s=args.duration,
            **kwargs,
        )


if __name__ == "__main__":
    main()

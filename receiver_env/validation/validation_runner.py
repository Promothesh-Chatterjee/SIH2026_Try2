"""Validation Runner and Synthetic RF IQ Generator for Phase 1 EW Receiver."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generator, Iterable, List, Optional, Tuple, Union
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.frontend import ReceiverFrontend
from receiver_env.pulse_detector.detector import PulseDetector
from receiver_env.pulse_detector.models import DetectedPulse, ReceiverInput
from receiver_env.validation.metrics import (
    GroundTruthPulse,
    ValidationMetrics,
    evaluate_detections,
)

# Optional import of PulseRecord from cognitive_ew_smart_scan if available
try:
    from cognitive_ew_smart_scan.src.environment.radio_environment import PulseRecord
except ImportError:
    @dataclass
    class PulseRecord:  # type: ignore[no-redef]
        toa_us: float
        frequency_mhz: float
        pulse_width_us: float
        amplitude_db: float
        aoa_deg: float = 0.0
        emitter_id: int = 0
        source_id: str = "synthetic"


@dataclass
class ValidationResult:
    """Complete results from a validation scenario run."""

    metrics: ValidationMetrics
    detected_pulses: List[DetectedPulse]
    truth_pulses: List[GroundTruthPulse]
    total_samples: int
    total_chunks: int
    elapsed_time_s: float
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Format a summary including execution speed and metrics."""
        sample_rate_mhz = self.metadata.get("sample_rate_hz", 20e6) / 1e6
        throughput_msps = (self.total_samples / 1e6) / max(1e-6, self.elapsed_time_s)
        realtime_ratio = throughput_msps / max(1e-6, sample_rate_mhz)
        header = (
            f"\n--- Validation Execution Summary ---\n"
            f"  Total Samples Processed : {self.total_samples:,} ({self.total_chunks} chunks)\n"
            f"  Elapsed Processing Time : {self.elapsed_time_s:.3f} s\n"
            f"  Processing Throughput   : {throughput_msps:.2f} MSamples/s ({realtime_ratio:.2f}x Realtime)\n"
        )
        return header + self.metrics.summary()


class SyntheticSignalGenerator:
    """Synthetic RF baseband IQ signal generator with ground truth tracking.

    STRICTLY FOR VALIDATION AND TESTING.
    Generates realistic complex IQ sample streams containing pulsed RF signals,
    carrier frequency offsets, configurable noise, and precise sample-level ground truth.
    """

    def __init__(
        self,
        sample_rate_hz: float = 20_000_000.0,
        center_frequency_hz: float = 3_000_000_000.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.rng = np.random.default_rng(seed)

    def generate_iq_from_truth(
        self,
        truth_pulses: List[GroundTruthPulse],
        total_samples: int,
        snr_db: float = 15.0,
        add_noise: bool = True,
        edge_taper_samples: int = 2,
    ) -> np.ndarray:
        """Synthesize complex IQ waveform containing given ground-truth pulses and AWGN.

        Args:
            truth_pulses: List of GroundTruthPulse objects with sample boundaries.
            total_samples: Total timeline duration in samples.
            snr_db: Signal-to-noise ratio in dB relative to unit amplitude (1.0).
            add_noise: Whether to inject complex AWGN.
            edge_taper_samples: Smooth pulse envelope rise/fall to prevent spectral splatter.

        Returns:
            np.ndarray of complex64 IQ samples.
        """
        signal = np.zeros(total_samples, dtype=np.complex64)
        t = np.arange(total_samples, dtype=np.float64) / self.sample_rate_hz

        for p in truth_pulses:
            if p.start_sample >= total_samples:
                continue
            start = max(0, p.start_sample)
            end = min(total_samples, p.end_sample + 1)
            dur = end - start
            if dur <= 0:
                continue

            # Complex carrier tone with random initial phase per pulse
            phase0 = float(self.rng.uniform(0.0, 2.0 * np.pi))
            omega = 2.0 * np.pi * p.frequency_offset_hz
            carrier = np.exp(1j * (omega * t[start:end] + phase0), dtype=np.complex64)

            # Envelope shaping
            env = np.full(dur, p.peak_amplitude, dtype=np.float32)
            if edge_taper_samples > 0 and dur > 2 * edge_taper_samples:
                taper = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, edge_taper_samples, endpoint=False)))
                env[:edge_taper_samples] *= taper.astype(np.float32)
                env[-edge_taper_samples:] *= taper[::-1].astype(np.float32)

            signal[start:end] += (env * carrier).astype(np.complex64)

        if add_noise:
            # Noise variance based on unit reference signal power (0 dBFS = 1.0)
            noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
            sigma = np.sqrt(noise_power / 2.0)
            noise_i = self.rng.normal(0.0, sigma, total_samples).astype(np.float32)
            noise_q = self.rng.normal(0.0, sigma, total_samples).astype(np.float32)
            noise = (noise_i + 1j * noise_q).astype(np.complex64)
            signal += noise

        return signal

    def generate_pulse_train(
        self,
        num_pulses: int = 10,
        pri_us: float = 100.0,
        pw_us: float = 10.0,
        snr_db: float = 15.0,
        freq_offset_hz: float = 1_000_000.0,
        initial_delay_us: float = 20.0,
        amplitude: float = 1.0,
    ) -> Tuple[np.ndarray, List[GroundTruthPulse]]:
        """Generate a periodic or staggered pulse train scenario.

        Args:
            num_pulses: Number of pulses in the scenario.
            pri_us: Pulse Repetition Interval in microseconds.
            pw_us: Pulse Width in microseconds.
            snr_db: Signal to noise ratio in dB.
            freq_offset_hz: Baseband frequency offset in Hz.
            initial_delay_us: Start delay before first pulse in microseconds.
            amplitude: Peak linear amplitude.

        Returns:
            Tuple of (iq_samples, truth_pulses).
        """
        pri_samples = int(round(pri_us * 1e-6 * self.sample_rate_hz))
        pw_samples = int(round(pw_us * 1e-6 * self.sample_rate_hz))
        delay_samples = int(round(initial_delay_us * 1e-6 * self.sample_rate_hz))

        total_samples = delay_samples + num_pulses * pri_samples + int(50e-6 * self.sample_rate_hz)
        truth_pulses: List[GroundTruthPulse] = []

        for i in range(num_pulses):
            start = delay_samples + i * pri_samples
            end = start + pw_samples - 1
            truth_pulses.append(
                GroundTruthPulse(
                    truth_id=i,
                    start_sample=start,
                    end_sample=end,
                    peak_amplitude=amplitude,
                    frequency_offset_hz=freq_offset_hz,
                )
            )

        iq = self.generate_iq_from_truth(truth_pulses, total_samples, snr_db=snr_db)
        return iq, truth_pulses

    def from_pulse_records(
        self,
        records: Sequence[PulseRecord],
        snr_db: float = 15.0,
        margin_us: float = 50.0,
    ) -> Tuple[np.ndarray, List[GroundTruthPulse]]:
        """Convert cognitive EW scenario PulseRecords into continuous IQ and GroundTruthPulses.

        Args:
            records: List of PulseRecord instances.
            snr_db: Complex noise SNR level.
            margin_us: Safety margin after last pulse.

        Returns:
            Tuple of (iq_samples, truth_pulses).
        """
        if not records:
            return np.zeros(1024, dtype=np.complex64), []

        sorted_records = sorted(records, key=lambda r: r.toa_us)
        max_time_us = max(r.toa_us + r.pulse_width_us for r in sorted_records) + margin_us
        total_samples = int(np.ceil(max_time_us * 1e-6 * self.sample_rate_hz))

        truth_pulses: List[GroundTruthPulse] = []
        for idx, rec in enumerate(sorted_records):
            start = int(round(rec.toa_us * 1e-6 * self.sample_rate_hz))
            dur = max(1, int(round(rec.pulse_width_us * 1e-6 * self.sample_rate_hz)))
            end = start + dur - 1

            freq_offset = (rec.frequency_mhz * 1e6) - self.center_frequency_hz
            # Amplitude in linear scale: rec.amplitude_db (default -30 to 0 dBFS)
            amp_linear = float(10.0 ** (rec.amplitude_db / 20.0))

            truth_pulses.append(
                GroundTruthPulse(
                    truth_id=idx,
                    start_sample=start,
                    end_sample=end,
                    peak_amplitude=amp_linear,
                    frequency_offset_hz=freq_offset,
                )
            )

        iq = self.generate_iq_from_truth(truth_pulses, total_samples, snr_db=snr_db)
        return iq, truth_pulses


class ValidationRunner:
    """Executes end-to-end streaming validation runs and verifies exit criteria."""

    def __init__(
        self,
        config: Optional[ReceiverConfig] = None,
        chunk_size: int = 2048,
    ) -> None:
        self.config = config or ReceiverConfig()
        self.chunk_size = chunk_size

    def chunk_stream(
        self,
        iq_samples: np.ndarray,
    ) -> Generator[ReceiverInput, None, None]:
        """Slice contiguous IQ samples into streaming ReceiverInput chunks."""
        total_samples = len(iq_samples)
        sample_rate = self.config.sample_rate_hz
        center_freq = self.config.center_frequency_hz
        chunk_size = self.chunk_size

        chunk_idx = 0
        for offset in range(0, total_samples, chunk_size):
            chunk_data = iq_samples[offset : offset + chunk_size]
            timestamp_us = (offset / sample_rate) * 1e6

            yield ReceiverInput(
                iq_samples=chunk_data,
                sample_rate_hz=sample_rate,
                center_frequency_hz=center_freq,
                timestamp_us=timestamp_us,
                chunk_index=chunk_idx,
                global_sample_offset=offset,
            )
            chunk_idx += 1

    def run_scenario(
        self,
        iq_samples: np.ndarray,
        truth_pulses: List[GroundTruthPulse],
        tolerance_samples: int = 30,
        min_iou: float = 0.3,
    ) -> ValidationResult:
        """Feed IQ chunk stream through Frontend and Detector, evaluate against truth.

        Args:
            iq_samples: Contiguous synthetic or recorded IQ samples.
            truth_pulses: Ground truth pulse boundaries.
            tolerance_samples: Latency tolerance for boundary matching.
            min_iou: Temporal intersection over union threshold.

        Returns:
            ValidationResult containing full metrics and output pulses.
        """
        frontend = ReceiverFrontend(self.config)
        detector = PulseDetector(self.config)

        all_detected: List[DetectedPulse] = []
        chunk_count = 0

        t0 = time.perf_counter()

        for chunk in self.chunk_stream(iq_samples):
            frontend_out = frontend.process_chunk(chunk)
            detected = detector.process_chunk(frontend_out)
            all_detected.extend(detected)
            chunk_count += 1

        # Flush any in-progress pulse at end of stream
        flushed = detector.flush()
        all_detected.extend(flushed)

        elapsed = time.perf_counter() - t0

        metrics = evaluate_detections(
            truth_pulses=truth_pulses,
            detected_pulses=all_detected,
            tolerance_samples=tolerance_samples,
            min_iou=min_iou,
        )

        return ValidationResult(
            metrics=metrics,
            detected_pulses=all_detected,
            truth_pulses=truth_pulses,
            total_samples=len(iq_samples),
            total_chunks=chunk_count,
            elapsed_time_s=elapsed,
            metadata={
                "sample_rate_hz": self.config.sample_rate_hz,
                "center_frequency_hz": self.config.center_frequency_hz,
                "chunk_size": self.chunk_size,
            },
        )

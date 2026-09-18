"""Frequency Hopping Emitter Flowgraph / Generator.

Matches the FreqAgileEmitter threat model required by SIH 2026 Phase 1 (T1.2, T1.3):
Hops between frequency channels on an agility schedule unknown to the ES receiver.
Outputs complex IQ samples locally or over a ZeroMQ PUB socket.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

import numpy as np


class FrequencyHoppingEmitter:
    """Simulates a frequency-hopping radar or comms emitter.

    Parameters
    ----------
    samp_rate : float
        Sampling frequency in S/s (default 2 MS/s).
    hop_frequencies : sequence of float
        Baseband center frequencies (Hz) for each hop channel.
    hop_interval_s : float
        Dwell time in seconds spent on each frequency before hopping.
    pulse_width_s : float
        Pulse duration in seconds (for pulsed agile radar).
    pri_s : float
        Pulse repetition interval in seconds.
    pattern : str
        Hop selection pattern: "fixed" (cyclic) or "random" (pseudo-random).
    amplitude : float
        Signal amplitude.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        samp_rate: float = 2.0e6,
        hop_frequencies: Optional[Sequence[float]] = None,
        hop_interval_s: float = 5.0e-4,  # 500 µs
        pulse_width_s: float = 1.0e-5,   # 10 µs
        pri_s: float = 1.0e-4,           # 100 µs
        pattern: str = "fixed",
        amplitude: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.samp_rate = float(samp_rate)
        self.hop_frequencies = (
            list(hop_frequencies)
            if hop_frequencies is not None
            else [-600e3, -200e3, 200e3, 600e3]
        )
        self.hop_interval_s = float(hop_interval_s)
        self.pulse_width_s = float(pulse_width_s)
        self.pri_s = float(pri_s)
        self.pattern = str(pattern).lower()
        self.amplitude = float(amplitude)
        self.seed = seed

        self._hop_samples = max(1, int(round(self.hop_interval_s * self.samp_rate)))
        self._pulse_samples = max(1, int(round(self.pulse_width_s * self.samp_rate)))
        self._pri_samples = max(self._pulse_samples + 1, int(round(self.pri_s * self.samp_rate)))
        self._rng = np.random.default_rng(seed)

    def generate_hop_sequence(self, n_hops: int) -> List[float]:
        """Generate the list of hop frequencies for n_hops."""
        if self.pattern == "fixed":
            return [
                self.hop_frequencies[i % len(self.hop_frequencies)]
                for i in range(n_hops)
            ]
        else:
            return [
                float(self._rng.choice(self.hop_frequencies))
                for _ in range(n_hops)
            ]

    def generate_samples(self, duration_s: float) -> np.ndarray:
        """Synthesize complex baseband IQ samples for the given duration."""
        total_samples = int(round(duration_s * self.samp_rate))
        if total_samples <= 0:
            return np.zeros(0, dtype=np.complex64)

        n_hops = int(np.ceil(total_samples / self._hop_samples))
        hop_freqs = self.generate_hop_sequence(n_hops)

        iq_out = np.zeros(total_samples, dtype=np.complex64)
        for h in range(n_hops):
            start_s = h * self._hop_samples
            end_s = min(total_samples, (h + 1) * self._hop_samples)
            if start_s >= total_samples:
                break

            n_chunk = end_s - start_s
            t_chunk = np.arange(start_s, end_s) / self.samp_rate
            freq = hop_freqs[h]

            # Modulate carrier
            carrier = np.exp(1j * 2 * np.pi * freq * t_chunk).astype(np.complex64)

            # Pulse gate according to PRI and pulse width
            sample_indices = np.arange(start_s, end_s)
            slot_in_pri = sample_indices % self._pri_samples
            pulse_mask = (slot_in_pri < self._pulse_samples).astype(np.float32)

            iq_out[start_s:end_s] = self.amplitude * carrier * pulse_mask

        return iq_out

    def run_zmq_publisher(
        self,
        endpoint: str = "tcp://*:55555",
        chunk_duration_s: float = 0.01,
        max_chunks: Optional[int] = None,
    ) -> None:
        """Publish real-time IQ stream over ZeroMQ PUB socket."""
        try:
            import zmq
        except ImportError:
            raise RuntimeError("pyzmq is required to publish over ZMQ.")

        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.bind(endpoint)
        print(f"[FrequencyHoppingEmitter] Publishing IQ to {endpoint}...")

        chunk_idx = 0
        try:
            while max_chunks is None or chunk_idx < max_chunks:
                iq = self.generate_samples(chunk_duration_s)
                socket.send(iq.tobytes())
                chunk_idx += 1
                time.sleep(chunk_duration_s)
        finally:
            socket.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency-Hopping Emitter Flowgraph / Generator")
    parser.add_argument("--samp-rate", type=float, default=2e6, help="Sampling rate in S/s")
    parser.add_argument("--hop-interval", type=float, default=1e-3, help="Hop interval in seconds")
    parser.add_argument("--pattern", type=str, choices=["fixed", "random"], default="fixed", help="Agility pattern")
    parser.add_argument("--zmq-endpoint", type=str, default="tcp://*:55555", help="ZMQ PUB endpoint")
    parser.add_argument("--max-chunks", type=int, default=10, help="Number of chunks to publish")
    args = parser.parse_args()

    emitter = FrequencyHoppingEmitter(
        samp_rate=args.samp_rate,
        hop_interval_s=args.hop_interval,
        pattern=args.pattern,
    )
    emitter.run_zmq_publisher(endpoint=args.zmq_endpoint, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()

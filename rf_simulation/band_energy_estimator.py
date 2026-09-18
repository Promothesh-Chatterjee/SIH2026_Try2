"""Band Energy Estimator for RF Simulation.

Audited and integrated as part of Phase 1 (T1.3):
Converts raw complex IQ samples (from numpy arrays, binary files, or GNU Radio ZMQ stream)
into per-band, per-time-slot energy measurements (|IQ|^2 over each band's frequency slice).

Pure numpy implementation with zero GNU Radio dependency.
Optional ZMQ streaming bridge via pyzmq.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BandEnergyEstimator:
    """Estimates energy per frequency band from complex IQ samples.

    Parameters
    ----------
    n_bands : int
        Number of frequency sub-bands across the receiver instantaneous bandwidth.
        Default is 32.
    sample_rate : float
        Sampling rate in S/s. Default is 2.0e6 (2 MS/s).
    samples_per_slot : int
        Number of IQ samples representing one dwell / time slot.
        Default is 200 (100 µs at 2 MS/s).
    """

    def __init__(
        self,
        n_bands: int = 32,
        sample_rate: float = 2.0e6,
        samples_per_slot: int = 200,
    ) -> None:
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if samples_per_slot <= 0:
            raise ValueError("samples_per_slot must be positive")

        self.n_bands = int(n_bands)
        self.sample_rate = float(sample_rate)
        self.samples_per_slot = int(samples_per_slot)

    def compute_energy_vector(self, iq_slot: np.ndarray) -> np.ndarray:
        """Compute the per-band energy measurement |IQ|^2 for a single time slot.

        Parameters
        ----------
        iq_slot : np.ndarray
            1D complex IQ samples of length >= n_bands.

        Returns
        -------
        energy : np.ndarray
            1D float32 array of shape (n_bands,) containing the integrated power
            in each frequency band.
        """
        iq_slot = np.asarray(iq_slot)
        if iq_slot.size == 0:
            return np.zeros(self.n_bands, dtype=np.float32)

        # FFT across the window to separate frequency channels
        n_samples = len(iq_slot)
        fft_vals = np.fft.fftshift(np.fft.fft(iq_slot))
        psd = (np.abs(fft_vals) ** 2) / float(n_samples)

        # Partition PSD into n_bands contiguous slices
        energy = np.zeros(self.n_bands, dtype=np.float32)
        bin_edges = np.linspace(0, n_samples, self.n_bands + 1, dtype=int)

        for b in range(self.n_bands):
            start_idx = bin_edges[b]
            end_idx = max(start_idx + 1, bin_edges[b + 1])
            slice_power = np.mean(psd[start_idx:end_idx])
            energy[b] = np.float32(slice_power)

        return energy

    def process_stream(self, iq_samples: np.ndarray) -> np.ndarray:
        """Process a continuous stream of IQ samples into a time-frequency energy matrix.

        Parameters
        ----------
        iq_samples : np.ndarray
            1D complex IQ array.

        Returns
        -------
        matrix : np.ndarray
            2D float32 array of shape (n_slots, n_bands).
        """
        iq_samples = np.asarray(iq_samples)
        total_slots = len(iq_samples) // self.samples_per_slot
        if total_slots == 0:
            return np.zeros((0, self.n_bands), dtype=np.float32)

        matrix = np.zeros((total_slots, self.n_bands), dtype=np.float32)
        for t in range(total_slots):
            slot = iq_samples[t * self.samples_per_slot : (t + 1) * self.samples_per_slot]
            matrix[t, :] = self.compute_energy_vector(slot)

        return matrix

    def run_zmq_bridge(
        self,
        sub_endpoint: str = "tcp://127.0.0.1:55555",
        pub_endpoint: str = "tcp://127.0.0.1:55556",
        max_messages: Optional[int] = None,
        timeout_ms: int = 2000,
    ) -> None:
        """Bridge GNU Radio raw IQ stream into energy measurement ZMQ stream.

        Subscribes to complex64 IQ on sub_endpoint and publishes float32 energy vectors
        to pub_endpoint.
        """
        try:
            import zmq
        except ImportError:
            raise RuntimeError("pyzmq is required to run the live ZMQ bridge.")

        context = zmq.Context.instance()
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect(sub_endpoint)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        sub_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        pub_socket = context.socket(zmq.PUB)
        pub_socket.bind(pub_endpoint)

        msg_count = 0
        try:
            while max_messages is None or msg_count < max_messages:
                try:
                    raw_bytes = sub_socket.recv()
                except zmq.Again:
                    continue

                iq_chunk = np.frombuffer(raw_bytes, dtype=np.complex64)
                if len(iq_chunk) < self.samples_per_slot:
                    continue

                energy = self.compute_energy_vector(iq_chunk[: self.samples_per_slot])
                pub_socket.send(energy.tobytes())
                msg_count += 1
        finally:
            sub_socket.close()
            pub_socket.close()


def main() -> None:
    """CLI entrypoint for running the band energy estimator."""
    parser = argparse.ArgumentParser(description="RF Simulation Band Energy Estimator")
    parser.add_argument("--n-bands", type=int, default=32, help="Number of frequency bands")
    parser.add_argument("--sample-rate", type=float, default=2e6, help="Sampling rate in S/s")
    parser.add_argument("--samples-per-slot", type=int, default=200, help="Samples per time slot")
    parser.add_argument("--zmq-sub", type=str, default="tcp://127.0.0.1:55555", help="ZMQ SUB address")
    parser.add_argument("--zmq-pub", type=str, default="tcp://127.0.0.1:55556", help="ZMQ PUB address")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum messages to bridge")
    args = parser.parse_args()

    estimator = BandEnergyEstimator(
        n_bands=args.n_bands,
        sample_rate=args.sample_rate,
        samples_per_slot=args.samples_per_slot,
    )
    print(
        f"[BandEnergyEstimator] Bridging IQ from {args.zmq_sub} to energy vectors at {args.zmq_pub} ({args.n_bands} bands)..."
    )
    estimator.run_zmq_bridge(
        sub_endpoint=args.zmq_sub,
        pub_endpoint=args.zmq_pub,
        max_messages=args.max_messages,
    )


if __name__ == "__main__":
    main()

"""Unit tests for Receiver Front End Digital SOS Filters."""

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.filters import DigitalFrontendFilter


def test_filter_passband_and_stopband():
    """Verify digital filter passes in-band frequencies and rejects out-of-band signals."""
    sample_rate = 20_000_000.0
    config = ReceiverConfig(
        sample_rate_hz=sample_rate,
        bandwidth_hz=6_000_000.0,
        filter_type="lowpass",
        filter_order=4,
    )
    dfilter = DigitalFrontendFilter(config)

    t = np.arange(4096) / sample_rate

    # In-band tone at 1 MHz
    in_band = np.exp(1j * 2 * np.pi * 1_000_000 * t).astype(np.complex64)
    out_in_band = dfilter.filter_chunk(in_band)
    # Ignore initial filter transient
    in_band_gain = np.mean(np.abs(out_in_band[500:]))
    assert 0.85 <= in_band_gain <= 1.15, f"In-band gain {in_band_gain} outside expected [0.85, 1.15]"

    dfilter.reset()

    # Out-of-band tone at 8 MHz (cutoff is 3 MHz for lowpass with bw=6MHz)
    out_band = np.exp(1j * 2 * np.pi * 8_000_000 * t).astype(np.complex64)
    out_out_band = dfilter.filter_chunk(out_band)
    out_band_gain = np.mean(np.abs(out_out_band[500:]))
    assert out_band_gain < 0.15, f"Out-of-band attenuation insufficient, gain: {out_band_gain}"


def test_filter_chunk_continuity():
    """Verify filtering a signal across chunks produces continuous output matching single-batch filtering."""
    sample_rate = 20_000_000.0
    config = ReceiverConfig(
        sample_rate_hz=sample_rate,
        bandwidth_hz=8_000_000.0,
        filter_type="bandpass",
        filter_order=4,
    )

    t = np.arange(4096) / sample_rate
    signal = (
        np.exp(1j * 2 * np.pi * 1_500_000 * t) +
        0.5 * np.exp(1j * 2 * np.pi * 3_000_000 * t)
    ).astype(np.complex64)

    # 1. Single batch filter
    single_filter = DigitalFrontendFilter(config)
    y_single = single_filter.filter_chunk(signal)

    # 2. Chunk-by-chunk filter
    chunk_filter = DigitalFrontendFilter(config)
    chunk_size = 1024
    chunks = [signal[i : i + chunk_size] for i in range(0, len(signal), chunk_size)]
    y_chunks = [chunk_filter.filter_chunk(c) for c in chunks]
    y_streamed = np.concatenate(y_chunks)

    # Should match with machine precision across chunk boundaries
    max_diff = np.max(np.abs(y_single - y_streamed))
    assert max_diff < 1e-6, f"Filter chunk boundary mismatch: max diff {max_diff}"


def test_filter_state_save_restore():
    """Verify filter SOS state can be serialized and restored."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)
    f1 = DigitalFrontendFilter(config)

    data1 = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    data2 = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)

    _ = f1.filter_chunk(data1)
    state = f1.save_state()
    out1 = f1.filter_chunk(data2)

    # Restore in new filter
    f2 = DigitalFrontendFilter(config)
    f2.restore_state(state)
    out2 = f2.filter_chunk(data2)

    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)

"""Unit tests for AmplitudeExtractor and IQHistoryBuffer."""

import numpy as np
import pytest
from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.parameter_extractor.amplitude_extractor import (
    AmplitudeExtractor,
    IQHistoryBuffer,
)


def test_iq_history_buffer_global_coordinate_addressing():
    """Verify IQHistoryBuffer stores global sample bounds and correctly retrieves slices."""
    buf = IQHistoryBuffer(max_capacity=1024)

    # Chunk 0: samples 1000..1511 (512 samples)
    chunk0 = np.full(512, 0.5 + 0.5j, dtype=np.complex64)
    buf.append_chunk(chunk0, global_sample_offset=1000)

    assert buf.start_global_sample == 1000
    assert buf.end_global_sample == 1511

    # Retrieve slice in global coordinates [1050..1099] (50 samples)
    slice_a = buf.get_slice(1050, 1099)
    assert len(slice_a) == 50
    np.testing.assert_allclose(slice_a, 0.5 + 0.5j)

    # Chunk 1: samples 1512..2023 (512 samples)
    chunk1 = np.full(512, 0.8 + 0.0j, dtype=np.complex64)
    buf.append_chunk(chunk1, global_sample_offset=1512)

    assert buf.start_global_sample == 1000
    assert buf.end_global_sample == 2023

    # Retrieve slice straddling chunk 0 and chunk 1 [1500..1530] (31 samples)
    slice_straddle = buf.get_slice(1500, 1530)
    assert len(slice_straddle) == 31
    # First 12 samples are chunk0 (1500..1511), next 19 are chunk1 (1512..1530)
    np.testing.assert_allclose(slice_straddle[:12], 0.5 + 0.5j)
    np.testing.assert_allclose(slice_straddle[12:], 0.8 + 0.0j)


def test_iq_history_buffer_capacity_eviction():
    """Verify buffer bounds slide correctly when capacity is exceeded."""
    buf = IQHistoryBuffer(max_capacity=500)

    # Append 400 samples starting at global offset 10,000
    chunk1 = np.ones(400, dtype=np.complex64)
    buf.append_chunk(chunk1, global_sample_offset=10_000)
    assert buf.start_global_sample == 10_000
    assert buf.end_global_sample == 10_399

    # Append another 300 samples starting at global offset 10,400 (total 700 > capacity 500)
    chunk2 = np.ones(300, dtype=np.complex64) * 2.0
    buf.append_chunk(chunk2, global_sample_offset=10_400)

    # Buffer must retain exactly 500 samples, start should advance by 200 to 10,200
    assert len(buf._samples) == 500
    assert buf.start_global_sample == 10_200
    assert buf.end_global_sample == 10_699

    # Requesting evicted region returns empty or clamped
    assert len(buf.get_slice(10_000, 10_100)) == 0


def test_amplitude_extraction_and_agc_compensation():
    """Verify amplitude dBFS computation and AGC gain compensation."""
    config = ReceiverConfig()
    extractor = AmplitudeExtractor(config)

    # Simulate chunk with peak 0.5 (-6.02 dBFS)
    iq_data = np.zeros(1000, dtype=np.complex64)
    iq_data[200:300] = 0.5 + 0.0j  # linear peak 0.5
    extractor.ingest_chunk(iq_data, global_sample_offset=0)

    pulse = DetectedPulse(
        pulse_id=1,
        global_start_sample=200,
        global_end_sample=299,
        peak_magnitude=0.5,
        confidence=1.0,
    )

    # Case A: AGC gain was 0 dB
    amp_db, peak = extractor.extract_amplitude(pulse, applied_gain_db=0.0)
    assert peak == pytest.approx(0.5, abs=1e-5)
    assert amp_db == pytest.approx(20.0 * np.log10(0.5), abs=0.05)  # ~ -6.02 dBFS

    # Case B: AGC attenuated a strong signal by 10 dB (applied_gain_db = -10.0 dB)
    # The true original signal power was 10 dB higher
    amp_db_comp, _ = extractor.extract_amplitude(pulse, applied_gain_db=-10.0)
    assert amp_db_comp == pytest.approx(-6.02 - (-10.0), abs=0.05)  # ~ +3.98 dBFS


def test_amplitude_extraction_log0_protection():
    """Verify zero-magnitude or sub-noise pulse slice does not produce NaN or -inf."""
    config = ReceiverConfig()
    extractor = AmplitudeExtractor(config)

    zero_iq = np.zeros(500, dtype=np.complex64)
    extractor.ingest_chunk(zero_iq, global_sample_offset=0)

    pulse = DetectedPulse(
        pulse_id=99,
        global_start_sample=50,
        global_end_sample=100,
        peak_magnitude=0.0,
        confidence=0.5,
    )
    amp_db, peak = extractor.extract_amplitude(pulse, applied_gain_db=0.0)

    assert not np.isnan(amp_db)
    assert not np.isinf(amp_db)
    assert amp_db <= -100.0

"""Unit tests for Automatic Gain Control (AGC)."""

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.agc import AutomaticGainControl


def test_agc_levels_weak_signal():
    """Verify AGC increases gain for low-amplitude continuous signals over streaming chunks."""
    config = ReceiverConfig(
        agc_enabled=True,
        agc_attack=0.1,
        agc_decay=0.05,
        agc_target_level=0.707,
    )
    agc = AutomaticGainControl(config)

    # Weak signal with magnitude 0.0707
    weak = np.full(1024, 0.05 + 0.05j, dtype=np.complex64)
    for _ in range(40):
        out, applied_gain = agc.process(weak)

    # Gain should have increased above 0 dB
    assert applied_gain > 0.0, f"Expected positive gain boost, got {applied_gain} dB"
    # Output envelope at end should be significantly higher than input
    assert np.mean(np.abs(out[-500:])) > 0.3


def test_agc_anti_clipping():
    """Verify AGC prevents clipping when overloaded with very strong signals."""
    config = ReceiverConfig(
        agc_enabled=True,
        agc_target_level=0.707,
    )
    agc = AutomaticGainControl(config)

    # Overload signal with magnitude 10.0
    strong = np.full(4096, 7.0 + 7.0j, dtype=np.complex64)
    out, applied_gain = agc.process(strong)

    # Max magnitude must NEVER exceed 1.0 (anti-clipping guard)
    max_mag = np.max(np.abs(out))
    assert max_mag <= 1.0, f"Signal clipped! Max magnitude was {max_mag}"
    # Gain should have been reduced (negative dB)
    assert applied_gain < 0.0


def test_agc_disabled_passthrough():
    """Verify AGC passes samples unchanged when agc_enabled is False."""
    config = ReceiverConfig(agc_enabled=False)
    agc = AutomaticGainControl(config)

    sig = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    out, gain_db = agc.process(sig)

    np.testing.assert_array_equal(out, sig)
    assert gain_db == 0.0


def test_agc_state_save_restore():
    """Verify AGC state serializes and restores faithfully."""
    config = ReceiverConfig(agc_enabled=True, agc_attack=0.05)
    agc1 = AutomaticGainControl(config)

    data1 = np.full(1024, 0.2 + 0.2j, dtype=np.complex64)
    data2 = np.full(1024, 0.5 + 0.5j, dtype=np.complex64)

    _ = agc1.process(data1)
    state = agc1.save_state()
    out1, gain1 = agc1.process(data2)

    agc2 = AutomaticGainControl(config)
    agc2.restore_state(state)
    out2, gain2 = agc2.process(data2)

    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)
    assert gain1 == pytest.approx(gain2, rel=1e-5)

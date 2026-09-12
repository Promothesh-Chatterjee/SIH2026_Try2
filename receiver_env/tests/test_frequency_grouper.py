"""Unit tests for Frequency Grouper."""

from __future__ import annotations

import pytest
from receiver_env.deinterleaver.frequency_grouper import FrequencyGrouper
from receiver_env.deinterleaver.models import DeinterleaverConfig


def test_frequency_grouper_adaptive_gates():
    config = DeinterleaverConfig(frequency_gate_mhz=0.5, min_frequency_gate_mhz=0.15)
    grouper = FrequencyGrouper(config)

    # Low confidence -> wide gate (0.5 MHz)
    gate_low = grouper.get_adaptive_gate(0.0)
    assert abs(gate_low - 0.5) < 1e-4

    # Medium confidence (0.5) -> gate = 0.5 * (1 - 0.25) = 0.375 MHz
    gate_med = grouper.get_adaptive_gate(0.5)
    assert abs(gate_med - 0.375) < 1e-4

    # High confidence (1.0) -> gate = 0.5 * (1 - 0.5) = 0.25 MHz
    gate_high = grouper.get_adaptive_gate(1.0)
    assert abs(gate_high - 0.25) < 1e-4


def test_frequency_grouper_compatibility():
    grouper = FrequencyGrouper()
    track_freq = 3000.0  # MHz

    # Inside gate
    assert grouper.is_compatible(3000.2, track_freq, 0.5) is True
    # Outside gate
    assert grouper.is_compatible(3001.0, track_freq, 0.5) is False


def test_frequency_grouper_score():
    grouper = FrequencyGrouper()
    track_freq = 3000.0

    # Exact match -> score = 1.0
    score_exact = grouper.compute_match_score(3000.0, track_freq, 0.8)
    assert abs(score_exact - 1.0) < 1e-4

    # Small offset -> score > 0.5
    score_close = grouper.compute_match_score(3000.1, track_freq, 0.8)
    assert 0.5 < score_close < 1.0

    # Large offset -> score = 0.0
    score_far = grouper.compute_match_score(3002.0, track_freq, 0.8)
    assert score_far == 0.0

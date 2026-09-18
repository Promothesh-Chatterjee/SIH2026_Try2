"""Shared pytest fixtures for ew_core test suite.

Provides common, deterministic fixtures for signal deinterleaving and RF environment tests.
"""

from typing import List
import pytest

from ew_core.deinterleaver.windowed_deinterleaver import (
    PulseDescriptorWord,
    make_synthetic_pdws,
)
from ew_core.environment.spectrum_env import SpectrumEnvironment


@pytest.fixture
def synthetic_pdw_stream() -> List[PulseDescriptorWord]:
    """Generates a deterministic list of PulseDescriptorWord objects for tests needing PDW inputs."""
    return make_synthetic_pdws(n_emitters=3, n_pulses=100, seed=42)


@pytest.fixture
def small_env() -> SpectrumEnvironment:
    """Returns a lightweight SpectrumEnvironment(n_bands=8, t_steps=50) for fast unit testing."""
    return SpectrumEnvironment(n_bands=8, t_steps=50)

"""Unit tests for PRI and Jitter Estimator."""

from __future__ import annotations

import numpy as np
import pytest
from receiver_env.deinterleaver.pri_estimator import PRIEstimator


def test_pri_estimator_stable():
    estimator = PRIEstimator()
    true_pri = 100.0  # us
    toas = [10.0 + i * true_pri for i in range(50)]
    pri, jitter, std, conf = estimator.estimate(toas)
    assert abs(pri - true_pri) < 0.5
    assert jitter < 1.0
    assert conf > 0.85


def test_pri_estimator_jittered():
    estimator = PRIEstimator()
    rng = np.random.default_rng(42)
    true_pri = 150.0  # us
    # 10% uniform jitter
    jitters = rng.uniform(-15.0, 15.0, size=60)
    toas = [10.0]
    for j in jitters:
        toas.append(toas[-1] + true_pri + j)

    pri, jitter, std, conf = estimator.estimate(toas)
    assert abs(pri - true_pri) < 5.0
    assert 3.0 < jitter < 15.0
    assert conf > 0.60


def test_pri_estimator_dropped_pulses():
    estimator = PRIEstimator()
    rng = np.random.default_rng(123)
    true_pri = 80.0  # us
    # Drop 20% of pulses randomly
    all_toas = [5.0 + i * true_pri for i in range(80)]
    retained_toas = [t for t in all_toas if rng.random() > 0.20]

    pri, jitter, std, conf = estimator.estimate(retained_toas)
    assert abs(pri - true_pri) < 2.0
    assert conf > 0.70


def test_pri_estimator_insufficient_pulses():
    estimator = PRIEstimator()
    assert estimator.estimate([]) == (0.0, 0.0, 0.0, 0.0)
    assert estimator.estimate([10.0]) == (0.0, 0.0, 0.0, 0.0)
    assert estimator.estimate([10.0, 110.0]) == (0.0, 0.0, 0.0, 0.0)

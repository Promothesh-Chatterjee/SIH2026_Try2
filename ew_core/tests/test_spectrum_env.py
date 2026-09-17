"""Unit and property tests for Phase 1: SpectrumEnvironment and Emitter Models."""

import numpy as np
import pytest
import gymnasium

from ew_core.environment.emitter_models import (
    BaseEmitter,
    StaticEmitter,
    FreqAgileEmitter,
    PeriodicScanEmitter,
)
from ew_core.environment.spectrum_env import SpectrumEnvironment
from ew_core.environment.simulator_adapter import SimulatorAdapter


class TestEmitterModels:
    def test_static_emitter_fixed_band_and_duty_cycle(self):
        """StaticEmitter stays on its assigned band and follows its duty cycle."""
        emitter = StaticEmitter(band_idx=5, duty_cycle=0.5, pri=4, emitter_id=10)
        assert emitter.emitter_id == 10
        assert emitter.get_band(0) == 5
        assert emitter.get_band(100) == 5

        # pri=4, duty_cycle=0.5 -> active on slots 0, 1; inactive on 2, 3
        # for t in 0..7
        expected_active = [True, True, False, False, True, True, False, False]
        actual_active = [emitter.step(t) for t in range(8)]
        assert actual_active == expected_active

    def test_freq_agile_emitter_fixed_pattern(self):
        """FreqAgileEmitter with 'fixed' pattern cycles predictably through hop_set."""
        hop_set = [3, 7, 12]
        emitter = FreqAgileEmitter(
            hop_set=hop_set,
            hop_interval=10,
            pattern="fixed",
            duty_cycle=1.0,
            emitter_id=20,
        )
        assert emitter.emitter_id == 20

        # t in [0, 9] -> band 3
        # t in [10, 19] -> band 7
        # t in [20, 29] -> band 12
        # t in [30, 39] -> band 3
        assert emitter.get_band(0) == 3
        assert emitter.get_band(9) == 3
        assert emitter.get_band(10) == 7
        assert emitter.get_band(19) == 7
        assert emitter.get_band(20) == 12
        assert emitter.get_band(29) == 12
        assert emitter.get_band(30) == 3

        # Always transmitting when duty_cycle=1.0
        assert all(emitter.step(t) for t in range(40))

    def test_freq_agile_emitter_random_pattern(self):
        """FreqAgileEmitter with 'random' pattern only chooses bands from hop_set."""
        hop_set = [2, 14, 25]
        emitter = FreqAgileEmitter(
            hop_set=hop_set,
            hop_interval=5,
            pattern="random",
            duty_cycle=1.0,
            emitter_id=21,
            seed=42,
        )
        bands = [emitter.get_band(t) for t in range(0, 100, 5)]
        assert all(b in hop_set for b in bands)
        # Should visit more than 1 distinct band over 20 hops
        assert len(set(bands)) > 1

    def test_periodic_scan_emitter_pattern(self):
        """PeriodicScanEmitter sweeps bands in scan_pattern with specified dwell_time and scan_period."""
        scan_pattern = [1, 5, 9, 13]
        dwell_time = 5
        scan_period = 30  # 4 * 5 = 20 slots active, 10 slots inactive/off-target
        emitter = PeriodicScanEmitter(
            scan_period=scan_period,
            dwell_time=dwell_time,
            scan_pattern=scan_pattern,
            emitter_id=30,
        )
        assert emitter.emitter_id == 30

        # In cycle 0 (t = 0..29):
        # t in [0, 4]: band 1, active=True
        # t in [5, 9]: band 5, active=True
        # t in [10, 14]: band 9, active=True
        # t in [15, 19]: band 13, active=True
        # t in [20, 29]: off-target / quiet, active=False
        for t in range(5):
            assert emitter.get_band(t) == 1
            assert emitter.step(t) is True
        for t in range(5, 10):
            assert emitter.get_band(t) == 5
            assert emitter.step(t) is True
        for t in range(10, 15):
            assert emitter.get_band(t) == 9
            assert emitter.step(t) is True
        for t in range(15, 20):
            assert emitter.get_band(t) == 13
            assert emitter.step(t) is True
        for t in range(20, 30):
            assert emitter.step(t) is False

        # In cycle 1 (t = 30..59):
        # t = 30 starts new cycle at band 1
        assert emitter.get_band(30) == 1
        assert emitter.step(30) is True

    def test_no_prior_intel_privacy_contract(self):
        """Verify that emitter operational internals are strictly private and not exposed via public attributes."""
        static = StaticEmitter(band_idx=4, duty_cycle=0.8, pri=10)
        agile = FreqAgileEmitter(hop_set=[1, 2, 3], hop_interval=20, pattern="fixed")
        periodic = PeriodicScanEmitter(scan_period=40, dwell_time=10, scan_pattern=[0, 1, 2, 3])

        # Public attributes must NOT include the raw intelligence fields
        for obj, forbidden_attrs in [
            (static, ["band_idx", "pri", "duty_cycle"]),
            (agile, ["hop_set", "hop_interval", "pattern"]),
            (periodic, ["scan_period", "dwell_time", "scan_pattern"]),
        ]:
            for attr in forbidden_attrs:
                assert not hasattr(obj, attr), f"{type(obj).__name__} exposes public attribute '{attr}'! Violates no-prior-intel rule."
                # But private attribute must exist
                assert hasattr(obj, f"_{attr}"), f"{type(obj).__name__} must store '{attr}' as private '_{attr}'."


class TestSpectrumEnvironment:
    def test_gymnasium_api_compliance(self):
        """SpectrumEnvironment conforms strictly to Gymnasium Env API specifications."""
        env = SpectrumEnvironment(n_bands=32, t_steps=100)
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 32
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        assert env.observation_space.shape == (32,)

        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (32,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (32,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "hit" in info
        assert "active_bands" in info
        assert "emitter_ids" in info
        assert "step" in info

    def test_gymnasium_make_registration(self):
        """Verifies environment can be created via gymnasium.make('SmartScanEW-v0')."""
        env = gymnasium.make("SmartScanEW-v0", n_bands=16, t_steps=50)
        assert env.action_space.n == 16
        obs, info = env.reset()
        assert obs.shape == (16,)

    def test_truth_matrix_shape_and_accuracy(self):
        """env.get_truth_matrix() returns correct shape (n_bands, t_steps) with exact binary ground truth."""
        emitters = [
            StaticEmitter(band_idx=2, duty_cycle=1.0, pri=1, emitter_id=1),
            StaticEmitter(band_idx=5, duty_cycle=0.5, pri=2, emitter_id=2),
        ]
        env = SpectrumEnvironment(n_bands=16, t_steps=50, emitter_configs=emitters)
        env.reset(seed=42)

        truth = env.get_truth_matrix()
        assert isinstance(truth, np.ndarray)
        assert truth.shape == (16, 50)
        assert truth.dtype == bool

        # Band 2 should be active on every time slot t=0..49
        assert np.all(truth[2, :] == True)
        # Band 5 should be active on even slots (t % 2 == 0) and inactive on odd slots
        for t in range(50):
            assert truth[5, t] == (t % 2 == 0)
        # Unoccupied band 0 should be inactive across all time
        assert np.all(truth[0, :] == False)

    def test_interception_reward_and_info(self):
        """Tuning to an active band yields reward 1.0 and hit=True; silent band yields reward 0.0 and hit=False."""
        emitters = [
            StaticEmitter(band_idx=7, duty_cycle=1.0, pri=1, emitter_id=101)
        ]
        env = SpectrumEnvironment(n_bands=16, t_steps=20, emitter_configs=emitters)
        env.reset(seed=42)

        # Step 0: Tune to band 7 (active)
        obs, reward, terminated, truncated, info = env.step(7)
        assert reward == 1.0
        assert info["hit"] is True
        assert 7 in info["active_bands"]
        assert 101 in info["emitter_ids"]
        assert info["step"] == 0

        # Step 1: Tune to band 0 (silent)
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0
        assert info["hit"] is False
        assert 7 in info["active_bands"]
        assert 101 in info["emitter_ids"]
        assert info["step"] == 1

    def test_episode_termination_and_logging(self):
        """Episode terminates when t reaches t_steps; episode log accurately records progress."""
        t_steps = 10
        env = SpectrumEnvironment(n_bands=8, t_steps=t_steps)
        obs, info = env.reset(seed=1)
        log = env.init_episode_log()

        terminated = False
        step_count = 0
        while not terminated:
            action = step_count % 8
            obs, reward, terminated, truncated, info = env.step(action)
            env.update_episode_log(log, action, reward, info)
            step_count += 1

        assert step_count == t_steps
        assert terminated is True
        assert len(log["hits"]) == t_steps
        assert len(log["chosen_bands"]) == t_steps
        assert len(log["active_bands_per_step"]) == t_steps


class TestSimulatorAdapter:
    def test_simulator_adapter_python_mode(self):
        """SimulatorAdapter in 'python' mode creates and steps the environment seamlessly."""
        adapter = SimulatorAdapter(mode="python", n_bands=16, t_steps=20)
        env = adapter.get_env()
        assert env is not None
        obs, info = env.reset(seed=10)
        assert obs.shape == (16,)
        obs, reward, terminated, truncated, info = env.step(3)
        assert "hit" in info

"""Test script to validate the CognitiveRFScanEnv with all new components."""
import sys
sys.path.insert(0, "C:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/cognitive_ew_smart_scan")

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import synthetic_records


def test_env():
    config = {
        "n_bands": 36,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 500.0,
        "frequency_step_mhz": 500.0,
        "detection_threshold_db": -140.0,
        "max_steps_per_episode": 10,
    }
    records = synthetic_records(n_pulses=100, n_emitters=3, seed=42)
    env = CognitiveRFScanEnv(config, records=records, seed=42)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Obs dim: {env.obs_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Band features: {env.band_features}")
    print(f"n_bands: {env.n_bands}")
    print(f"Semantic memory: {env.semantic_memory is not None}")
    print(f"Periodic interceptor: {env.periodic_interceptor is not None}")
    print(f"Emitter tracker: {env.emitter_tracker is not None}")

    # Test step
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(i % 36)
        print(f"Step {i}: reward={reward:.3f}, hit={info['hit']}, preemptive_band={info.get('preemptive_band')}")

    print("Environment validation passed!")
    return True


if __name__ == "__main__":
    test_env()
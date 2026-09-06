import sys
sys.path.insert(0, '.')
import numpy as np
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.data.scenario_generator import ScenarioSource
import yaml

# Load config
with open('configs/training_config.yaml') as f:
    train_cfg = yaml.safe_load(f)
with open('configs/model_config.yaml') as f:
    model_cfg = yaml.safe_load(f)

env_cfg = train_cfg.get('environment', {})
seed = train_cfg.get('seed', 42)

# Build scenario source (world/stare mode)
data_dir = 'data'
subset = train_cfg.get('subset', 'train')
world_mode = train_cfg.get('world_mode', 'stare')

train_source = ScenarioSource(
    data_root=data_dir,
    mode='stare',
    subset=subset,
    freq_min_mhz=float(env_cfg.get('freq_min_mhz', 0.0)),
    freq_max_mhz=float(env_cfg.get('freq_max_mhz', 18000.0)),
    time_horizon_us=float(env_cfg.get('time_horizon_us', 0.0)) or None,
    max_pulses=int(env_cfg.get('max_pulses', 50000)),
    seed=seed,
    source_type='world',
    allow_synthetic_fallback=False,
)
print(f'ScenarioSource files: {len(train_source.files)}')
print(f'Eligible files: {len(train_source.eligible_files)}')
print(f'Source label: {train_source.source_label}')

# Build env
env = CognitiveRFScanEnv(
    env_cfg,
    records=None,
    seed=seed,
    records_provider=train_source.sample,
    deinterleaver_model=None,
    deinterleaver_config={},
)
obs, info = env.reset()
print(f'Observation dim: {env.obs_dim}')
print(f'Action space: {env.action_space.n}')
print(f'Belief enabled: {env.perception_enabled}')
print(f'First obs shape: {obs.shape}')

# Take a step
action = 0  # SHORT_DWELL on band 0
next_obs, reward, terminated, truncated, info = env.step(action)
print(f'Step - reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}')
print(f'Info keys: {list(info.keys())}')
print(f'Hit: {info.get("hit")}')
print(f'Entropy before: {info.get("entropy_before")}')
print(f'Entropy after: {info.get("entropy_after")}')
print(f'Information gain: {info.get("information_gain")}')

env.close()
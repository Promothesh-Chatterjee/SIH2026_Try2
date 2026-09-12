"""Generate Immutable Baseline Replay Reservoir.

Rolls out the frozen Gate-25k production baseline across diverse TSRD training scenarios
to capture 5,000 transitions (25 scenarios x 200 steps). This reservoir anchors continuation
training, preventing catastrophic forgetting and policy collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import numpy as np
import torch

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records, ScenarioSource
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_baseline_reservoir")


def generate_reservoir(
    baseline_ckpt: Path,
    output_path: Path,
    data_root: Path = Path("D:/TSRD"),
    n_scenarios: int = 25,
    steps_per_scenario: int = 200,
    seed: int = 42,
) -> Path:
    baseline_ckpt = baseline_ckpt.resolve()
    output_path = output_path.resolve()
    logger.info("Loading baseline from: %s", baseline_ckpt)

    device = torch.device("cpu")
    payload = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    state_dict = payload.get("online_drqn", payload.get("model", payload))

    model = DRQNScheduler(
        obs_dim=360,
        n_bands=36,
        n_actions=180,
        lstm_hidden=256,
        lstm_layers=2,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    train_source = ScenarioSource(data_root=data_root, mode="stare", subset="train", source_type="world")
    logger.info("Found %d eligible training scenarios", len(train_source.eligible_files))

    buffer = SequenceReplayBuffer(
        capacity=n_scenarios * steps_per_scenario + 1000,
        seq_len=16,
        obs_dim=360,
        burn_in=8,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    # Pick n_scenarios distinct files
    sampled_indices = rng.choice(len(train_source.eligible_files), size=min(n_scenarios, len(train_source.eligible_files)), replace=False)
    selected_files = [train_source.eligible_files[i] for i in sampled_indices]

    total_hits = 0
    total_steps = 0
    bands_visited = set()

    epsilon = 0.08  # Baseline exploration rate at Gate 25k

    for idx, path in enumerate(selected_files):
        scen_id = path.stem
        records = load_h5_records(path)
        env = CognitiveRFScanEnv(
            config={
                "n_bands": 36,
                "dwell_modes": 5,
                "features_per_band": 10,
                "ema_alpha": 0.30,
                "ema_alpha_miss_confirmed": 0.20,
                "semantic_memory_enabled": False,
            },
            records=records,
            seed=seed + idx,
        )
        obs, _ = env.reset()
        hx = None

        for step in range(steps_per_scenario):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q_vals, _, hx = model(obs_t, hx)
            
            if rng.random() < epsilon:
                action = int(rng.integers(0, 180))
            else:
                action = int(torch.argmax(q_vals.squeeze()).item())


            band = action // 5
            bands_visited.add(band)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated or (step == steps_per_scenario - 1))

            hit = bool(info.get("hit", False))
            if hit:
                total_hits += 1
            total_steps += 1

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                hit_prob=1.0 if hit else 0.0,
                intercept_time_us=float(info.get("intercept_time_us", np.nan)),
                scenario_id=scen_id,
            )
            obs = next_obs
            if done:
                break

    logger.info(
        "Reservoir collection complete: %d transitions across %d scenarios | Hit rate: %.2f%% | Unique bands: %d/36",
        total_steps, len(selected_files), (total_hits / max(1, total_steps)) * 100, len(bands_visited)
    )

    buffer.save_episodes(output_path)

    # Compute hash
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha256 = h.hexdigest()
    logger.info("Saved baseline reservoir to %s (SHA-256: %s)", output_path, sha256)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Baseline Replay Reservoir")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("cognitive_ew_smart_scan/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cognitive_ew_smart_scan/checkpoints/production_baseline/baseline_reservoir_5k.pkl"),
    )
    parser.add_argument("--scenarios", type=int, default=25)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_reservoir(
        baseline_ckpt=args.baseline,
        output_path=args.output,
        n_scenarios=args.scenarios,
        steps_per_scenario=args.steps,
        seed=args.seed,
    )

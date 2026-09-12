"""Diagnostic evaluation runner for Gate-25k frozen baseline without changing training behavior."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.training.val_set import FixedValidationSet
from cognitive_ew_smart_scan.src.training.diagnostics.action_tracker import ActionTracker
from cognitive_ew_smart_scan.src.training.diagnostics.q_telemetry import QTelemetry
from cognitive_ew_smart_scan.src.training.diagnostics.reward_tracker import RewardTracker
from cognitive_ew_smart_scan.src.training.diagnostics.scenario_tracker import ScenarioTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_baseline_diagnostics")


def evaluate_diagnostics(
    ckpt_path: Path,
    data_root: Path = Path("D:/TSRD"),
    steps_per_scenario: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    device = torch.device("cpu")

    # Load frozen checkpoint
    logger.info("Loading baseline checkpoint from %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    # Load validation scenario set
    from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records
    val_source = FixedValidationSet(data_root=data_root, n_files=10, seed=seed)

    action_tracker = ActionTracker(n_bands=36, n_modes=5, window_size=steps_per_scenario * len(val_source.files_used))
    reward_tracker = RewardTracker(gamma=0.99)
    scenario_tracker = ScenarioTracker()
    q_telemetry = QTelemetry(halt_ceiling=50.0)

    q_all_observed: list[float] = []

    for scen_idx, (path, scen_id, _) in enumerate(val_source.files_used, 1):
        logger.info("Evaluating baseline diagnostics on %s [%d/%d]...", scen_id, scen_idx, len(val_source.files_used))
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
            seed=seed,
        )
        obs, _ = env.reset()
        hx = None
        hits = 0

        for _ in range(steps_per_scenario):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q_vals, aux, hx = model(obs_t, hx)
                q_np = q_vals.squeeze().numpy()
                action = int(np.argmax(q_np))
                q_all_observed.append(float(np.max(q_np)))

            action_tracker.step(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_tracker.step(reward, info)
            hits += int(info.get("hit", False))
            obs = next_obs

        ir = hits / float(steps_per_scenario)
        scenario_tracker.record_scenario(
            scenario_id=scen_id,
            intercept_rate=ir,
            pd=float(info.get("decision_level_pd", 1.0) or 1.0),
            pfa=float(info.get("pfa", 0.0) or 0.0),
            distinct_bands=len(set(action_tracker.history)),
        )

    act_diag = action_tracker.get_diagnostics()
    rew_diag = reward_tracker.get_episode_summary()
    scen_diag = scenario_tracker.get_summary()

    q_tensor = torch.tensor(q_all_observed, dtype=torch.float32)
    q_eval = q_telemetry.evaluate_step(
        q_online=q_tensor,
        target_q_unclamped=q_tensor,
        td_errors=torch.zeros(1),
        online_model=model,
    )

    result = {
        "step": 25000,
        "top_band_dominance": act_diag["top_band_dominance"],
        "unique_bands": act_diag["unique_bands"],
        "band_entropy": act_diag["band_entropy"],
        "q_max": q_eval["q_max"],
        "target_q_unclamped_max": q_eval["target_q_unclamped_max"],
        "gradient_norm": 0.0,  # Inference evaluation has zero gradient norm
        "reward_components": rew_diag["components_raw"],
        "scenario_metrics": scen_diag["scenarios"],
        "summary": {
            "mean_ir": scen_diag["mean_ir"],
            "agile_ir": scen_diag["agile_ir"],
            "sparse_ir": scen_diag["sparse_ir"],
            "worst_case_ir": scen_diag["worst_case_ir"],
            "safety_halt": act_diag["safety_halt"] or q_eval["safety_halt"],
            "diagnostic_warning": act_diag["diagnostic_warning"] or q_eval["diagnostic_warning"],
        },
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline Diagnostics")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("cognitive_ew_smart_scan/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cognitive_ew_smart_scan/reports/baseline_gate25k_diagnostics.json"),
    )
    args = parser.parse_args()

    results = evaluate_diagnostics(args.ckpt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved baseline diagnostics to %s", args.output)
    print(json.dumps(results, indent=2))

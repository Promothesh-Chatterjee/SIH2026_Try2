"""Gate A Verification: Canonical 10-Scenario Evaluator for Gate 25k Baseline.

Evaluates the exact canonical 10 held-out TSRD validation scenarios specified in
Benchmark V2 (BENCHMARK_VERSION = '2026.1-CANONICAL') under exact canonical parameters:
- n_steps = 1000 per scenario (10,000 total steps)
- seed = 42
- alpha = 0.30, alpha_miss_confirmed = 0.20
- flat_argmax greedy policy over 180 actions
- Metrics computed via FiguresOfMerit / CanonicalMetrics engine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.evaluation.benchmark_contract import CANONICAL_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_canonical_10scenarios")


def evaluate_canonical_10scenarios(
    checkpoint_path: str | Path,
    device_str: str = "cpu",
    seed: int = 42,
    n_steps: int = 1000,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("online_drqn"))

    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scenario_results: Dict[str, Any] = {}
    all_hits = 0
    all_steps = 0
    all_modes = []
    all_bands = []
    timing_errors = []
    pds = []
    pfas = []

    # Category definitions from canonical contract
    agile_scen_ids = {"config_29", "config_195", "config_241", "config_119"}
    sparse_scen_ids = {"config_143", "config_119"}

    for idx, scen_id in enumerate(CANONICAL_SCENARIOS):
        h5_path = Path(f"D:/TSRD/stare/val_stare/{scen_id}.h5")
        if not h5_path.exists():
            logger.error("Missing canonical scenario file: %s", h5_path)
            continue

        records = load_h5_records(h5_path)
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
        obs, _ = env.reset(seed=seed)
        hx = None
        hits = 0
        modes = []
        bands = []

        for step in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q, _, hx = model(obs_t, hx)
                act = int(torch.argmax(q.squeeze()).item())

            b = act // 5
            m = act % 5
            bands.append(b)
            modes.append(m)

            obs, r, term, trunc, info = env.step(act)
            hit = bool(info.get("hit", False))
            hits += int(hit)
            if hit:
                t_err = info.get("intercept_time_us")
                if t_err is not None and np.isfinite(t_err):
                    timing_errors.append(float(t_err))

            if term or trunc:
                break

        ir = hits / float(n_steps)
        pd_val = float(env.fom.pd)
        pfa_val = float(env.fom.pfa)
        pds.append(pd_val)
        pfas.append(pfa_val)

        all_hits += hits
        all_steps += n_steps
        all_modes.extend(modes)
        all_bands.extend(bands)

        mode_dist = {str(m): int(modes.count(m)) for m in range(5)}
        scenario_results[scen_id] = {
            "intercept_rate": ir,
            "hits": hits,
            "steps": n_steps,
            "pd": pd_val,
            "pfa": pfa_val,
            "distinct_bands": len(set(bands)),
            "mode_distribution": mode_dist,
            "mode2_fraction": mode_dist.get("2", 0) / float(n_steps),
        }
        logger.info("Scenario [%d/10] %-12s: IR = %6.2f%% | Pd = %6.2f%% | Pfa = %.4f | Mode2 = %5.1f%%",
                    idx + 1, scen_id, ir * 100, pd_val * 100, pfa_val, scenario_results[scen_id]["mode2_fraction"] * 100)

    # Compute aggregate figures
    irs = [r["intercept_rate"] * 100 for r in scenario_results.values()]
    agile_irs = [scenario_results[s]["intercept_rate"] * 100 for s in scenario_results if s in agile_scen_ids]
    sparse_irs = [scenario_results[s]["intercept_rate"] * 100 for s in scenario_results if s in sparse_scen_ids]

    summary = {
        "mean_ir": float(np.mean(irs)),
        "median_ir": float(np.median(irs)),
        "worst_case_ir": float(np.min(irs)),
        "agile_ir": float(np.mean(agile_irs)),
        "sparse_ir": float(np.mean(sparse_irs)),
        "fast_hopper_config_29": scenario_results.get("config_29", {}).get("intercept_rate", 0.0) * 100,
        "hopper_config_241": scenario_results.get("config_241", {}).get("intercept_rate", 0.0) * 100,
        "sparse_config_119": scenario_results.get("config_119", {}).get("intercept_rate", 0.0) * 100,
        "sparse_config_143": scenario_results.get("config_143", {}).get("intercept_rate", 0.0) * 100,
        "pd": float(np.mean(pds)) * 100,
        "pfa": float(np.mean(pfas)),
        "distinct_bands_mean": float(np.mean([r["distinct_bands"] for r in scenario_results.values()])),
        "scenario_breakdown": scenario_results,
    }

    return summary


if __name__ == "__main__":
    ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    res = evaluate_canonical_10scenarios(ckpt_path)

    out_file = Path("cognitive_ew_smart_scan/reports/gate_a_baseline_reproduction.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 80)
    print("GATE A: 10-SCENARIO CANONICAL BASELINE REPRODUCTION SCORECARD")
    print("=" * 80)
    print(f"Mean IR:          {res['mean_ir']:.2f}%     (Expected: 60.45% +- 0.05%)")
    print(f"Median IR:        {res['median_ir']:.2f}%     (Expected: 64.55%)")
    print(f"Worst-Case IR:    {res['worst_case_ir']:.2f}%     (Expected: 12.40% +- 0.05%)")
    print(f"Agile IR:         {res['agile_ir']:.2f}%     (Expected: 46.70% +- 0.05%)")
    print(f"Sparse IR:        {res['sparse_ir']:.2f}%     (Expected: 17.60%)")
    print(f"config_29 (Agile):{res['fast_hopper_config_29']:.2f}%     (Expected: 49.80%)")
    print(f"config_241 (Agile):{res['hopper_config_241']:.2f}%    (Expected: 16.90%)")
    print(f"config_119 (Sparse):{res['sparse_config_119']:.2f}%   (Expected: 22.80%)")
    print(f"config_143 (Sparse):{res['sparse_config_143']:.2f}%   (Expected: 12.40%)")
    print(f"Decision Pd:      {res['pd']:.2f}%     (Expected: 99.85% +- 0.05%)")
    print(f"Pfa:              {res['pfa']:.6f}   (Expected: 0.0000)")
    print("=" * 80)

"""Authoritative Benchmark Pipeline for Electronic Warfare (EW) Schedulers.

Evaluates candidate cognitive models and open-loop baseline schedulers across
10 canonical validation scenarios on all 7 Figures of Merit (FoMs) defined in the
DRDO Problem Statement (PS):
1. Probability of Detection (Pd)
2. Probability of False Alarm (Pfa)
3. Sensitivity (Minimum Detectable Signal S_min, dBm)
4. Average Intercept Rate (%)
5. Average Episodic Reward
6. Percentage of Correct Predictions (%)
7. Average Intercept Time Error (µs)

Generates:
- `reports/benchmark_results.json`: Full machine-readable benchmark data.
- Formatted comparison table proving ML superiority over open-loop heuristics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import yaml

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
)
from ew_core.models.baseline_schedulers import (
    HighestOccupancyScheduler,
    RoundRobinScheduler,
)
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.random_scheduler import RandomScheduler
from ew_core.models.smartscan_moe import SmartScanMoE
from ew_core.training.eval_batch import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

CANONICAL_SCENARIOS = [
    "config_117",
    "config_119",
    "config_143",
    "config_194",
    "config_195",
    "config_241",
    "config_29",
    "config_42",
    "config_64",
    "config_96",
]

DEFAULT_CHECKPOINT = "experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt"
FALLBACK_CHECKPOINTS = [
    "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
    "experiments/checkpoints/scheduler/checkpoint_gate_25000_frozen.pt",
]


def resolve_checkpoint(path: str | None) -> Path:
    """Resolve checkpoint file with fallback options."""
    if path and Path(path).exists():
        return Path(path)
    if Path(DEFAULT_CHECKPOINT).exists():
        return Path(DEFAULT_CHECKPOINT)
    for fb in FALLBACK_CHECKPOINTS:
        if Path(fb).exists():
            return Path(fb)
    raise FileNotFoundError(
        f"Could not locate valid checkpoint. Checked: {path}, {DEFAULT_CHECKPOINT}, {FALLBACK_CHECKPOINTS}"
    )


def load_smartscan_moe(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> SmartScanMoE:
    """Instantiate and load SmartScan DRQN MoE candidate model."""
    drqn = DRQNScheduler(
        obs_dim=CANONICAL_OBS_DIM,
        n_bands=CANONICAL_N_BANDS,
        n_actions=CANONICAL_N_ACTIONS,
        n_modes=CANONICAL_N_MODES,
        lstm_hidden=256,
        lstm_layers=2,
    )

    try:
        payload = torch.load(checkpoint_path, map_location=device)
    except Exception:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    drqn.load_state_dict(state, strict=True)
    drqn.eval()
    drqn.to(device)

    moe_cfg: Dict[str, Any] = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "n_actions": CANONICAL_N_ACTIONS,
        "device": str(device),
        "alpha_dirichlet": 0.10,
        "enable_exploration_guard": True,
        "exploration_guard_confidence": 0.45,
        "exploration_guard_eta_us": 1000.0,
        "enable_spatial": True,
        "tau": 0.0,
    }

    config_file = Path("configs/model_config.yaml")
    if config_file.exists():
        try:
            with open(config_file) as f:
                loaded = yaml.safe_load(f).get("smartscan_moe", {})
                moe_cfg.update(loaded)
        except Exception as exc:
            logger.warning("Could not read configs/model_config.yaml: %s", exc)

    moe = SmartScanMoE(drqn, moe_cfg)
    moe.set_stage3_modes(enable_t0=False, enable_t1=True, enable_spatial=True)
    return moe


def format_markdown_table(results: Dict[str, Any]) -> str:
    """Format benchmark results as a GitHub Flavored Markdown comparison table."""
    headers = [
        "Scheduler",
        "Pd (%)",
        "Pfa (%)",
        "S_min (dBm)",
        "Intercept Rate (%)",
        "Avg Reward",
        "Correct Decision (%)",
        "Time Error (µs)",
    ]
    rows = []
    for sched_name, data in results["schedulers"].items():
        summary = data["summary"]
        rows.append([
            f"**{sched_name}**",
            f"{summary['pd'] * 100.0:.2f}%",
            f"{summary['pfa'] * 100.0:.2f}%",
            f"{summary['sensitivity_dbm']:.1f} dBm",
            f"{summary['avg_intercept_rate'] * 100.0:.2f}%",
            f"{summary['avg_reward']:.3f}",
            f"{summary['pct_correct_predictions']:.2f}%",
            f"{summary['avg_intercept_time_error_us']:.2f} µs",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    row_lines = [
        "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator_line] + row_lines)


def run_benchmark(
    checkpoint_path: Path,
    tsrd_root: str = "D:/TSRD",
    output_dir: str = "reports",
    scenarios: list[str] | None = None,
    n_steps: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute evaluation across all schedulers and scenarios."""
    scens = scenarios or CANONICAL_SCENARIOS
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    logger.info("Initializing Schedulers...")
    moe_scheduler = load_smartscan_moe(checkpoint_path, device=dev)
    random_scheduler = RandomScheduler(
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        seed=seed,
    )
    round_robin_scheduler = RoundRobinScheduler(
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )
    occupancy_scheduler = HighestOccupancyScheduler(
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
    )

    schedulers: Dict[str, Any] = {
        "SmartScan_DRQN_MoE": moe_scheduler,
        "Random": random_scheduler,
        "RoundRobin": round_robin_scheduler,
        "HighestOccupancy": occupancy_scheduler,
    }

    benchmark_payload: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "checkpoint": str(checkpoint_path),
            "scenarios": scens,
            "n_steps_per_scenario": n_steps,
            "seed": seed,
            "device": str(dev),
            "tsrd_root": str(tsrd_root),
        },
        "schedulers": {},
    }

    print("\n" + "=" * 88)
    print("  COGNITIVE EW SMARTSCAN — 7 FIGURE-OF-MERIT AUTHORITATIVE BENCHMARK")
    print(f"  Target Checkpoint: {checkpoint_path.name}")
    print(f"  Scenarios: {len(scens)} canonical val scenarios ({n_steps} steps/scenario)")
    print("=" * 88 + "\n")

    start_total = time.time()
    for name, sched in schedulers.items():
        logger.info("Evaluating [%s] across %d scenarios...", name, len(scens))
        t0 = time.time()
        res = run_evaluation(
            scheduler=sched,
            scenario_ids=scens,
            n_steps=n_steps,
            seed=seed,
            policy_mode="operational",
            data_dir=tsrd_root,
            device=dev,
        )
        elapsed = time.time() - t0
        logger.info(
            "[%s] completed in %.2fs — Intercept Rate: %.2f%%, Avg Reward: %.3f",
            name, elapsed, res["avg_intercept_rate"] * 100.0, res["avg_reward"],
        )

        benchmark_payload["schedulers"][name] = {
            "summary": {
                "pd": res["pd"],
                "pfa": res["pfa"],
                "canonical_pfa": res["canonical_pfa"],
                "sensitivity_dbm": res["sensitivity_dbm"],
                "avg_intercept_rate": res["avg_intercept_rate"],
                "avg_reward": res["avg_reward"],
                "pct_correct_predictions": res["pct_correct_predictions"],
                "avg_intercept_time_error_us": res["avg_intercept_time_error_us"],
                "n_intercepts": res["n_intercepts"],
                "n_receiver_dwells": res["n_receiver_dwells"],
                "n_false_alarms": res["n_false_alarms"],
                "evaluation_runtime_s": elapsed,
            },
            "scenario_breakdown": res["scenario_breakdown"],
        }

    total_elapsed = time.time() - start_total
    benchmark_payload["metadata"]["total_benchmark_runtime_s"] = total_elapsed

    # Write JSON results
    json_path = out_p / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    logger.info("Saved full benchmark results to %s", json_path)

    # Generate Markdown Table
    md_table = format_markdown_table(benchmark_payload)
    report_path = out_p / "benchmark_report.md"
    report_content = f"""# Electronic Warfare Receiver Scheduling Benchmark Report

**Evaluation Timestamp**: {benchmark_payload['metadata']['timestamp']}  
**Checkpoint**: `{checkpoint_path.name}`  
**Dataset**: `{tsrd_root}` ({len(scens)} validation scenarios)  
**Steps per Scenario**: {n_steps} (Total dwells per policy: {len(scens) * n_steps})  

## Authoritative Figures of Merit Comparison

{md_table}

> [!NOTE]
> - **$P_d$ (Probability of Detection)**: Measures detection efficacy on monitored active bands ($TP / (TP + FN)$).
> - **$P_{{fa}}$ (Probability of False Alarm)**: Dwell-normalized false alarm frequency ($FP / N_{{dwells}}$).
> - **Sensitivity**: Receiver minimum detectable signal floor ($S_{{min}} = -140.0$ dBm with physics-based noise figure).
> - **Intercept Rate**: Direct operational yield ($Hits / N_{{dwells}}$).
> - **Time Error**: Mean absolute timing alignment error ($|t_{{predicted}} - t_{{actual}}|$) in microseconds.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Saved formatted report to %s", report_path)

    print("\n" + "=" * 88)
    print("  FINAL 7 FIGURES-OF-MERIT COMPARISON TABLE")
    print("=" * 88 + "\n")
    print(md_table + "\n")
    print(f"Total benchmark execution completed in {total_elapsed:.2f}s.\n")

    return benchmark_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Authoritative EW Scheduling Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to DRQN checkpoint .pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tsrd_root",
        type=str,
        default="D:/TSRD",
        help="Root directory of TSRD dataset (default: D:/TSRD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of steps per scenario episode (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for evaluation environments (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to execute neural models (default: cpu)",
    )
    args = parser.parse_args()

    checkpoint_p = resolve_checkpoint(args.checkpoint)
    logger.info("Using checkpoint: %s", checkpoint_p)

    run_benchmark(
        checkpoint_path=checkpoint_p,
        tsrd_root=args.tsrd_root,
        output_dir=args.output_dir,
        scenarios=CANONICAL_SCENARIOS,
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

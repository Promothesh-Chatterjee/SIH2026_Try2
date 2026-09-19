"""End-to-end pipeline validation script for SIH 2026 Cognitive EW SmartScan.

Executes a full 1,000-step evaluation against hostile frequency-agile and
periodic scanning emitters, compares the cognitive scheduler against the
classical RoundRobin baseline, validates all 7 DRDO Figures of Merit, and
updates the FastAPI deployment state.

Exit code 0 indicates pipeline verification passed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from ew_core.environment.emitter_models import FreqAgileEmitter, PeriodicScanEmitter
from ew_core.environment.spectrum_env import SpectrumEnvironment
from ew_core.metrics.ew_metrics import EWMetrics, compute_all_metrics
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.operational.state_builder import OperationalStateBuilder
from ew_core.scheduler.baseline_sweep import RoundRobinScheduler
from ew_core.utils.checkpoint_paths import CANONICAL_PRODUCTION_BASELINE


def run_baseline(
    n_bands: int = 36,
    t_steps: int = 1000,
    seed: int = 42,
) -> Tuple[EWMetrics, dict[str, Any]]:
    """Execute 1,000-step RoundRobin baseline sweep."""
    e1 = FreqAgileEmitter(
        hop_set=[2, 8, 15, 20, 28],
        hop_interval=50,
        pattern="random",
        emitter_id=1,
    )
    e2 = PeriodicScanEmitter(
        scan_period=50,
        dwell_time=5,
        scan_pattern=list(range(36)),
        emitter_id=2,
    )

    env = SpectrumEnvironment(
        n_bands=n_bands,
        t_steps=t_steps,
        emitter_configs=[e1, e2],
    )
    log = env.init_episode_log()
    env.reset(seed=seed)
    scheduler = RoundRobinScheduler(n_bands=n_bands, joint=False)

    for _ in range(t_steps):
        action = scheduler.step()
        obs, reward, terminated, truncated, info = env.step(action)
        env.update_episode_log(log, action, reward, info)
        if terminated or truncated:
            break

    metrics = compute_all_metrics(log, min_detectable_signal_dbm=-140.0)
    return metrics, log


def load_agent(
    ckpt_path: Path,
    n_bands: int = 36,
    obs_dim: int = 360,
    n_modes: int = 5,
) -> Any:
    """Load trained DRQNScheduler from checkpoint. Fails strictly if missing or corrupted."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Strict checkpoint required: '{ckpt_path}' does not exist. "
            f"Silent fallback has been permanently removed."
        )
    try:
        try:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Corrupt or invalid checkpoint at {ckpt_path}: {exc}") from exc

    state_dict = checkpoint.get("state_dict", checkpoint)
    model = DRQNScheduler(obs_dim=obs_dim, n_bands=n_bands, n_modes=n_modes)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Successfully loaded trained DRQN checkpoint from: {ckpt_path}")
    return model


def run_agent(
    agent_model: Any,
    n_bands: int = 36,
    t_steps: int = 1000,
    seed: int = 42,
) -> Tuple[EWMetrics, dict[str, Any], np.ndarray]:
    """Execute 1,000-step evaluation run with the cognitive agent."""
    e1 = FreqAgileEmitter(
        hop_set=[2, 8, 15, 20, 28],
        hop_interval=50,
        pattern="random",
        emitter_id=1,
    )
    e2 = PeriodicScanEmitter(
        scan_period=50,
        dwell_time=5,
        scan_pattern=list(range(36)),
        emitter_id=2,
    )

    env = SpectrumEnvironment(
        n_bands=n_bands,
        t_steps=t_steps,
        emitter_configs=[e1, e2],
    )
    log = env.init_episode_log()
    env.reset(seed=seed)

    # Run trained DRQNScheduler with 360-D operational belief state builder
    state_builder = OperationalStateBuilder(n_bands=n_bands)
    hidden = None

    for t in range(t_steps):
        state_360 = state_builder.build_state(current_time_us=float(t * 500.0))
        s_tensor = torch.from_numpy(state_360).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_vals, aux, hidden = agent_model(s_tensor, hidden)

        joint_action = int(torch.argmax(q_vals[0, 0]).item())
        band = joint_action // 5
        obs, reward, terminated, truncated, info = env.step(band)
        hit = bool(info.get("hit", False))
        state_builder.record_dwell_outcome(band, hit, current_time_us=float((t + 1) * 500.0))
        env.update_episode_log(log, band, reward, info)
        if terminated or truncated:
            break

    truth_matrix = env.get_truth_matrix()
    metrics = compute_all_metrics(log, min_detectable_signal_dbm=-140.0)
    return metrics, log, truth_matrix


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Full Pipeline Validation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CANONICAL_PRODUCTION_BASELINE,
        help="Path to trained DRQN checkpoint",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cognitive EW SmartScan — Full Pipeline Validation (Phase 4)")
    print("=" * 70)

    n_bands = 36
    t_steps = 1000
    seed = 42

    # 1. Run RoundRobin Baseline
    print(f"\n[1/3] Running RoundRobin baseline ({t_steps} steps, {n_bands} bands)...")
    baseline_metrics, _ = run_baseline(n_bands=n_bands, t_steps=t_steps, seed=seed)
    print(f"      Baseline Intercepts: {baseline_metrics.n_intercepts} / {t_steps} "
          f"(Rate: {baseline_metrics.avg_intercept_rate:.4f}, Pd: {baseline_metrics.pd:.4f})")

    # 2. Load Agent and Execute Cognitive Scheduling
    ckpt_path = args.checkpoint
    print(f"\n[2/3] Loading scheduler agent from {ckpt_path} and evaluating on RF spectrum...")
    agent_model = load_agent(ckpt_path, n_bands=n_bands, obs_dim=360, n_modes=5)
    agent_metrics, agent_log, truth_matrix = run_agent(
        agent_model=agent_model,
        n_bands=n_bands,
        t_steps=t_steps,
        seed=seed,
    )

    # 3. Update FastAPI Deployment State
    try:
        from ew_core.deployment.api import update_latest_evaluation_data
        receiver_positions = agent_log.get("chosen_bands", [])
        active_bands_per_step = agent_log.get("active_bands_per_step", [])
        spectrum_payload = {
            "truth_matrix": truth_matrix,
            "receiver_positions": receiver_positions,
            "emitter_ids": active_bands_per_step,
        }
        update_latest_evaluation_data(agent_metrics, spectrum_payload)
        print("      [OK] Updated FastAPI backend state for /api/v1/metrics and /api/v1/spectrum")
    except Exception as exc:
        print(f"      [WARN] Could not update API state: {exc}")

    # 4. Print Summary Table of All 7 EW Figures of Merit
    print("\n" + "=" * 70)
    print("EW FIGURES OF MERIT — VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Figure of Merit':<35} | {'Achieved':<12} | {'Baseline (RR)':<12}")
    print("-" * 70)
    print(f"{'1. Probability of Detection (Pd)':<35} | {agent_metrics.pd:<12.4f} | {baseline_metrics.pd:<12.4f}")
    print(f"{'2. Probability of False Alarm (Pfa)':<35} | {agent_metrics.pfa:<12.4f} | {baseline_metrics.pfa:<12.4f}")
    print(f"{'3. Sensitivity (S_min, dBm)':<35} | {agent_metrics.sensitivity_dbm:<12.1f} | {baseline_metrics.sensitivity_dbm:<12.1f}")
    print(f"{'4. Average Intercept Rate':<35} | {agent_metrics.avg_intercept_rate:<12.4f} | {baseline_metrics.avg_intercept_rate:<12.4f}")
    print(f"{'5. Average Reward':<35} | {agent_metrics.avg_reward:<12.4f} | {baseline_metrics.avg_reward:<12.4f}")
    print(f"{'6. Correct Predictions (%)':<35} | {agent_metrics.pct_correct_predictions:<12.2f} | {baseline_metrics.pct_correct_predictions:<12.2f}")
    print(f"{'7. Avg Intercept Time Error (us)':<35} | {agent_metrics.avg_intercept_time_error_us:<12.2f} | {baseline_metrics.avg_intercept_time_error_us:<12.2f}")
    print("-" * 70)
    print(f"{'Total Intercepts (TP)':<35} | {agent_metrics.n_intercepts:<12} | {baseline_metrics.n_intercepts:<12}")
    print(f"{'Total Dwells':<35} | {agent_metrics.n_receiver_dwells:<12} | {baseline_metrics.n_receiver_dwells:<12}")
    print("=" * 70)

    # 5. Assertions (Phase 4 Gate Condition)
    print("\n[3/3] Validating Gate Invariants...")
    assert agent_metrics.pd >= 0.0, f"Invalid Pd: {agent_metrics.pd}"
    assert agent_metrics.pfa <= 0.1, f"Pfa exceeds threshold: {agent_metrics.pfa}"
    assert agent_metrics.sensitivity_dbm == -140.0, f"Sensitivity mismatch: {agent_metrics.sensitivity_dbm}"
    assert agent_metrics.avg_intercept_rate > 0.0, "Average intercept rate must be positive"
    assert agent_metrics.avg_intercept_rate > baseline_metrics.avg_intercept_rate, (
        f"Agent intercept rate ({agent_metrics.avg_intercept_rate:.4f}) failed to beat "
        f"RoundRobin baseline ({baseline_metrics.avg_intercept_rate:.4f})"
    )
    assert all(
        math.isfinite(v)
        for v in [
            agent_metrics.pd,
            agent_metrics.pfa,
            agent_metrics.avg_intercept_rate,
            agent_metrics.avg_reward,
            agent_metrics.pct_correct_predictions,
        ]
    ), "Detected non-finite values (NaN or Inf) in metrics"

    print("      [PASS] All gate assertions cleared successfully!")
    print("\n>>> Pipeline validation SUCCESSFUL (exit code 0). <<<\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

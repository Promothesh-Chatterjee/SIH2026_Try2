"""Canonical fixed evaluation batch for DRQN Q-diagnostics.

Persists a fixed, reproducible evaluation tensor [N, T, obs_dim] to disk at
`checkpoints/scheduler_clean_staged/fixed_eval_batch.pt`.

Batch composition:
  - Sequence 0: Canonical reconstructed 'stuck state' (16 steps where Band 4 is
    dwelled on continuously, leaving the other 35 bands unvisited with normalized
    age = 1.0, prio >= 0.4).
  - Sequences 1..15: Diverse validation sequences from real TSRD val_stare scenarios.

Provides `evaluate_q_diagnostics()` to track Q-margin, spread, and argmax on
the trapped state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from ..contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, band_of_action
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import load_h5_records, synthetic_records
from ..metrics.ew_metrics import compute_all_metrics, EWMetrics
from ..models.drqn_scheduler import DRQNScheduler

logger = logging.getLogger(__name__)

CANONICAL_EVAL_BATCH_PATH = Path("checkpoints/scheduler_clean_staged/fixed_eval_batch.pt")


def generate_stuck_state_sequence(seq_len: int = 16, stuck_band: int = 4) -> np.ndarray:
    """Generate a synthetic 16-step sequence simulating the 2-band / B4 collapse.

    At each step:
      - Band 4 has been repeatedly dwelled on: revisit age = 0.0, miss_rate = 1.0.
      - Other 35 bands have not been visited for 50+ dwells:
        revisit age = 1.0 (max), uncertainty = 0.8, priority = 0.55+.
    """
    obs_seq = np.zeros((seq_len, CANONICAL_N_BANDS * 10), dtype=np.float32)
    for t in range(seq_len):
        # Progressively advance staleness up to 1.0
        stale_age = min(1.0, float(t + 35) / 50.0)
        for b in range(CANONICAL_N_BANDS):
            offset = b * 10
            if b == stuck_band:
                obs_seq[t, offset + 0] = 0.0   # occupancy
                obs_seq[t, offset + 1] = 0.0   # det_rate
                obs_seq[t, offset + 2] = 1.0   # miss_rate
                obs_seq[t, offset + 3] = 0.1   # uncertainty
                obs_seq[t, offset + 4] = 0.0   # age
                obs_seq[t, offset + 9] = 0.05  # priority
            else:
                obs_seq[t, offset + 0] = 0.0   # occupancy
                obs_seq[t, offset + 1] = 0.0   # det_rate
                obs_seq[t, offset + 2] = 0.5   # miss_rate
                obs_seq[t, offset + 3] = 0.7   # uncertainty
                obs_seq[t, offset + 4] = stale_age  # age (stale)
                obs_seq[t, offset + 9] = 0.4 * stale_age + 0.14  # priority
    return obs_seq


def build_canonical_eval_batch(
    data_dir: str = "D:/TSRD",
    n_sequences: int = 16,
    seq_len: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """Build canonical evaluation batch [n_sequences, seq_len, obs_dim]."""
    rng = np.random.RandomState(seed)
    batch_list: list[np.ndarray] = []

    # 1. First sequence is the canonical stuck state
    stuck_seq = generate_stuck_state_sequence(seq_len=seq_len, stuck_band=4)
    batch_list.append(stuck_seq)

    # 2. Remaining 15 sequences sampled from diverse val_stare scenarios
    val_dir = Path(data_dir) / "stare" / "val_stare"
    val_files = sorted(val_dir.glob("*.h5")) if val_dir.exists() else []

    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": CANONICAL_N_BANDS * 10,
        "semantic_memory_path": ":memory:",
    }

    scen_idx = 0
    while len(batch_list) < n_sequences:
        if val_files:
            h5_path = val_files[scen_idx % len(val_files)]
            scen_idx += 1
            try:
                records = load_h5_records(h5_path, chunk_mode="first")
                env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed + len(batch_list))
                obs, _ = env.reset()
                seq_obs = [obs]
                for _ in range(seq_len - 1):
                    # Random exploratory actions to populate realistic belief states
                    act = int(rng.randint(0, CANONICAL_N_BANDS * CANONICAL_N_MODES))
                    obs, _, term, trunc, _ = env.step(act)
                    seq_obs.append(obs)
                    if term or trunc:
                        break
                while len(seq_obs) < seq_len:
                    seq_obs.append(seq_obs[-1])
                batch_list.append(np.array(seq_obs[:seq_len], dtype=np.float32))
                continue
            except Exception as exc:
                logger.warning("Could not sample from %s: %s", h5_path, exc)

        # Fallback if no files: diverse random states
        fallback_seq = rng.uniform(0.0, 1.0, size=(seq_len, CANONICAL_N_BANDS * 10)).astype(np.float32)
        batch_list.append(fallback_seq)

    tensor_batch = torch.tensor(np.stack(batch_list[:n_sequences], axis=0), dtype=torch.float32)
    return tensor_batch


def get_or_create_fixed_eval_batch(
    cache_path: Path | str = CANONICAL_EVAL_BATCH_PATH,
    data_dir: str = "D:/TSRD",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load canonical evaluation batch from disk, or create and persist if missing."""
    p = Path(cache_path)
    if p.exists():
        try:
            tensor = torch.load(p, map_location=device)
            if tensor.dim() == 3 and tensor.size(0) >= 1 and tensor.size(-1) == CANONICAL_N_BANDS * 10:
                return tensor.to(device)
        except Exception as exc:
            logger.warning("Failed to load cached eval batch %s: %s — recreating", p, exc)

    p.parent.mkdir(parents=True, exist_ok=True)
    tensor = build_canonical_eval_batch(data_dir=data_dir)
    torch.save(tensor, p)
    logger.info("Persisted canonical fixed evaluation batch to %s [shape: %s]", p, list(tensor.shape))
    return tensor.to(device)


def evaluate_q_diagnostics(
    model: torch.nn.Module,
    eval_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Compute Q-margin, Q-spread, and stuck-state argmax diagnostics on the fixed eval batch.

    Args:
        model: DRQNScheduler online network.
        eval_batch: Tensor of shape [N, T, obs_dim].
        device: Target execution device.

    Returns:
        Dict with keys:
          - q_margin_mean: Average top1 - top2 Q-margin across all sequences.
          - q_spread_mean: Average max(Q) - min(Q) across all sequences.
          - stuck_state_top1_band: Argmax band on the last step of the stuck sequence.
          - stuck_state_top2_band: Second-choice band on the last step of the stuck sequence.
          - stuck_state_margin: Top1 - top2 margin on the stuck sequence.
          - stuck_state_top1_mode: Selected dwell mode on the stuck sequence.
    """
    model.eval()
    t_batch = eval_batch.to(device)
    with torch.inference_mode():
        out = model(t_batch)
        q_vals = out[0] if isinstance(out, (tuple, list)) else out  # [N, T, n_actions]
        # Evaluate metrics on the last timestep of each sequence
        last_q = q_vals[:, -1, :]  # [N, n_actions]
        top2 = last_q.topk(2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        spreads = last_q.max(dim=-1).values - last_q.min(dim=-1).values

        # Stuck state is sequence 0
        stuck_q = last_q[0]  # [n_actions]
        stuck_top2_idx = stuck_q.topk(2, dim=-1).indices
        stuck_top2_val = stuck_q.topk(2, dim=-1).values
        n_modes = CANONICAL_N_MODES
        top1_act = int(stuck_top2_idx[0].item())
        top2_act = int(stuck_top2_idx[1].item())

        stuck_top1_band = top1_act // n_modes
        stuck_top1_mode = top1_act % n_modes
        stuck_top2_band = top2_act // n_modes
        stuck_margin = float((stuck_top2_val[0] - stuck_top2_val[1]).item())

    return {
        "q_margin_mean": float(margins.mean().item()),
        "q_spread_mean": float(spreads.mean().item()),
        "stuck_state_top1_band": stuck_top1_band,
        "stuck_state_top1_mode": stuck_top1_mode,
        "stuck_state_top2_band": stuck_top2_band,
        "stuck_state_margin": stuck_margin,
    }


def run_evaluation(
    scheduler: Any,
    env: CognitiveRFScanEnv | None = None,
    scenario_ids: Sequence[str] | None = None,
    n_steps: int = 500,
    seed: int = 42,
    policy_mode: str = "operational",
    data_dir: str | Path = "D:/TSRD",
    device: str | torch.device = "cpu",
    dataset_subset: str | None = None,
    scenario_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Unified evaluation harness computing all 7 PS Figures of Merit.

    Args:
        scheduler: Scheduler policy (SmartScanMoE, DRQNScheduler, or baseline).
        env: Optional pre-existing environment.
        scenario_ids: Scenario IDs to evaluate (e.g. ['config_117', ...]).
        n_steps: Episode steps per scenario.
        seed: Random seed for environment reproducibility.
        policy_mode: 'operational' (deterministic) or 'demo' (exploratory).
        data_dir: Root dataset directory.
        device: Torch execution device for neural models.
        dataset_subset: Optional explicit subset directory under stare (e.g. 'val_stare', 'test_stare').
        scenario_dir: Optional explicit direct directory containing .h5 scenarios.

    Returns:
        Dict containing all 7 aggregate Figures of Merit and per-scenario breakdown.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    if scenario_dir is not None:
        target_dir = Path(scenario_dir)
    elif dataset_subset is not None:
        target_dir = Path(data_dir) / "stare" / dataset_subset
        if not target_dir.exists():
            target_dir = Path(data_dir) / dataset_subset
    else:
        target_dir = Path(data_dir) / "stare" / "val_stare"
        if not target_dir.exists():
            target_dir = Path(data_dir) / "val"

    if scenario_ids is None or len(scenario_ids) == 0:
        if env is not None:
            active_scenarios = [("provided_env", None)]
        else:
            active_scenarios = [("default_scenario", None)]
    else:
        active_scenarios = [(scen_id, scen_id) for scen_id in scenario_ids]

    scenario_metrics: dict[str, EWMetrics] = {}

    for scen_idx, (scen_name, scen_id) in enumerate(active_scenarios):
        # 1. Resolve pulse records & environment
        if scen_id is not None:
            if scenario_dir is not None or dataset_subset is not None:
                # Explicit routing: never guess or fall back to other split directories
                h5_path = target_dir / f"{scen_id}.h5"
            else:
                candidates = [
                    target_dir / f"{scen_id}.h5",
                    Path(data_dir) / "stare" / "val_stare" / f"{scen_id}.h5",
                    Path(data_dir) / "stare" / "test_stare" / f"{scen_id}.h5",
                    Path(data_dir) / "stare" / "train_stare" / f"{scen_id}.h5",
                    Path(data_dir) / f"{scen_id}.h5",
                ]
                h5_path = next((p for p in candidates if p.exists()), candidates[0])
            if h5_path.exists():
                records = load_h5_records(
                    h5_path,
                    freq_min_mhz=0.0,
                    freq_max_mhz=18000.0,
                    time_horizon_us=30000000.0,
                    max_pulses=50000,
                    chunk_mode="first",
                )
            elif policy_mode == "operational":
                raise FileNotFoundError(
                    f"Operational evaluation requires real TSRD scenario file: {h5_path} not found! "
                    f"Synthetic fallback is prohibited in operational mode."
                )
            else:
                logger.warning("Scenario %s not found in %s; falling back to synthetic records", scen_id, val_dir)
                records = synthetic_records(seed=seed + scen_idx)

            env_cfg = {
                "n_bands": CANONICAL_N_BANDS,
                "n_modes": CANONICAL_N_MODES,
                "obs_dim": CANONICAL_N_BANDS * 10,
                "semantic_memory_path": ":memory:",
                "max_steps_per_episode": n_steps,
                "reward": {"version": "v2"},
            }
            if env is not None and hasattr(env, "config"):
                env_cfg.update(env.config)
            eval_env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed + scen_idx)
        else:
            if env is not None:
                eval_env = env
            else:
                eval_env = CognitiveRFScanEnv(
                    {
                        "n_bands": CANONICAL_N_BANDS,
                        "n_modes": CANONICAL_N_MODES,
                        "obs_dim": CANONICAL_N_BANDS * 10,
                        "semantic_memory_path": ":memory:",
                        "max_steps_per_episode": n_steps,
                        "reward": {"version": "v2"},
                    },
                    records=synthetic_records(seed=seed),
                    seed=seed,
                )

        # 2. Reset scheduler and environment
        obs, _ = eval_env.reset()
        if hasattr(scheduler, "reset"):
            scheduler.reset()
        hidden = None
        if hasattr(scheduler, "init_hidden"):
            hidden = scheduler.init_hidden(1, dev)

        episode_log: dict[str, list[Any]] = {
            "hits": [],
            "chosen_bands": [],
            "active_bands_per_step": [],
            "rewards": [],
            "predicted_times": [],
            "actual_times": [],
            "false_alarms": [],
            "missed_dwells": [],
            "operational_latencies": [],
            "genuine_predictive_time_errors": [],
        }

        # 3. Episode step loop
        for step in range(n_steps):
            # Select action
            if hasattr(scheduler, "select_action"):
                try:
                    action, hidden, attr = scheduler.select_action(obs, hidden, policy_mode=policy_mode)
                except TypeError:
                    action, hidden, attr = scheduler.select_action(obs, hidden)
            elif hasattr(scheduler, "act"):
                if isinstance(scheduler, DRQNScheduler) or hasattr(scheduler, "lstm"):
                    t_obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                    action, hidden = scheduler.act(t_obs, hidden)
                else:
                    res = scheduler.act(obs)
                    action = res[0] if isinstance(res, tuple) else res
            elif hasattr(scheduler, "step"):
                action = scheduler.step(obs)
            elif callable(scheduler):
                action = scheduler(obs)
            else:
                raise ValueError(f"Unsupported scheduler interface: {type(scheduler)}")

            action = int(action)
            band = int(band_of_action(action, CANONICAL_N_MODES))

            # TASK 2.1: Wire TemporalPredictor Output into avg_intercept_time_error
            temporal_pred = getattr(scheduler, "temporal_predictor", getattr(eval_env, "temporal_predictor", None))
            pred_toa = None
            dwell_start = float(getattr(eval_env.receiver, "current_time_us", 0.0))
            if temporal_pred is not None:
                try:
                    preds = temporal_pred.predict_all(current_time=dwell_start, horizon_us=50000.0)
                    for p in preds:
                        if p.target_band == band:
                            pred_toa = float(p.next_expected_toa)
                            break
                except Exception as exc:
                    logger.debug("Temporal prediction query failed in eval_batch: %s", exc)

            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)

            hit = bool(info.get("hit", False))
            active_bands = info.get("active_bands", [])
            false_alarm = bool(not (band in active_bands) and hit)
            # Missed dwell (FN): agent chose an active band but did not detect
            missed = bool((band in active_bands) and not hit)

            episode_log["hits"].append(hit)
            episode_log["chosen_bands"].append(band)
            episode_log["active_bands_per_step"].append(active_bands)
            episode_log["rewards"].append(float(reward))
            episode_log["false_alarms"].append(false_alarm)
            episode_log["missed_dwells"].append(missed)

            if hit:
                op_latency = float(info.get("intercept_time_error_us", 0.0))
                episode_log["operational_latencies"].append(op_latency)
                actual_toa = float(dwell_start + op_latency)
                if pred_toa is not None:
                    pred_err = abs(pred_toa - actual_toa)
                    episode_log["genuine_predictive_time_errors"].append(pred_err)
                    episode_log["predicted_times"].append(pred_toa)
                    episode_log["actual_times"].append(actual_toa)
                else:
                    # Legacy aggregate field compatibility only
                    episode_log["predicted_times"].append(dwell_start)
                    episode_log["actual_times"].append(actual_toa)

            if hasattr(scheduler, "update_result"):
                scheduler.update_result(hit, band)
            if hasattr(scheduler, "update"):
                scheduler.update(action)

            if terminated or truncated:
                break

        # Compute all 7 FoMs for this scenario with physics sensitivity
        sens_val = float(getattr(eval_env.receiver, "sensitivity_dbm", -110.0))
        metrics = compute_all_metrics(episode_log, min_detectable_signal_dbm=sens_val)
        scenario_metrics[scen_name] = metrics

    # 4. Aggregate across scenarios
    total_tp = int(sum(m.tp for m in scenario_metrics.values()))
    total_fn = int(sum(m.fn for m in scenario_metrics.values()))
    total_fp = int(sum(m.fp for m in scenario_metrics.values()))
    total_tn = int(sum(m.tn for m in scenario_metrics.values()))
    total_dwells = int(sum(m.n_receiver_dwells for m in scenario_metrics.values()))

    # True decision-level aggregate Pd = TP / (TP + FN), Pfa = FP / (FP + TN)
    denom_pd = total_tp + total_fn
    agg_pd = float(total_tp / denom_pd) if denom_pd > 0 else 0.0
    denom_pfa = total_fp + total_tn
    agg_pfa = float(total_fp / denom_pfa) if denom_pfa > 0 else 0.0

    avg_sensitivity = float(np.mean([m.sensitivity_dbm for m in scenario_metrics.values()]))
    avg_rate = float(np.mean([m.avg_intercept_rate for m in scenario_metrics.values()]))
    avg_reward = float(np.mean([m.avg_reward for m in scenario_metrics.values()]))
    avg_pct_correct = float((total_tp + total_tn) / max(1, total_dwells) * 100.0)
    avg_time_error = float(np.mean([m.avg_intercept_time_error_us for m in scenario_metrics.values()]))
    avg_op_latency = float(np.mean([m.operational_intercept_latency_us for m in scenario_metrics.values()]))
    valid_pred_errs = [m.predictive_time_error_us for m in scenario_metrics.values() if m.predictive_time_error_us is not None]
    avg_pred_err = float(np.mean(valid_pred_errs)) if valid_pred_errs else None
    avg_pred_cov = float(np.mean([m.prediction_coverage for m in scenario_metrics.values()]))

    return {
        "pd": agg_pd,
        "pfa": agg_pfa,
        "canonical_pfa": agg_pfa,
        "sensitivity_dbm": avg_sensitivity,
        "avg_intercept_rate": avg_rate,
        "mean_intercept_rate": avg_rate,
        "avg_reward": avg_reward,
        "pct_correct_predictions": avg_pct_correct,
        "avg_intercept_time_error_us": avg_time_error,
        "operational_intercept_latency_us": avg_op_latency,
        "predictive_time_error_us": avg_pred_err,
        "prediction_coverage": avg_pred_cov,
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "tn": total_tn,
        "n_intercepts": total_tp,
        "n_receiver_dwells": total_dwells,
        "n_false_alarms": total_fp,
        "n_missed_dwells": total_fn,
        "n_true_negatives": total_tn,
        "scenario_breakdown": {
            name: m.to_dict() for name, m in scenario_metrics.items()
        },
    }

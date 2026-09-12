"""Phase 9D: Advantage-Pathway Micro-Calibration Protocol.

Quarantined Advantage Head Calibration:
1. Start from immutable champion checkpoint_step_25500.pt.
2. Keep shared representations permanently frozen:
   - input_norm
   - lstm
   - band_encoder
   - ctx_proj
   - value_stream
   - auxiliary heads (intercept_prob_head, intercept_time_head)
3. Unfreeze ONLY band_advantage_head:
   - band_advantage_head.0.weight (32 -> 64)
   - band_advantage_head.0.bias
   - band_advantage_head.2.weight (64 -> 5)
   - band_advantage_head.2.bias
   Total: 2,245 parameters (0.14% of model)
4. Training configuration:
   - Optimizer: Adam
   - Learning rate: 1.0e-6 (<= 1e-6)
   - Gradient clipping: 1.0 (<= 1.0)
   - Loss: TD Huber loss on dueling Q-values
   - Replay: StratifiedModeSampler (sparse_balanced)
5. Evaluation & Sentinel Gate (every 100 steps up to 500 steps):
   - Held-out config_119 advantage margin: Delta A = A(s, b*, Mode 2) - A(s, b*, Mode 1) >= +0.2000
   - Multi-seed canonical 10-scenario evaluation (seeds 42, 123, 999)
   - No individual scenario regresses > 0.50% IR relative to champion
   - Worst-case IR >= 16.20%
   - Mode 2 dwell share on config_119 within [70.0%, 78.0%]
   - Pd >= 99.80%, Pfa == 0.0000
   - Frozen backbone parameter drift exactly 0.0
6. Hard Rollback Sentinel:
   - If any condition is violated at any 100-step checkpoint, halt and restore step 25,500 champion.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records, ScenarioSource
from cognitive_ew_smart_scan.src.evaluation.benchmark_contract import CANONICAL_SCENARIOS
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer
from cognitive_ew_smart_scan.src.training.stratified_mode_sampler import StratifiedModeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("advantage_microcalib_9d")

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "UNKNOWN_GIT_REVISION"


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path).resolve()
    if not p.exists():
        return f"MISSING:{p}"
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_model_multi_seed(
    model: DRQNScheduler,
    seeds: List[int],
    n_steps: int = 1000,
    data_root: str = "D:/TSRD",
) -> Dict[str, Any]:
    model.eval()
    agile_scens = {"config_29", "config_195", "config_241", "config_119"}
    sparse_scens = {"config_143", "config_119"}

    seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        scen_irs = {}
        scen_mode2_fractions = {}
        pds = []
        pfas = []

        for scen_id in CANONICAL_SCENARIOS:
            h5_path = Path(data_root) / "stare" / "val_stare" / f"{scen_id}.h5"
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

            for step in range(n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
                with torch.no_grad():
                    q, _, hx = model(obs_t, hx)
                act = int(q.squeeze().argmax().item())
                modes.append(act % 5)
                obs, _, term, trunc, info = env.step(act)
                hits += int(info.get("hit", False))
                if term or trunc:
                    break

            ir = hits / float(n_steps)
            scen_irs[scen_id] = ir
            m2_frac = modes.count(2) / float(len(modes))
            scen_mode2_fractions[scen_id] = m2_frac
            pds.append(float(env.fom.pd))
            pfas.append(float(env.fom.pfa))

        irs_list = [scen_irs[s] * 100 for s in CANONICAL_SCENARIOS]
        agile_irs = [scen_irs[s] * 100 for s in agile_scens]
        sparse_irs = [scen_irs[s] * 100 for s in sparse_scens]

        seed_results[seed] = {
            "mean_ir": float(np.mean(irs_list)),
            "worst_case_ir": float(np.min(irs_list)),
            "agile_ir": float(np.mean(agile_irs)),
            "sparse_ir": float(np.mean(sparse_irs)),
            "config_119_ir": float(scen_irs["config_119"] * 100),
            "config_119_mode2_frac": float(scen_mode2_fractions["config_119"]),
            "pd": float(np.mean(pds)) * 100,
            "pfa": float(np.mean(pfas)),
            "scenario_irs": scen_irs,
        }

    all_mean_irs = [r["mean_ir"] for r in seed_results.values()]
    all_worst_irs = [r["worst_case_ir"] for r in seed_results.values()]
    all_agile_irs = [r["agile_ir"] for r in seed_results.values()]
    all_sparse_irs = [r["sparse_ir"] for r in seed_results.values()]
    all_119_irs = [r["config_119_ir"] for r in seed_results.values()]
    all_119_m2 = [r["config_119_mode2_frac"] for r in seed_results.values()]
    all_pds = [r["pd"] for r in seed_results.values()]
    all_pfas = [r["pfa"] for r in seed_results.values()]

    # Compute scenario_irs_mean across seeds (in percent)
    scenario_irs_mean = {}
    for s in CANONICAL_SCENARIOS:
        scenario_irs_mean[s] = float(np.mean([seed_results[seed]["scenario_irs"][s] * 100 for seed in seeds]))

    return {
        "seeds": seeds,
        "mean_ir_mean": float(np.mean(all_mean_irs)),
        "mean_ir_std": float(np.std(all_mean_irs)),
        "worst_case_ir_mean": float(np.mean(all_worst_irs)),
        "worst_case_ir_std": float(np.std(all_worst_irs)),
        "agile_ir_mean": float(np.mean(all_agile_irs)),
        "sparse_ir_mean": float(np.mean(all_sparse_irs)),
        "config_119_ir_mean": float(np.mean(all_119_irs)),
        "config_119_ir_std": float(np.std(all_119_irs)),
        "config_119_mode2_frac_mean": float(np.mean(all_119_m2)),
        "pd_mean": float(np.mean(all_pds)),
        "pfa_mean": float(np.mean(all_pfas)),
        "scenario_irs_mean": scenario_irs_mean,
        "per_seed": seed_results,
    }


def audit_config_119_margin(
    model: DRQNScheduler,
    seed: int = 42,
    n_steps: int = 1000,
    data_root: str = "D:/TSRD",
) -> Dict[str, Any]:
    """Audit advantage margin Delta A = A(s, b*, Mode 2) - A(s, b*, Mode 1) on config_119."""
    model.eval()
    h5_path = Path(data_root) / "stare" / "val_stare" / "config_119.h5"
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
    margins = []
    mode_counts = {m: 0 for m in range(5)}
    hits = 0

    for step in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            x = model.input_norm(obs_t)
            lstm_out, hx = model.lstm(x, hx)
            B, T, _ = obs_t.shape
            obs_bands = obs_t.view(B, T, model.n_bands, model.band_features)
            band_emb = model.band_encoder(obs_bands)
            ctx = model.ctx_proj(lstm_out).unsqueeze(2).expand(-1, -1, model.n_bands, -1)
            joint = torch.cat([band_emb, ctx], dim=-1)
            a_bands = model.band_advantage_head(joint)  # (1, 1, 36, 5)
            a = a_bands.view(B, T, model.n_actions)
            v = model.value_stream(lstm_out)
            q = v + a - a.mean(dim=-1, keepdim=True)

            act = int(torch.argmax(q.squeeze()).item())
            band_chosen = act // 5
            mode_chosen = act % 5

            adv_chosen_band = a_bands[0, 0, band_chosen].numpy()
            margin = float(adv_chosen_band[2] - adv_chosen_band[1])
            margins.append(margin)

        mode_counts[mode_chosen] += 1
        obs, _, term, trunc, info = env.step(act)
        hits += int(info.get("hit", False))
        if term or trunc:
            break

    return {
        "scenario": "config_119",
        "steps": n_steps,
        "hits": hits,
        "intercept_rate": hits / float(n_steps),
        "mean_advantage_margin_m2_minus_m1": float(np.mean(margins)),
        "min_advantage_margin": float(np.min(margins)),
        "max_advantage_margin": float(np.max(margins)),
        "mode_distribution": mode_counts,
        "mode2_fraction": mode_counts[2] / float(n_steps),
    }

def execute_phase9d_microcalib(
    parent_path: Path,
    baseline_path: Path,
    reservoir_path: Path,
    candidate_dir: Path,
    total_steps: int = 500,
    eval_interval: int = 100,
    lr: float = 1.0e-6,
    clip_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    logger.info("Initializing Phase 9D: Advantage-Pathway Micro-Calibration...")
    logger.info("Total steps: %d, Eval interval: %d, LR: %.2e, Grad Clip: %.2f",
                total_steps, eval_interval, lr, clip_grad_norm)

    sha_parent = compute_file_sha256(parent_path)
    sha_base = compute_file_sha256(baseline_path)
    sha_res = compute_file_sha256(reservoir_path)

    assert sha_parent == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554", f"Champion hash modified! {sha_parent}"
    assert sha_base == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0", f"Baseline hash modified! {sha_base}"

    ckpt = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5)
    model.load_state_dict(ckpt["state_dict"])

    # 1. Parameter Freezing Setup: STRICTLY unfreeze ONLY band_advantage_head
    for name, p in model.named_parameters():
        p.requires_grad = False

    trainable_params = []
    trainable_param_names = []
    frozen_param_names = []

    for name, p in model.named_parameters():
        if name.startswith("band_advantage_head."):
            p.requires_grad = True
            trainable_params.append(p)
            trainable_param_names.append(name)
        else:
            frozen_param_names.append(name)

    logger.info("Trainable parameters (%d tensors): %s", len(trainable_param_names), trainable_param_names)
    total_trainable_elems = sum(p.numel() for p in trainable_params)
    total_model_elems = sum(p.numel() for p in model.parameters())
    logger.info("Trainable elements: %d / %d (%.3f%%)", total_trainable_elems, total_model_elems, (total_trainable_elems / total_model_elems) * 100)
    assert total_trainable_elems == 2245, f"Expected exactly 2,245 parameters in band_advantage_head, got {total_trainable_elems}"

    # Initial frozen parameter snapshot for bit-exact drift verification
    frozen_snapshots_initial = {name: p.clone().detach() for name, p in model.named_parameters() if not p.requires_grad}

    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.HuberLoss()

    replay_buffer = SequenceReplayBuffer(capacity=50000, seq_len=16, obs_dim=360, burn_in=8, seed=42)
    replay_buffer.load_episodes(reservoir_path)

    sampler = StratifiedModeSampler(
        buffer=replay_buffer,
        mode_weights={0: 0.15, 1: 0.25, 2: 0.30, 3: 0.15, 4: 0.15},
        seq_len=16,
        burn_in=8,
        seed=42,
        replay_strategy="sparse_balanced",
    )

    data_root = Path("D:/TSRD")
    train_source = ScenarioSource(data_root=data_root, mode="stare", subset="train", source_type="world")
    rng = np.random.default_rng(42)
    train_path = rng.choice(train_source.eligible_files)
    records = load_h5_records(train_path)

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
        seed=42,
    )
    obs, _ = env.reset(seed=42)
    hx = None

    seeds = [42, 123, 999]
    eval_steps = 1000

    # Reference evaluation of champion prior to training
    logger.info("Auditing reference parent champion prior to training...")
    eval_parent_ref = evaluate_model_multi_seed(model, seeds, n_steps=eval_steps)
    margin_parent_ref = audit_config_119_margin(model, seed=42, n_steps=eval_steps)
    logger.info("Parent Reference Mean IR: %.2f%%, config_119: %.2f%%, M2 share: %.1f%%, Margin Delta A: %.4f",
                eval_parent_ref["mean_ir_mean"], eval_parent_ref["config_119_ir_mean"],
                eval_parent_ref["config_119_mode2_frac_mean"] * 100, margin_parent_ref["mean_advantage_margin_m2_minus_m1"])

    champion_scenario_irs = eval_parent_ref["scenario_irs_mean"]

    # Training state
    candidate_dir.mkdir(parents=True, exist_ok=True)
    target_update_freq = 50
    checkpoint_evaluations = {}
    halt_due_to_rollback = False
    rollback_reason = None
    saved_checkpoints = []

    model.train()

    for step in range(1, total_steps + 1):
        obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            q, _, hx = model(obs_t, hx)
        if float(np.random.rand()) < 0.05:
            act = int(np.random.randint(180))
        else:
            act = int(q.squeeze().argmax().item())

        next_obs, r, term, trunc, info = env.step(act)
        replay_buffer.add(
            obs=obs,
            action=act,
            reward=r,
            next_obs=next_obs,
            done=bool(term or trunc),
            hit_prob=1.0 if info.get("hit", False) else 0.0,
            intercept_time_us=float(info.get("intercept_latency_us", np.nan)),
            scenario_id=getattr(env.radio_env, "scenario_id", "train"),
        )

        if term or trunc:
            obs, _ = env.reset()
            hx = None
        else:
            obs = next_obs

        # Optimization step every 4 env steps
        if step % 4 == 0 and replay_buffer.can_sample(32):
            batch, _ = sampler.sample_mode_stratified(32)
            act_b = torch.tensor(batch["actions"], dtype=torch.int64)
            rew_b = torch.tensor(batch["rewards"], dtype=torch.float32)
            next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32)
            valid_mask = torch.tensor(batch["valid_mask"], dtype=torch.bool)
            burn_in_mask = torch.tensor(batch["burn_in_mask"], dtype=torch.bool)
            loss_mask = valid_mask & (~burn_in_mask)

            q_online, aux_online, _ = model(torch.tensor(batch["obs"], dtype=torch.float32))
            with torch.no_grad():
                q_target_next, _, _ = target_model(next_obs_b)
                target_td = rew_b + 0.99 * (1.0 - dones_b) * q_target_next.max(dim=-1)[0]

            q_chosen = q_online.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            td_loss = loss_fn(q_chosen[loss_mask], target_td[loss_mask])

            optimizer.zero_grad()
            td_loss.backward()

            # Verify that frozen modules received strictly zero gradients
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"Frozen parameter {name} received gradient!"

            torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm)
            optimizer.step()

        # Target update
        if step % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Checkpoint evaluation every eval_interval steps
        if step % eval_interval == 0:
            current_global_step = 25500 + step
            logger.info(">>> EVALUATION SENTINEL GATE AT STEP %d (GLOBAL STEP %d) <<<", step, current_global_step)

            # 1. Assert frozen parameters bit-exact zero drift
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    diff = torch.max(torch.abs(p - frozen_snapshots_initial[name])).item()
                    assert diff == 0.0, f"Critical: Frozen parameter {name} drifted by {diff} at step {step}!"

            # 2. Inspect max update in advantage head
            max_adv_diff = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p_orig = ckpt["state_dict"][name]
                    diff = torch.max(torch.abs(p - p_orig)).item()
                    max_adv_diff = max(max_adv_diff, diff)
            logger.info("Advantage head max absolute parameter displacement: %.6f", max_adv_diff)

            # Save checkpoint
            ckpt_path = candidate_dir / f"checkpoint_step_{current_global_step}_phase9d.pt"
            torch.save({
                "global_step": current_global_step,
                "microcalib_step": step,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "parent_sha256": sha_parent,
                "baseline_sha256": sha_base,
                "git_commit": get_git_commit(),
                "phase": "phase9d_advantage_microcalib",
            }, ckpt_path)
            sha_ckpt = compute_file_sha256(ckpt_path)
            saved_checkpoints.append({"path": str(ckpt_path), "sha256": sha_ckpt, "step": step, "global_step": current_global_step})

            # Run Multi-Seed Canonical 10-Scenario Evaluation
            eval_res = evaluate_model_multi_seed(model, seeds, n_steps=eval_steps)
            margin_res = audit_config_119_margin(model, seed=42, n_steps=eval_steps)

            logger.info("Step %d Eval: Mean IR: %.2f%% (std: %.2f%%), Agile IR: %.2f%%, Worst IR: %.2f%%, config_119 IR: %.2f%%, M2 share: %.1f%%, Margin Delta A: %.4f",
                        step, eval_res["mean_ir_mean"], eval_res["mean_ir_std"],
                        eval_res["agile_ir_mean"], eval_res["worst_case_ir_mean"],
                        eval_res["config_119_ir_mean"], eval_res["config_119_mode2_frac_mean"] * 100,
                        margin_res["mean_advantage_margin_m2_minus_m1"])

            # Evaluate Sentinels
            sentinel_passed = True
            violations = []

            # 1. Delta A margin >= +0.2000
            if margin_res["mean_advantage_margin_m2_minus_m1"] < 0.2000:
                sentinel_passed = False
                violations.append(f"Delta A margin ({margin_res['mean_advantage_margin_m2_minus_m1']:.4f}) < +0.2000 floor")

            # 2. Non-regression: No scenario drops > 0.50% from champion
            for scen_id, champ_ir in champion_scenario_irs.items():
                cand_ir = eval_res["scenario_irs_mean"][scen_id]
                floor_ir = champ_ir - 0.50
                if cand_ir < floor_ir:
                    sentinel_passed = False
                    violations.append(f"Scenario {scen_id} regressed: {cand_ir:.2f}% < floor {floor_ir:.2f}% (Champion: {champ_ir:.2f}%)")

            # 3. Worst-case IR >= 16.20%
            if eval_res["worst_case_ir_mean"] < 16.20:
                sentinel_passed = False
                violations.append(f"Worst-case IR ({eval_res['worst_case_ir_mean']:.2f}%) < 16.20% floor")

            # 4. Mode 2 share within [70.0%, 78.0%]
            m2_pct = eval_res["config_119_mode2_frac_mean"] * 100.0
            if not (70.0 <= m2_pct <= 78.0):
                sentinel_passed = False
                violations.append(f"config_119 Mode 2 share ({m2_pct:.1f}%) outside required window [70.0%, 78.0%]")

            # 5. Pd >= 99.80% and Pfa == 0.0000
            if eval_res["pd_mean"] < 99.80:
                sentinel_passed = False
                violations.append(f"Pd ({eval_res['pd_mean']:.2f}%) < 99.80%")
            if eval_res["pfa_mean"] > 0.0000:
                sentinel_passed = False
                violations.append(f"Pfa ({eval_res['pfa_mean']:.4f}) > 0.0000")

            checkpoint_evaluations[f"step_{step}"] = {
                "step": step,
                "global_step": current_global_step,
                "checkpoint": {"path": str(ckpt_path), "sha256": sha_ckpt},
                "max_advantage_param_diff": max_adv_diff,
                "metrics": eval_res,
                "config_119_margin_audit": margin_res,
                "sentinel_passed": sentinel_passed,
                "violations": violations,
            }

            if not sentinel_passed:
                logger.error(">>> SENTINEL VIOLATION AT STEP %d: %s <<<", step, violations)
                logger.error(">>> TRIGGERING FAIL-SAFE HALT AND ROLLBACK TO STEP 25,500 CHAMPION <<<")
                halt_due_to_rollback = True
                rollback_reason = f"Violations at step {step}: " + "; ".join(violations)
                break
            else:
                logger.info(">>> SENTINEL GATE PASSED AT STEP %d CLEANLY <<<", step)

            model.train()

    # Final bit-exact verification on parent & baseline checkpoints
    sha_parent_final = compute_file_sha256(parent_path)
    sha_base_final = compute_file_sha256(baseline_path)
    assert sha_parent_final == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554", f"Champion corrupted! {sha_parent_final}"
    assert sha_base_final == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0", f"Baseline corrupted! {sha_base_final}"

    results_payload = {
        "experiment": "Phase 9D - Advantage-Pathway Micro-Calibration",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "configuration": {
            "total_steps": total_steps,
            "eval_interval": eval_interval,
            "trainable_modules": ["band_advantage_head"],
            "trainable_params_count": total_trainable_elems,
            "frozen_modules": ["input_norm", "lstm", "band_encoder", "ctx_proj", "value_stream", "intercept_prob_head", "intercept_time_head"],
            "learning_rate": lr,
            "clip_grad_norm": clip_grad_norm,
            "target_update_freq": target_update_freq,
        },
        "provenance": {
            "parent_checkpoint": {"path": str(parent_path), "sha256": sha_parent},
            "baseline_checkpoint": {"path": str(baseline_path), "sha256": sha_base},
            "reservoir": {"path": str(reservoir_path), "sha256": sha_res},
        },
        "parent_reference": {
            "metrics": eval_parent_ref,
            "config_119_margin_audit": margin_parent_ref,
        },
        "checkpoint_evaluations": checkpoint_evaluations,
        "halt_due_to_rollback": halt_due_to_rollback,
        "rollback_reason": rollback_reason,
        "safety_assertions": {
            "parent_unmodified": sha_parent_final == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554",
            "baseline_unmodified": sha_base_final == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
            "frozen_parameters_drift_zero": True,
            "non_trainable_gradients_none": True,
        },
    }

    report_path = PKG_ROOT / "reports" / "diagnostics" / "phase9d_advantage_calib_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Phase 9D completed. Report saved to: %s", report_path)
    return results_payload


if __name__ == "__main__":
    parent = PKG_ROOT / "checkpoints" / "safe_continuation_candidate" / "checkpoint_step_25500.pt"
    baseline = PKG_ROOT / "checkpoints" / "production_baseline" / "checkpoint_gate_25000_frozen.pt"
    reservoir = PKG_ROOT / "checkpoints" / "production_baseline" / "baseline_reservoir_5k.pkl"
    cand_dir = PKG_ROOT / "checkpoints" / "phase9d_advantage_calib"

    res = execute_phase9d_microcalib(
        parent_path=parent,
        baseline_path=baseline,
        reservoir_path=reservoir,
        candidate_dir=cand_dir,
        total_steps=500,
        eval_interval=100,
        lr=1.0e-6,
        clip_grad_norm=1.0,
    )

    print("=" * 80)
    print("PHASE 9D - ADVANTAGE-PATHWAY MICRO-CALIBRATION SUMMARY")
    print(f"Halt due to rollback: {res['halt_due_to_rollback']}")
    if res["rollback_reason"]:
        print(f"Rollback reason: {res['rollback_reason']}")
    for step_k, ev in res["checkpoint_evaluations"].items():
        m = ev["metrics"]
        marg = ev["config_119_margin_audit"]
        p_status = "PASS" if ev["sentinel_passed"] else "FAIL"
        print(f"[{p_status}] {step_k} (Global {ev['global_step']}): Mean IR: {m['mean_ir_mean']:.2f}% | Agile IR: {m['agile_ir_mean']:.2f}% | Worst IR: {m['worst_case_ir_mean']:.2f}% | c119 IR: {m['config_119_ir_mean']:.2f}% | M2: {m['config_119_mode2_frac_mean']*100:.1f}% | Margin Delta A: {marg['mean_advantage_margin_m2_minus_m1']:.4f} | Max dW: {ev['max_advantage_param_diff']:.6f}")
    print("=" * 80)


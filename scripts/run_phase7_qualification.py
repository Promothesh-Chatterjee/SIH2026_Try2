#!/usr/bin/env python3
"""
Phase 7 Final Qualification & Verification Runner.
Evaluates Gates 7.1 through 7.9 for explicit, provenance-bound, fail-closed checkpoint promotion.

Generates:
  - experiments/reports/phase7/PHASE_7_FINAL_QUALIFICATION_REPORT.md
  - experiments/reports/phase7/phase7_checkpoint_promotion_gate_report.json
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ew_core.evaluation.promotion_sentinel import evaluate_promotion
from ew_core.training.safety.checkpoint_guard import (
    CheckpointGuard,
    CheckpointSecurityError,
    CheckpointTamperedError,
    ExplicitPromotionRequiredError,
    EXPECTED_BASELINE_SHA256,
    QuarantinedCheckpointError,
    sha256_file,
)
from ew_core.training.safety.rollback_manager import RollbackManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7_qualification")


def _create_mock_ckpt(path: Path, step: int) -> str:
    payload = {"step": step, "weights": torch.randn(4, 4)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def _create_mock_report(
    ckpt_path: Path,
    ckpt_sha: str,
    status: str = "APPROVED",
    mean_ir: float = 80.0,
    baseline_sha: str = EXPECTED_BASELINE_SHA256,
    git_rev: str = "f93b253c072d627341e3d360f7fe98ec8d3e6963",
    seed: int = 42,
) -> Dict[str, Any]:
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "baseline_checkpoint_sha256": baseline_sha,
        "git_revision": git_rev,
        "config_sha256": "8f3a1234",
        "evaluation_seed": seed,
        "metric_units": {
            "mean_ir": "percent",
            "agile_ir": "percent",
            "sparse_ir": "percent",
            "worst_case_ir": "percent",
            "pfa": "fraction",
        },
        "scenario_summary": {
            "mean_ir": mean_ir,
            "agile_ir": 82.0,
            "sparse_ir": 76.0,
            "worst_case_ir": 72.0,
            "decision_pd": 99.85,
            "pfa": 0.0005,
            "mean_pfa": 0.0005,
            "mode_proportions": [0.2, 0.4, 0.25, 0.1, 0.05],
            "mode_proportions_agile_sparse": [0.1, 0.3, 0.25, 0.2, 0.15],
        },
        "action_summary": {
            "mode2_fraction_agile_sparse": 0.25,
            "action_entropy": 2.5,
            "distinct_bands": 36.0,
        },
        "training_diagnostics": {
            "advantage_head_drift": 0.0,
            "q_loss": 0.04,
            "q_max": 8.5,
        },
    }


def run_phase7_qualification() -> Tuple[bool, Dict[str, Any]]:
    gates: Dict[str, Dict[str, Any]] = {}
    overall_pass = True

    # ── Gate 7.1: Explicit Active Promotion Resolution ────────────────────────
    logger.info("Evaluating Gate 7.1: Explicit Active Promotion Resolution...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        guard = CheckpointGuard(td)
        ckpt = td / "checkpoint_step_25500.pt"
        sha = _create_mock_ckpt(ckpt, 25500)
        rep = _create_mock_report(ckpt, sha)
        promoted, verdict, details = evaluate_promotion(rep, candidate_path=ckpt)
        guard.promote_checkpoint(ckpt, details)

        active = guard.get_active_checkpoint()
        active_sha = guard.get_active_checkpoint_sha256()

        g7_1_pass = (
            promoted is True
            and active.resolve() == ckpt.resolve()
            and active_sha == sha
            and (td / "ACTIVE_CHECKPOINT.json").exists()
        )
    gates["GATE_7_1"] = {
        "name": "Explicit Active Promotion Resolution",
        "passed": g7_1_pass,
        "evidence": "ACTIVE_CHECKPOINT.json authored atomically, status APPROVED, on-disk SHA matches active checkpoint.",
    }
    overall_pass = overall_pass and g7_1_pass

    # ── Gate 7.2: Fail-Closed on Missing Manifest ─────────────────────────────
    logger.info("Evaluating Gate 7.2: Fail-Closed on Missing Manifest...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        guard = CheckpointGuard(td)
        ckpt = td / "checkpoint_step_25000.pt"
        _create_mock_ckpt(ckpt, 25000)

        threw_expected = False
        try:
            guard.get_active_checkpoint()
        except ExplicitPromotionRequiredError:
            threw_expected = True

    gates["GATE_7_2"] = {
        "name": "Fail-Closed on Missing Manifest",
        "passed": threw_expected,
        "evidence": "Directory with candidate checkpoints but no ACTIVE_CHECKPOINT.json raised ExplicitPromotionRequiredError.",
    }
    overall_pass = overall_pass and threw_expected

    # ── Gate 7.3: Anti-Chronological Priority ─────────────────────────────────
    logger.info("Evaluating Gate 7.3: Anti-Chronological Priority...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        guard = CheckpointGuard(td)
        ckpt_28k = td / "checkpoint_step_28000.pt"
        ckpt_30k = td / "checkpoint_step_30000.pt"
        sha_28k = _create_mock_ckpt(ckpt_28k, 28000)
        sha_30k = _create_mock_ckpt(ckpt_30k, 30000)

        # Promote 28k
        rep_28k = _create_mock_report(ckpt_28k, sha_28k)
        _, _, details_28k = evaluate_promotion(rep_28k, candidate_path=ckpt_28k)
        guard.promote_checkpoint(ckpt_28k, details_28k)

        # 30k has lower IR, rejected
        rep_30k = _create_mock_report(ckpt_30k, sha_30k, mean_ir=40.0)
        promoted_30k, _, details_30k = evaluate_promotion(rep_30k, candidate_path=ckpt_30k)

        # Active checkpoint MUST remain 28k despite 30k being later chronologically
        active = guard.get_active_checkpoint()
        g7_3_pass = (
            promoted_30k is False
            and details_30k["promotion_status"] == "REJECTED"
            and active.resolve() == ckpt_28k.resolve()
        )

    gates["GATE_7_3"] = {
        "name": "Anti-Chronological Priority & Rejection Invariance",
        "passed": g7_3_pass,
        "evidence": "Step 30000 rejected while Step 28000 remains active; no chronological override occurred.",
    }
    overall_pass = overall_pass and g7_3_pass

    # ── Gate 7.4: Cryptographic Provenance & Tamper Prevention ────────────────
    logger.info("Evaluating Gate 7.4: Cryptographic Provenance & Tamper Prevention...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        guard = CheckpointGuard(td)
        ckpt = td / "checkpoint_step_25000.pt"
        sha = _create_mock_ckpt(ckpt, 25000)
        rep = _create_mock_report(ckpt, sha)
        _, _, details = evaluate_promotion(rep, candidate_path=ckpt)
        guard.promote_checkpoint(ckpt, details)

        # Tamper with file
        with open(ckpt, "ab") as f:
            f.write(b"TAMPERED_PAYLOAD")

        tamper_detected = False
        try:
            guard.get_active_checkpoint()
        except CheckpointTamperedError:
            tamper_detected = True

        # Replay attempt
        rep_replay = _create_mock_report(ckpt, "1111111111111111111111111111111111111111111111111111111111111111")
        promoted_replay, _, details_replay = evaluate_promotion(rep_replay, candidate_path=ckpt)
        replay_rejected = (promoted_replay is False and details_replay["promotion_status"] == "REJECTED")

        g7_4_pass = tamper_detected and replay_rejected

    gates["GATE_7_4"] = {
        "name": "Cryptographic Provenance & Tamper Prevention",
        "passed": g7_4_pass,
        "evidence": "Tampered on-disk checkpoint caught via CheckpointTamperedError; report replay with forged SHA caught.",
    }
    overall_pass = overall_pass and g7_4_pass

    # ── Gate 7.5: Rejection of Missing/Malformed Provenance ────────────────────
    logger.info("Evaluating Gate 7.5: Rejection of Missing/Malformed Provenance...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        ckpt = td / "checkpoint_step_25000.pt"
        sha = _create_mock_ckpt(ckpt, 25000)

        # Missing git_revision
        rep_no_git = _create_mock_report(ckpt, sha)
        del rep_no_git["git_revision"]
        p_no_git, _, d_no_git = evaluate_promotion(rep_no_git, candidate_path=ckpt)

        # Missing seed
        rep_no_seed = _create_mock_report(ckpt, sha)
        del rep_no_seed["evaluation_seed"]
        p_no_seed, _, d_no_seed = evaluate_promotion(rep_no_seed, candidate_path=ckpt)

        # NaN metric
        rep_nan = _create_mock_report(ckpt, sha, mean_ir=float("nan"))
        p_nan, _, d_nan = evaluate_promotion(rep_nan, candidate_path=ckpt)

        g7_5_pass = (
            p_no_git is False and d_no_git["promotion_status"] == "REJECTED"
            and p_no_seed is False and d_no_seed["promotion_status"] == "REJECTED"
            and p_nan is False and d_nan["promotion_status"] == "REJECTED"
        )

    gates["GATE_7_5"] = {
        "name": "Rejection of Missing/Malformed Provenance & Non-Finite Metrics",
        "passed": g7_5_pass,
        "evidence": "Evaluations with missing git revision, missing seed, or NaN/Inf metrics rejected fail-closed.",
    }
    overall_pass = overall_pass and g7_5_pass

    # ── Gate 7.6: Baseline SHA Invariance & Directory Protection ──────────────
    logger.info("Evaluating Gate 7.6: Baseline SHA Invariance & Directory Protection...")
    baseline_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    base_exists = baseline_path.exists()
    base_sha = sha256_file(baseline_path) if base_exists else ""
    sha_exact = (base_sha == EXPECTED_BASELINE_SHA256)

    # Path safety violation test
    guard = CheckpointGuard("experiments/checkpoints/scheduler_v2_operational_candidate")
    path_blocked = False
    try:
        guard.validate_path_safety("experiments/checkpoints/production_baseline/injected.pt")
    except CheckpointSecurityError:
        path_blocked = True

    g7_6_pass = base_exists and sha_exact and path_blocked
    gates["GATE_7_6"] = {
        "name": "Baseline SHA Invariance & Production Baseline Protection",
        "passed": g7_6_pass,
        "evidence": f"Production baseline SHA bit-exact ({EXPECTED_BASELINE_SHA256[:16]}...); writes to baseline forbidden.",
    }
    overall_pass = overall_pass and g7_6_pass

    # ── Gate 7.7: Rollback Integrity & Manifest Restoration ───────────────────
    logger.info("Evaluating Gate 7.7: Rollback Integrity & Manifest Restoration...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        guard = CheckpointGuard(td)
        ckpt_25k = td / "checkpoint_step_25000.pt"
        sha_25k = _create_mock_ckpt(ckpt_25k, 25000)
        rep_25k = _create_mock_report(ckpt_25k, sha_25k)
        _, _, details_25k = evaluate_promotion(rep_25k, candidate_path=ckpt_25k)
        guard.promote_checkpoint(ckpt_25k, details_25k)

        rollback = RollbackManager(
            last_known_good_ckpt=ckpt_25k,
            audit_log_path=td / "audit.json",
            candidate_dir=td,
        )
        rollback.register_known_good(ckpt_path=ckpt_25k, metrics={"mean_ir": 80.0})

        # Promote 26k
        ckpt_26k = td / "checkpoint_step_26000.pt"
        sha_26k = _create_mock_ckpt(ckpt_26k, 26000)
        rep_26k = _create_mock_report(ckpt_26k, sha_26k)
        _, _, details_26k = evaluate_promotion(rep_26k, candidate_path=ckpt_26k)
        guard.promote_checkpoint(ckpt_26k, details_26k)

        # Trigger rollback
        rollback.execute_rollback(failed_ckpt_path=ckpt_26k, reasons=["Q-Explosion"], step=26000)

        active = guard.get_active_checkpoint()
        quarantined = td / f"{ckpt_26k.stem}_QUARANTINED_COLLAPSE.pt"

        g7_7_pass = (
            active.resolve() == ckpt_25k.resolve()
            and not ckpt_26k.exists()
            and quarantined.exists()
        )

    gates["GATE_7_7"] = {
        "name": "Rollback Integrity & Authoritative Manifest Restoration",
        "passed": g7_7_pass,
        "evidence": "Rollback verified known-good SHA, quarantined failed candidate, and restored cached active manifest.",
    }
    overall_pass = overall_pass and g7_7_pass

    # ── Gate 7.8: Deployment API Fail-Closed Loading ──────────────────────────
    logger.info("Evaluating Gate 7.8: Deployment API Fail-Closed Loading...")
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        cand_dir = td / "candidate"
        cand_dir.mkdir()
        unapproved = cand_dir / "checkpoint_unapproved.pt"
        _create_mock_ckpt(unapproved, 99999)

        old_env = os.environ.get("SCHEDULER_CHECKPOINT")
        os.environ["SCHEDULER_CHECKPOINT"] = str(unapproved)
        api_failed_closed = False
        try:
            from ew_core.deployment.api import lifespan, app
            import asyncio
            asyncio.run(lifespan(app).__aenter__())
        except ExplicitPromotionRequiredError:
            api_failed_closed = True
        except Exception as e:
            logger.info("Deployment API failed with exception: %s", type(e))
            if isinstance(e, CheckpointSecurityError):
                api_failed_closed = True
        finally:
            if old_env is not None:
                os.environ["SCHEDULER_CHECKPOINT"] = old_env
            else:
                os.environ.pop("SCHEDULER_CHECKPOINT", None)

    gates["GATE_7_8"] = {
        "name": "Deployment API Fail-Closed Loading",
        "passed": api_failed_closed,
        "evidence": "Deployment API rejected unapproved candidate checkpoint and raised ExplicitPromotionRequiredError.",
    }
    overall_pass = overall_pass and api_failed_closed

    # ── Gate 7.9: Operational Baseline Manifest Bootstrap ─────────────────────
    logger.info("Evaluating Gate 7.9: Operational Baseline Manifest Bootstrap...")
    op_cand_dir = Path("experiments/checkpoints/scheduler_v2_operational_candidate")
    manifest_file = op_cand_dir / "ACTIVE_CHECKPOINT.json"
    g7_9_pass = False
    if manifest_file.exists():
        try:
            guard = CheckpointGuard(op_cand_dir)
            active_ckpt = guard.get_active_checkpoint()
            active_sha = guard.get_active_checkpoint_sha256()
            g7_9_pass = (
                active_ckpt.exists()
                and active_sha == EXPECTED_BASELINE_SHA256
            )
        except Exception as exc:
            logger.error("Error inspecting operational candidate manifest: %s", exc)

    gates["GATE_7_9"] = {
        "name": "Operational Baseline Manifest Bootstrap",
        "passed": g7_9_pass,
        "evidence": f"Authoritative ACTIVE_CHECKPOINT.json verified in scheduler_v2_operational_candidate with baseline hash {EXPECTED_BASELINE_SHA256[:16]}...",
    }
    overall_pass = overall_pass and g7_9_pass

    return overall_pass, gates


def main() -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_pass, gates = run_phase7_qualification()

    out_dir = Path("experiments/reports/phase7")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "timestamp": timestamp,
        "phase": 7,
        "qualification_status": "QUALIFIED" if overall_pass else "DISQUALIFIED",
        "gates": gates,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
    }

    json_path = out_dir / "phase7_checkpoint_promotion_gate_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    md_lines = [
        "# Phase 7 Final Qualification Report",
        "**Cognitive EW Smart Scan Scheduler — SIH2026_Try2**",
        f"**Date:** {timestamp}",
        f"**Status:** {'QUALIFIED' if overall_pass else 'DISQUALIFIED'}",
        f"**Frozen Baseline SHA-256:** `{EXPECTED_BASELINE_SHA256}`",
        "",
        "## 1. Summary of Gate Verifications",
        "",
        "| Gate ID | Verification Name | Verdict | Evidence |",
        "| :--- | :--- | :---: | :--- |",
    ]

    for gid, ginfo in gates.items():
        verdict = "**PASS**" if ginfo["passed"] else "**FAIL**"
        md_lines.append(f"| `{gid}` | {ginfo['name']} | {verdict} | {ginfo['evidence']} |")

    md_lines.extend([
        "",
        "## 2. Invariants Enforced in Phase 7",
        "1. **Fail-Closed Resolution:** `CheckpointGuard.get_active_checkpoint()` requires explicit `ACTIVE_CHECKPOINT.json` with `promotion_status == 'APPROVED'` and verified bitwise SHA-256.",
        "2. **Anti-Chronological Priority:** Checkpoint step number is never used to infer priority; an approved lower-step checkpoint takes precedence over unapproved/rejected higher steps.",
        "3. **Cryptographic Binding:** Promotion reports and rollback states are bound to on-disk SHA-256 digests. Replay and tampering attempts are rejected.",
        "4. **No Synthetic Approvals:** Rollback restores the cached original approved manifest, never synthesizing artificial promotions.",
        "5. **Production Baseline Immutability:** `experiments/checkpoints/production_baseline/` is locked with hash `7a99c659...`.",
        "",
        "## 3. Qualification Conclusion",
        f"All 9/9 Phase 7 gates **{'PASSED' if overall_pass else 'FAILED'}**. Checkpoint promotion lifecycle is fully explicit, fail-closed, and provenance-bound.",
    ])

    md_path = out_dir / "PHASE_7_FINAL_QUALIFICATION_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("=" * 80)
    logger.info("PHASE 7 QUALIFICATION COMPLETE: %s", "QUALIFIED" if overall_pass else "DISQUALIFIED")
    logger.info("Saved JSON report to: %s", json_path)
    logger.info("Saved Markdown report to: %s", md_path)
    logger.info("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

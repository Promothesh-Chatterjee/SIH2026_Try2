"""
Phase 7 Dedicated Test Suite: Checkpoint Promotion & Provenance-Bound Lifecycle.

Tests all 16 verification invariants:
1. Manifest selects approved checkpoint regardless of step chronology.
2. Chronology inversion: step 30k rejected, step 28k approved -> step 28k active.
3. Unevaluated inversion: step 31k unevaluated, step 28k approved -> step 28k active.
4. No manifest fails closed (ExplicitPromotionRequiredError).
5. Quarantined checkpoint excluded / raises QuarantinedCheckpointError.
6. Tampered active checkpoint detected (CheckpointTamperedError).
7. Promotion report replay rejected (report SHA != file SHA).
8. Missing provenance rejected (missing git rev, seed).
9. Malformed / NaN / Inf metrics rejected.
10. Baseline SHA mismatch blocks promotion.
11. Atomic manifest update leaves no corrupt state.
12. Idempotent promotion preserves valid active state.
13. Rollback verifies known-good SHA and restores original manifest.
14. Tampered known-good causes rollback to fail closed.
15. Frozen baseline immutability check.
16. Deployment loader rejects unapproved scheduler checkpoint (fail-closed test).
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from ew_core.evaluation.promotion_sentinel import evaluate_promotion
from ew_core.training.safety.checkpoint_guard import (
    CheckpointGuard,
    CheckpointSecurityError,
    CheckpointTamperedError,
    ExplicitPromotionRequiredError,
    EXPECTED_BASELINE_SHA256,
    QuarantinedCheckpointError,
)
from ew_core.training.safety.rollback_manager import RollbackManager


def _create_mock_checkpoint(path: Path, step: int) -> str:
    payload = {"step": step, "weights": torch.randn(5, 5)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_mock_report(
    ckpt_path: Path,
    ckpt_sha: str,
    status: str = "APPROVED",
    mean_ir: float = 80.0,
    agile_ir: float = 82.0,
    sparse_ir: float = 78.0,
    worst_case_ir: float = 75.0,
    pfa: float = 0.0005,
    pd: float = 99.85,
    mode2_agile: float = 0.20,
    head_drift: float = 0.0,
    baseline_sha: str = EXPECTED_BASELINE_SHA256,
    git_rev: str = "abc1234",
    seed: int = 42,
    units: dict = None,
) -> Dict[str, Any]:
    if units is None:
        units = {
            "mean_ir": "percent",
            "agile_ir": "percent",
            "sparse_ir": "percent",
            "worst_case_ir": "percent",
            "pfa": "fraction",
        }
    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "baseline_checkpoint_sha256": baseline_sha,
        "git_revision": git_rev,
        "config_sha256": "def5678",
        "evaluation_seed": seed,
        "metric_units": units,
        "scenario_summary": {
            "mean_ir": mean_ir,
            "agile_ir": agile_ir,
            "sparse_ir": sparse_ir,
            "worst_case_ir": worst_case_ir,
            "decision_pd": pd,
            "pfa": pfa,
            "mean_pfa": pfa,
            "mode_proportions": [0.3, 0.4, mode2_agile, 0.05, 0.05],
            "mode_proportions_agile_sparse": [0.2, 0.3, mode2_agile, 0.05, 0.05],
        },
        "action_summary": {
            "mode2_fraction_agile_sparse": mode2_agile,
            "action_entropy": 2.5,
            "distinct_bands": 36.0,
        },
        "training_diagnostics": {
            "advantage_head_drift": head_drift,
            "q_loss": 0.05,
            "q_max": 10.0,
        },
    }


# Test 1: Manifest selects approved checkpoint regardless of step chronology
def test_manifest_selects_approved_checkpoint_regardless_of_chronology(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt_10k = tmp_path / "checkpoint_step_10000.pt"
    ckpt_20k = tmp_path / "checkpoint_step_20000.pt"
    sha_10k = _create_mock_checkpoint(ckpt_10k, 10000)
    sha_20k = _create_mock_checkpoint(ckpt_20k, 20000)

    report = _create_mock_report(ckpt_10k, sha_10k)
    promoted, verdict, details = evaluate_promotion(report, candidate_path=ckpt_10k)
    assert promoted is True
    guard.promote_checkpoint(ckpt_10k, details)

    active = guard.get_active_checkpoint()
    assert active.resolve() == ckpt_10k.resolve()
    assert guard.get_active_checkpoint_sha256() == sha_10k


# Test 2: Chronology inversion: step 30k rejected, step 28k approved -> step 28k active
def test_chronology_inversion_step_30k_rejected_step_28k_approved(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt_28k = tmp_path / "checkpoint_step_28000.pt"
    ckpt_30k = tmp_path / "checkpoint_step_30000.pt"
    sha_28k = _create_mock_checkpoint(ckpt_28k, 28000)
    sha_30k = _create_mock_checkpoint(ckpt_30k, 30000)

    # Promote 28k
    rep_28k = _create_mock_report(ckpt_28k, sha_28k)
    promoted_28k, _, details_28k = evaluate_promotion(rep_28k, candidate_path=ckpt_28k)
    assert promoted_28k is True
    guard.promote_checkpoint(ckpt_28k, details_28k)

    # 30k fails promotion (e.g. Mean IR below gate)
    rep_30k = _create_mock_report(ckpt_30k, sha_30k, mean_ir=40.0)
    promoted_30k, _, details_30k = evaluate_promotion(rep_30k, candidate_path=ckpt_30k)
    assert promoted_30k is False
    assert details_30k["promotion_status"] == "REJECTED"

    # Attempting to promote rejected candidate must fail closed
    with pytest.raises(ValueError, match="Cannot promote checkpoint with non-APPROVED status"):
        guard.promote_checkpoint(ckpt_30k, details_30k)

    # Active checkpoint MUST remain 28k
    active = guard.get_active_checkpoint()
    assert active.resolve() == ckpt_28k.resolve()


# Test 3: Unevaluated inversion: step 31k unevaluated, step 28k approved -> step 28k active
def test_unevaluated_inversion_step_31k_unevaluated_step_28k_approved(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt_28k = tmp_path / "checkpoint_step_28000.pt"
    ckpt_31k = tmp_path / "checkpoint_step_31000.pt"
    sha_28k = _create_mock_checkpoint(ckpt_28k, 28000)
    sha_31k = _create_mock_checkpoint(ckpt_31k, 31000)

    # Promote 28k
    rep_28k = _create_mock_report(ckpt_28k, sha_28k)
    promoted_28k, _, details_28k = evaluate_promotion(rep_28k, candidate_path=ckpt_28k)
    guard.promote_checkpoint(ckpt_28k, details_28k)

    # 31k exists on disk but is unevaluated
    active = guard.get_active_checkpoint()
    assert active.resolve() == ckpt_28k.resolve()


# Test 4: No manifest fails closed (ExplicitPromotionRequiredError)
def test_no_manifest_fails_closed(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    _create_mock_checkpoint(ckpt, 25000)

    with pytest.raises(ExplicitPromotionRequiredError, match="No explicit active promotion manifest"):
        guard.get_active_checkpoint()


# Test 5: Quarantined checkpoint excluded / raises QuarantinedCheckpointError
def test_quarantined_checkpoint_excluded(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha)
    promoted, _, details = evaluate_promotion(rep, candidate_path=ckpt)
    guard.promote_checkpoint(ckpt, details)

    # Move checkpoint to quarantine directory
    quarantine_dir = tmp_path / ".quarantine"
    quarantine_dir.mkdir()
    quarantined_file = quarantine_dir / ckpt.name
    shutil.move(ckpt, quarantined_file)

    with pytest.raises(QuarantinedCheckpointError):
        guard.get_active_checkpoint()


# Test 6: Tampered active checkpoint detected (CheckpointTamperedError)
def test_tampered_active_checkpoint_detected(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha)
    promoted, _, details = evaluate_promotion(rep, candidate_path=ckpt)
    guard.promote_checkpoint(ckpt, details)

    # Tamper with the checkpoint on disk
    with open(ckpt, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    with pytest.raises(CheckpointTamperedError, match="CRITICAL INTEGRITY FAILURE"):
        guard.get_active_checkpoint()


# Test 7: Promotion report replay rejected (report SHA != file SHA)
def test_promotion_report_replay_rejected(tmp_path):
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    # Fake SHA in report
    rep = _create_mock_report(ckpt, "0000000000000000000000000000000000000000000000000000000000000000")
    promoted, verdict, details = evaluate_promotion(rep, candidate_path=ckpt)
    assert promoted is False
    assert details["promotion_status"] == "REJECTED"
    assert "replay" in verdict.lower() or "mismatch" in verdict.lower()


# Test 8: Missing provenance rejected (missing git rev, seed)
def test_missing_provenance_rejected(tmp_path):
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha)
    del rep["git_revision"]

    promoted, verdict, details = evaluate_promotion(rep, candidate_path=ckpt)
    assert promoted is False
    assert details["promotion_status"] == "REJECTED"
    assert "git_revision" in verdict.lower()


# Test 9: Malformed / NaN / Inf metrics rejected
def test_malformed_or_nan_inf_metrics_rejected(tmp_path):
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha, mean_ir=float("nan"))
    promoted, verdict, details = evaluate_promotion(rep, candidate_path=ckpt)
    assert promoted is False
    assert details["promotion_status"] == "REJECTED"
    assert "nan" in verdict.lower() or "invalid" in verdict.lower() or "non-finite" in verdict.lower()

    rep_inf = _create_mock_report(ckpt, sha, mean_ir=float("inf"))
    promoted_inf, verdict_inf, details_inf = evaluate_promotion(rep_inf, candidate_path=ckpt)
    assert promoted_inf is False
    assert details_inf["promotion_status"] == "REJECTED"


# Test 10: Baseline SHA mismatch blocks promotion
def test_baseline_sha_mismatch_blocks_promotion(tmp_path):
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(
        ckpt,
        sha,
        baseline_sha="1111111111111111111111111111111111111111111111111111111111111111",
    )
    promoted, verdict, details = evaluate_promotion(rep, candidate_path=ckpt)
    assert promoted is False
    assert details["promotion_status"] == "REJECTED"
    assert "baseline" in verdict.lower()


# Test 11: Atomic manifest update leaves no corrupt state
def test_atomic_manifest_update_leaves_no_corrupt_state(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha)
    promoted, _, details = evaluate_promotion(rep, candidate_path=ckpt)
    guard.promote_checkpoint(ckpt, details)

    tmp_manifest = tmp_path / "ACTIVE_CHECKPOINT.json.tmp"
    assert not tmp_manifest.exists()

    manifest_file = tmp_path / "ACTIVE_CHECKPOINT.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["checkpoint_sha256"] == sha
    assert data["promotion_status"] == "APPROVED"


# Test 12: Idempotent promotion preserves valid active state
def test_idempotent_promotion_preserves_valid_active_state(tmp_path):
    guard = CheckpointGuard(tmp_path)
    ckpt = tmp_path / "checkpoint_step_25000.pt"
    sha = _create_mock_checkpoint(ckpt, 25000)

    rep = _create_mock_report(ckpt, sha)
    _, _, details = evaluate_promotion(rep, candidate_path=ckpt)

    # First promotion
    guard.promote_checkpoint(ckpt, details)
    active1 = guard.get_active_checkpoint()

    # Second promotion of identical checkpoint
    guard.promote_checkpoint(ckpt, details)
    active2 = guard.get_active_checkpoint()

    assert active1.resolve() == active2.resolve()


# Test 13: Rollback verifies known-good SHA and restores original manifest
def test_rollback_verifies_known_good_and_restores_original_manifest(tmp_path):
    guard = CheckpointGuard(tmp_path)

    ckpt_25k = tmp_path / "checkpoint_step_25000.pt"
    sha_25k = _create_mock_checkpoint(ckpt_25k, 25000)
    rep_25k = _create_mock_report(ckpt_25k, sha_25k)
    _, _, details_25k = evaluate_promotion(rep_25k, candidate_path=ckpt_25k)
    guard.promote_checkpoint(ckpt_25k, details_25k)

    # Initialize RollbackManager and register known-good
    rollback = RollbackManager(
        last_known_good_ckpt=ckpt_25k,
        audit_log_path=tmp_path / "rollback_audit.json",
        candidate_dir=tmp_path,
    )
    rollback.register_known_good(ckpt_path=ckpt_25k, metrics={"mean_ir": 80.0})

    # Promote 26k
    ckpt_26k = tmp_path / "checkpoint_step_26000.pt"
    sha_26k = _create_mock_checkpoint(ckpt_26k, 26000)
    rep_26k = _create_mock_report(ckpt_26k, sha_26k)
    _, _, details_26k = evaluate_promotion(rep_26k, candidate_path=ckpt_26k)
    guard.promote_checkpoint(ckpt_26k, details_26k)
    assert guard.get_active_checkpoint().resolve() == ckpt_26k.resolve()

    # Trigger rollback due to safety failure
    rollback.execute_rollback(failed_ckpt_path=ckpt_26k, reasons=["Q-Explosion"], step=26000)

    # Verify 25k is active again and 26k is quarantined
    active = guard.get_active_checkpoint()
    assert active.resolve() == ckpt_25k.resolve()
    assert not ckpt_26k.exists()
    quarantined = tmp_path / f"{ckpt_26k.stem}_QUARANTINED_COLLAPSE.pt"
    assert quarantined.exists()


# Test 14: Tampered known-good causes rollback to fail closed
def test_tampered_known_good_causes_rollback_fail_closed(tmp_path):
    guard = CheckpointGuard(tmp_path)

    ckpt_25k = tmp_path / "checkpoint_step_25000.pt"
    sha_25k = _create_mock_checkpoint(ckpt_25k, 25000)
    rep_25k = _create_mock_report(ckpt_25k, sha_25k)
    _, _, details_25k = evaluate_promotion(rep_25k, candidate_path=ckpt_25k)
    guard.promote_checkpoint(ckpt_25k, details_25k)

    rollback = RollbackManager(
        last_known_good_ckpt=ckpt_25k,
        audit_log_path=tmp_path / "rollback_audit.json",
        candidate_dir=tmp_path,
    )
    rollback.register_known_good(ckpt_path=ckpt_25k, metrics={"mean_ir": 80.0})

    ckpt_26k = tmp_path / "checkpoint_step_26000.pt"
    sha_26k = _create_mock_checkpoint(ckpt_26k, 26000)

    # Tamper with 25k on disk
    with open(ckpt_25k, "ab") as f:
        f.write(b"CORRUPTION")

    # Rollback must fail closed
    with pytest.raises(RuntimeError, match="CRITICAL INTEGRITY FAILURE"):
        rollback.execute_rollback(failed_ckpt_path=ckpt_26k, reasons=["Divergence"], step=26000)


# Test 15: Frozen baseline immutability check
def test_frozen_baseline_immutability():
    baseline_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    assert baseline_path.exists(), "Frozen baseline file must exist"
    actual_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_BASELINE_SHA256, "Frozen baseline SHA-256 must be bit-exact"

    # Attempting to validate path safety targeting production_baseline must fail closed
    guard = CheckpointGuard("experiments/checkpoints/scheduler_v2_operational_candidate")
    with pytest.raises(CheckpointSecurityError, match="CRITICAL PATH SAFETY VIOLATION"):
        guard.validate_path_safety("experiments/checkpoints/production_baseline/override.pt")


# Test 16: Deployment loader rejects unapproved scheduler checkpoint (fail-closed test)
def test_deployment_loader_rejects_unapproved_scheduler_checkpoint(tmp_path, monkeypatch):
    # Set SCHEDULER_CHECKPOINT to an unpromoted candidate file in a candidate directory
    cand_dir = tmp_path / "candidate"
    cand_dir.mkdir()
    unapproved_ckpt = cand_dir / "checkpoint_step_99999.pt"
    _create_mock_checkpoint(unapproved_ckpt, 99999)

    monkeypatch.setenv("SCHEDULER_CHECKPOINT", str(unapproved_ckpt))

    # Test api loading fail-closed behavior
    from ew_core.deployment.api import lifespan, app

    with pytest.raises(ExplicitPromotionRequiredError):
        # Trigger lifespan startup
        import asyncio
        asyncio.run(lifespan(app).__aenter__())

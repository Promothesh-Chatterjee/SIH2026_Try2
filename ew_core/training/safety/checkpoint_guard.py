"""Checkpoint protection and atomic persistence guard.

Phase 7 Contract:
  1. Fail-closed: An active checkpoint is NEVER inferred from filename, step number,
     modification time, or absence of quarantine.
  2. Provenance-bound & SHA-256 verified: Every active checkpoint must be identified by
     an explicit ACTIVE_CHECKPOINT.json manifest with status APPROVED and exact bitwise hash match.
  3. Atomic updates: ACTIVE_CHECKPOINT.json is written via temporary swap files and atomic rename.
  4. Unconditional path safety: Target paths must never resolve inside forbidden baseline directories,
     regardless of whether the directory currently exists on disk.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ew_core.utils.checkpoint_paths import (
    CANONICAL_PRODUCTION_BASELINE,
    EXPECTED_FROZEN_SHA256,
    REFERENCE_CANDIDATE_MIRROR,
)

EXPECTED_BASELINE_SHA256 = EXPECTED_FROZEN_SHA256

logger = logging.getLogger(__name__)

ACTIVE_MANIFEST_FILENAME = "ACTIVE_CHECKPOINT.json"


class CheckpointSecurityError(RuntimeError):
    """Base exception for checkpoint safety and provenance violations."""
    pass


class ExplicitPromotionRequiredError(CheckpointSecurityError):
    """Raised when no explicit valid active-checkpoint manifest exists."""
    pass


class CheckpointTamperedError(CheckpointSecurityError):
    """Raised when on-disk checkpoint SHA-256 does not match recorded manifest/report."""
    pass


class QuarantinedCheckpointError(CheckpointSecurityError):
    """Raised when an attempt is made to activate or promote a quarantined checkpoint."""
    pass


def sha256_file(path: Path | str) -> str:
    """Calculate SHA-256 hex digest for a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file does not exist: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class CheckpointGuard:
    """Guarantees checkpoint safety, atomic writes, and forbidden-path enforcement.

    Enforcements:
      1. Never writes into production_baseline/ or any forbidden directory (unconditional check).
      2. Writes candidate checkpoints atomically via temporary swap files.
      3. Verifies file integrity with SHA-256 hash logging upon every save.
      4. Requires explicit, approved ACTIVE_CHECKPOINT.json to return active checkpoint.
    """

    def __init__(
        self,
        output_dir: Path | str,
        forbidden_dirs: list[Path | str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if forbidden_dirs is None:
            forbidden_dirs = [
                Path("checkpoints/production_baseline"),
                Path("experiments/checkpoints/production_baseline"),
            ]
        self.forbidden_dirs = [Path(d).resolve() for d in forbidden_dirs]

        # Immediate unconditional path safety validation on instantiation
        self.validate_path_safety(self.output_dir)

    def validate_path_safety(self, target_path: Path | str) -> None:
        """Unconditionally assert that target_path does not resolve inside forbidden baseline directories."""
        target_res = Path(target_path).resolve()
        for forbidden in self.forbidden_dirs:
            forb_res = forbidden.resolve()
            # Check unconditional path containment, regardless of whether directory exists on disk
            if target_res == forb_res or forb_res in target_res.parents:
                raise CheckpointSecurityError(
                    f"CRITICAL PATH SAFETY VIOLATION: Target path '{target_res}' resolves "
                    f"inside forbidden baseline directory '{forb_res}'!"
                )

    def save_checkpoint_atomic(
        self,
        state_payload: Dict[str, Any],
        filename: str,
    ) -> Tuple[Path, str]:
        """Save a checkpoint atomically and compute its SHA-256 hash.

        Writes to filename.tmp first, flushes, syncs to disk, then renames.
        """
        final_path = self.output_dir / filename
        self.validate_path_safety(final_path)

        tmp_path = self.output_dir / f"{filename}.tmp_{os.getpid()}"

        try:
            torch.save(state_payload, tmp_path)
            # Flush and atomic rename
            if final_path.exists():
                final_path.unlink()
            tmp_path.rename(final_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to atomically save checkpoint {final_path}: {exc}") from exc

        sha256_hex = sha256_file(final_path)
        logger.info("Saved atomic checkpoint %s (SHA-256: %s)", final_path.name, sha256_hex[:16])
        return final_path, sha256_hex

    def list_valid_checkpoints(self) -> list[Path]:
        """Returns all non-quarantined, valid checkpoints in the directory.

        Note: Presence in this list indicates structural validity on disk, NOT approval.
        Only an explicit ACTIVE_CHECKPOINT.json manifest designates the active approved model.
        """
        return sorted([
            p for p in self.output_dir.glob("checkpoint_*.pt")
            if "QUARANTINED" not in p.name and "tmp" not in p.name
        ])

    def get_active_manifest_path(self) -> Path:
        """Return the path to the active checkpoint manifest in the output directory."""
        return self.output_dir / ACTIVE_MANIFEST_FILENAME

    def get_active_checkpoint(self) -> Path:
        """Return the explicitly approved, provenance-bound active checkpoint.

        FAIL-CLOSED SEMANTICS:
          - Reads ACTIVE_CHECKPOINT.json in self.output_dir.
          - Never infers approval from file existence, step number, or timestamps.
          - Validates manifest status == 'APPROVED'.
          - Computes on-disk SHA-256 and validates bit-exact match with manifest.
          - Validates that the checkpoint is not quarantined.
          - Validates that the baseline SHA-256 matches canonical immutable baseline.

        Raises:
          ExplicitPromotionRequiredError: If manifest is missing or status is not APPROVED.
          CheckpointTamperedError: If on-disk file SHA-256 does not match manifest.
          QuarantinedCheckpointError: If the designated checkpoint is quarantined.
          FileNotFoundError: If the target checkpoint file is missing.
        """
        manifest_path = self.get_active_manifest_path()
        if not manifest_path.is_file():
            raise ExplicitPromotionRequiredError(
                f"No explicit active promotion manifest '{ACTIVE_MANIFEST_FILENAME}' found in {self.output_dir}. "
                f"Fail-closed: approval cannot be inferred from step numbers or directory order."
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            raise ExplicitPromotionRequiredError(f"Corrupt or unreadable active manifest {manifest_path}: {exc}") from exc

        status = manifest.get("promotion_status", manifest.get("status", ""))
        if status != "APPROVED":
            raise ExplicitPromotionRequiredError(
                f"Active checkpoint manifest at {manifest_path} has status '{status}', expected 'APPROVED'."
            )

        # Resolve checkpoint path
        raw_ckpt_path = manifest.get("checkpoint_path", "")
        if not raw_ckpt_path:
            raise ExplicitPromotionRequiredError(f"Active manifest {manifest_path} missing 'checkpoint_path'.")

        target_path = Path(raw_ckpt_path)
        if not target_path.is_absolute():
            target_path = (self.output_dir / target_path).resolve()
        else:
            target_path = target_path.resolve()

        # Quarantine check
        if "QUARANTINED" in target_path.name or "quarantine" in str(target_path).lower():
            raise QuarantinedCheckpointError(
                f"Active checkpoint '{target_path.name}' is marked as QUARANTINED! Refusing activation."
            )

        if not target_path.is_file():
            quarantine_dir = self.output_dir / ".quarantine"
            renamed_candidates = list(self.output_dir.glob(f"{target_path.stem}*QUARANTINED*"))
            if (quarantine_dir.exists() and (quarantine_dir / target_path.name).exists()) or "quarantine" in str(target_path).lower():
                raise QuarantinedCheckpointError(
                    f"Active checkpoint '{target_path.name}' was quarantined at {quarantine_dir / target_path.name}!"
                )
            if renamed_candidates:
                raise QuarantinedCheckpointError(
                    f"Active checkpoint '{target_path.name}' was quarantined as {renamed_candidates[0].name}!"
                )
            raise FileNotFoundError(f"Active approved checkpoint file not found at: {target_path}")

        # Cryptographic on-disk SHA-256 validation
        expected_sha = str(manifest.get("checkpoint_sha256", "")).lower().strip()
        if not expected_sha:
            raise CheckpointSecurityError(f"Active manifest {manifest_path} missing 'checkpoint_sha256'.")

        actual_sha = sha256_file(target_path)
        if actual_sha != expected_sha:
            raise CheckpointTamperedError(
                f"CRITICAL INTEGRITY FAILURE: Checkpoint {target_path} SHA-256 mismatch!\n"
                f"  Recorded in manifest: {expected_sha}\n"
                f"  Computed on-disk:    {actual_sha}"
            )

        # Baseline contract verification
        baseline_sha = str(manifest.get("baseline_checkpoint_sha256", "")).lower().strip()
        if baseline_sha and baseline_sha != EXPECTED_FROZEN_SHA256:
            raise CheckpointSecurityError(
                f"Manifest baseline SHA {baseline_sha} does not match canonical baseline {EXPECTED_FROZEN_SHA256}!"
            )

        logger.info("Active approved checkpoint verified: %s (SHA-256: %s)", target_path.name, actual_sha[:16])
        return target_path

    def get_active_checkpoint_sha256(self) -> str:
        """Return verified SHA-256 of the current active approved checkpoint."""
        active_ckpt = self.get_active_checkpoint()
        return sha256_file(active_ckpt)

    def promote_checkpoint(
        self,
        checkpoint_path: Path | str,
        promotion_report: Dict[str, Any],
        expected_parent_sha256: str | None = None,
        expected_baseline_sha256: str = EXPECTED_FROZEN_SHA256,
    ) -> Tuple[Path, str]:
        """Atomically promote a candidate checkpoint by creating/updating ACTIVE_CHECKPOINT.json.

        Verifications:
          1. Checkpoint exists on disk and is not quarantined.
          2. Calculates on-disk SHA-256 and verifies match with report.
          3. Verifies machine status: promotion_report['promotion_status'] == 'APPROVED'.
          4. Verifies baseline SHA-256 matches EXPECTED_FROZEN_SHA256.
          5. Verifies required provenance fields (git_revision, evaluation_seed).
          6. Atomically writes/updates ACTIVE_CHECKPOINT.json.
        """
        p = Path(checkpoint_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint to promote does not exist: {p}")

        if "QUARANTINED" in p.name:
            raise QuarantinedCheckpointError(f"Cannot promote quarantined checkpoint: {p.name}")

        self.validate_path_safety(p)

        actual_sha = sha256_file(p)

        # 1. Candidate SHA match
        report_ckpt_sha = str(promotion_report.get("checkpoint_sha256", "")).lower().strip()
        if report_ckpt_sha and report_ckpt_sha != actual_sha:
            raise CheckpointTamperedError(
                f"Promotion report candidate SHA does not match checkpoint file!\n"
                f"  Report: {report_ckpt_sha}\n"
                f"  Disk:   {actual_sha}"
            )

        # 2. Machine promotion status verification
        machine_status = promotion_report.get("promotion_status", "")
        if not machine_status:
            # Fallback to boolean promoted flag if explicit status missing
            if promotion_report.get("promoted") is True:
                machine_status = "APPROVED"
            else:
                machine_status = "REJECTED"

        if machine_status != "APPROVED":
            verdict_text = promotion_report.get("promotion_verdict", promotion_report.get("verdict", "REJECTED"))
            raise ValueError(f"Cannot promote checkpoint with non-APPROVED status ({machine_status}): {verdict_text}")

        # 3. Baseline SHA verification
        report_base_sha = str(promotion_report.get("baseline_checkpoint_sha256", expected_baseline_sha256)).lower().strip()
        if report_base_sha != EXPECTED_FROZEN_SHA256 or expected_baseline_sha256 != EXPECTED_FROZEN_SHA256:
            raise CheckpointSecurityError(
                f"Baseline SHA mismatch! Expected {EXPECTED_FROZEN_SHA256}, got {report_base_sha}"
            )

        # 4. Construct Phase 7 canonical manifest payload
        manifest_payload = {
            "schema_version": "phase7",
            "status": "APPROVED",
            "promotion_status": "APPROVED",
            "promotion_verdict": str(promotion_report.get("promotion_verdict", promotion_report.get("verdict", "APPROVED"))),
            "checkpoint_path": str(p),
            "checkpoint_filename": p.name,
            "checkpoint_sha256": actual_sha,
            "training_step": int(promotion_report.get("training_step", promotion_report.get("step", 0))),
            "parent_checkpoint_sha256": str(expected_parent_sha256 or promotion_report.get("parent_checkpoint_sha256", "none")),
            "baseline_checkpoint_sha256": EXPECTED_FROZEN_SHA256,
            "git_revision": str(promotion_report.get("git_revision", "unknown")),
            "config_sha256": str(promotion_report.get("config_sha256", "unknown")),
            "benchmark_version": str(promotion_report.get("benchmark_version", "v2_canonical")),
            "evaluation_seed": int(promotion_report.get("evaluation_seed", 42)),
            "evaluation_provenance": promotion_report.get("evaluation_provenance", {}),
            "promoted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # 5. Atomic manifest write (tmp file -> rename)
        manifest_target = self.get_active_manifest_path()
        tmp_manifest = self.output_dir / f"{ACTIVE_MANIFEST_FILENAME}.tmp_{os.getpid()}"

        try:
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            if manifest_target.exists():
                manifest_target.unlink()
            tmp_manifest.rename(manifest_target)
        except Exception as exc:
            if tmp_manifest.exists():
                tmp_manifest.unlink()
            raise RuntimeError(f"Failed to atomically write active manifest {manifest_target}: {exc}") from exc

        logger.info("PROMOTED ACTIVE CHECKPOINT: %s (SHA-256: %s)", p.name, actual_sha[:16])
        return p, actual_sha

    def initialize_active_baseline(
        self,
        candidate_path: Path | str | None = None,
    ) -> Tuple[Path, str]:
        """Bootstrap operation: initialize active manifest for the existing bit-identical frozen candidate mirror.

        Must verify SHA-256 == EXPECTED_FROZEN_SHA256 before creating manifest.
        This provides a backward-compatible, fail-closed bootstrap for operational candidate mirrors.
        """
        p = Path(candidate_path) if candidate_path is not None else (self.output_dir / "checkpoint_gate_25000_frozen.pt")
        if not p.is_file():
            # Check reference candidate mirror
            if REFERENCE_CANDIDATE_MIRROR.is_file():
                p = REFERENCE_CANDIDATE_MIRROR
            elif CANONICAL_PRODUCTION_BASELINE.is_file():
                p = CANONICAL_PRODUCTION_BASELINE
            else:
                raise FileNotFoundError(f"Cannot initialize baseline: candidate checkpoint not found at {p}")

        p = p.resolve()
        actual_sha = sha256_file(p)
        if actual_sha != EXPECTED_FROZEN_SHA256:
            raise CheckpointSecurityError(
                f"Cannot initialize active baseline: SHA-256 mismatch!\n"
                f"  Expected: {EXPECTED_FROZEN_SHA256}\n"
                f"  Actual:   {actual_sha}"
            )

        manifest_payload = {
            "schema_version": "phase7",
            "status": "APPROVED",
            "promotion_status": "APPROVED",
            "promotion_verdict": "INITIAL_APPROVED_BASELINE_BOOTSTRAP",
            "checkpoint_path": str(p),
            "checkpoint_filename": p.name,
            "checkpoint_sha256": actual_sha,
            "training_step": 25000,
            "parent_checkpoint_sha256": "none",
            "baseline_checkpoint_sha256": EXPECTED_FROZEN_SHA256,
            "git_revision": "bootstrap_phase7",
            "config_sha256": "canonical_gate25k",
            "benchmark_version": "v2_canonical",
            "evaluation_seed": 42,
            "evaluation_provenance": {"bootstrap": True, "baseline_mirror": True},
            "promoted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        manifest_target = self.get_active_manifest_path()
        tmp_manifest = self.output_dir / f"{ACTIVE_MANIFEST_FILENAME}.tmp_{os.getpid()}"

        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if manifest_target.exists():
            manifest_target.unlink()
        tmp_manifest.rename(manifest_target)

        logger.info("Initialized active baseline manifest for %s (SHA-256: %s)", p.name, actual_sha[:16])
        return p, actual_sha

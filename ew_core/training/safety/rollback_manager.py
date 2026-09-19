"""Rollback manager for candidate checkpoint recovery and audit logging.

Phase 7 Contract:
  1. Cryptographic binding: Known-good state is bound to its computed SHA-256.
  2. Fail-closed rollback: Before restoring or activating the known-good checkpoint,
     re-computes SHA-256 on disk. If modified or corrupted, halts with CRITICAL integrity failure.
  3. No fabricated approvals: Retains the previously approved active manifest and restores
     it atomically upon rollback, rather than synthesizing a fresh promotion record.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ACTIVE_MANIFEST_FILENAME = "ACTIVE_CHECKPOINT.json"


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


class RollbackManager:
    """Retains the last known-good checkpoint and executes automatic rollback upon safety halt."""

    def __init__(
        self,
        last_known_good_ckpt: Path | str,
        audit_log_path: Path | str,
        candidate_dir: Path | str | None = None,
    ) -> None:
        self.last_known_good_ckpt = Path(last_known_good_ckpt).resolve()
        if not self.last_known_good_ckpt.is_file():
            raise FileNotFoundError(f"Initial known-good checkpoint not found: {self.last_known_good_ckpt}")

        if "QUARANTINED" in self.last_known_good_ckpt.name:
            raise ValueError(f"Cannot initialize RollbackManager with quarantined checkpoint: {self.last_known_good_ckpt.name}")

        self.last_known_good_sha256 = sha256_file(self.last_known_good_ckpt)
        self.candidate_dir = Path(candidate_dir).resolve() if candidate_dir is not None else self.last_known_good_ckpt.parent

        # Retain last known-good approved manifest
        self.last_known_good_manifest: Optional[Dict[str, Any]] = None
        manifest_cand = self.candidate_dir / ACTIVE_MANIFEST_FILENAME
        if manifest_cand.is_file():
            try:
                with open(manifest_cand, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    if m.get("status") == "APPROVED" or m.get("promotion_status") == "APPROVED":
                        self.last_known_good_manifest = m
            except Exception as exc:
                logger.warning("Could not load existing manifest for rollback tracking: %s", exc)

        self.audit_log_path = Path(audit_log_path).resolve()
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        if self.audit_log_path.exists():
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("rollback_events", [])
            except Exception:
                self.history = []

    def register_known_good(
        self,
        ckpt_path: Path | str,
        metrics: Dict[str, Any],
        manifest_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Promote a newly accepted staged checkpoint to the last known-good status."""
        p = Path(ckpt_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint to register does not exist: {p}")

        if "QUARANTINED" in p.name:
            raise ValueError(f"Cannot register quarantined checkpoint as known-good: {p.name}")

        sha = sha256_file(p)
        self.last_known_good_ckpt = p
        self.last_known_good_sha256 = sha

        # Retain manifest
        if manifest_payload is not None:
            self.last_known_good_manifest = dict(manifest_payload)
        else:
            manifest_file = p.parent / ACTIVE_MANIFEST_FILENAME
            if manifest_file.is_file():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        self.last_known_good_manifest = json.load(f)
                except Exception as exc:
                    logger.warning("Could not cache manifest for known-good %s: %s", p.name, exc)

        logger.info(
            "Registered new last-known-good checkpoint: %s (SHA-256: %s, IR: %.2f%%)",
            p.name, sha[:16], metrics.get("mean_ir", 0.0)
        )

    def execute_rollback(
        self,
        failed_ckpt_path: Path | str | None,
        reasons: List[str],
        step: int,
        telemetry: Dict[str, Any] | None = None,
    ) -> Path:
        """Verify known-good SHA, restore original approved manifest, quarantine failed candidate, and log event."""
        # 1. Cryptographic verification of known-good checkpoint before rollback
        if not self.last_known_good_ckpt.is_file():
            raise FileNotFoundError(
                f"CRITICAL INTEGRITY FAILURE: Known-good checkpoint missing at {self.last_known_good_ckpt}!"
            )

        actual_sha = sha256_file(self.last_known_good_ckpt)
        if actual_sha != self.last_known_good_sha256:
            raise RuntimeError(
                f"CRITICAL INTEGRITY FAILURE: Known-good checkpoint {self.last_known_good_ckpt} was tampered with!\n"
                f"  Registered SHA: {self.last_known_good_sha256}\n"
                f"  Current SHA:    {actual_sha}\n"
                f"Halting immediately without silent restoration."
            )

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "step": int(step),
            "reasons": list(reasons),
            "failed_checkpoint": str(failed_ckpt_path) if failed_ckpt_path else None,
            "restored_to": str(self.last_known_good_ckpt),
            "restored_sha256": actual_sha,
            "telemetry": telemetry or {},
        }
        self.history.append(entry)

        # 2. Persist audit log
        with open(self.audit_log_path, "w", encoding="utf-8") as f:
            json.dump({"rollback_events": self.history}, f, indent=2)

        logger.error(
            "ROLLBACK EXECUTED at step %d! Restoring to last known-good: %s (SHA: %s). Reasons: %s",
            step, self.last_known_good_ckpt.name, actual_sha[:16], "; ".join(reasons)
        )

        # 3. If failed checkpoint exists, quarantine it
        if failed_ckpt_path is not None:
            failed_p = Path(failed_ckpt_path)
            if failed_p.exists() and "QUARANTINED" not in failed_p.name:
                quarantine_name = failed_p.with_name(f"{failed_p.stem}_QUARANTINED_COLLAPSE.pt")
                try:
                    failed_p.rename(quarantine_name)
                    logger.info("Quarantined failed checkpoint to %s", quarantine_name.name)
                except Exception as exc:
                    logger.warning("Could not rename failed checkpoint: %s", exc)

        # 4. Restore original active manifest atomically (no fabricated approvals)
        if self.last_known_good_manifest is not None:
            manifest_target = self.candidate_dir / ACTIVE_MANIFEST_FILENAME
            tmp_manifest = self.candidate_dir / f"{ACTIVE_MANIFEST_FILENAME}.tmp_{os.getpid()}"
            try:
                with open(tmp_manifest, "w", encoding="utf-8") as f:
                    json.dump(self.last_known_good_manifest, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                if manifest_target.exists():
                    manifest_target.unlink()
                tmp_manifest.rename(manifest_target)
                logger.info("Restored known-good active manifest at %s", manifest_target)
            except Exception as exc:
                if tmp_manifest.exists():
                    tmp_manifest.unlink()
                logger.warning("Could not restore known-good active manifest: %s", exc)

        return self.last_known_good_ckpt

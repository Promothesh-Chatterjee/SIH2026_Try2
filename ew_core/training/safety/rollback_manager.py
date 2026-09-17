"""Rollback manager for candidate checkpoint recovery and audit logging."""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List
import torch

logger = logging.getLogger(__name__)


class RollbackManager:
    """Retains the last known-good checkpoint and executes automatic rollback upon safety halt."""

    def __init__(
        self,
        last_known_good_ckpt: Path | str,
        audit_log_path: Path | str,
    ) -> None:
        self.last_known_good_ckpt = Path(last_known_good_ckpt).resolve()
        if not self.last_known_good_ckpt.exists():
            raise FileNotFoundError(f"Initial known-good checkpoint not found: {self.last_known_good_ckpt}")

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

    def register_known_good(self, ckpt_path: Path | str, metrics: Dict[str, Any]) -> None:
        """Promote a newly accepted staged checkpoint to the last known-good status."""
        p = Path(ckpt_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint to register does not exist: {p}")
        self.last_known_good_ckpt = p
        logger.info("Registered new last-known-good checkpoint: %s (IR: %.2f%%)", p.name, metrics.get("mean_ir", 0.0))

    def execute_rollback(
        self,
        failed_ckpt_path: Path | str | None,
        reasons: List[str],
        step: int,
        telemetry: Dict[str, Any] | None = None,
    ) -> Path:
        """Log halt audit event and restore to the last known-good checkpoint."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "step": int(step),
            "reasons": list(reasons),
            "failed_checkpoint": str(failed_ckpt_path) if failed_ckpt_path else None,
            "restored_to": str(self.last_known_good_ckpt),
            "telemetry": telemetry or {},
        }
        self.history.append(entry)

        # Persist audit log
        with open(self.audit_log_path, "w", encoding="utf-8") as f:
            json.dump({"rollback_events": self.history}, f, indent=2)

        logger.error(
            "ROLLBACK EXECUTED at step %d! Restoring to last known-good: %s. Reasons: %s",
            step, self.last_known_good_ckpt.name, "; ".join(reasons)
        )

        # If failed checkpoint exists, quarantine it
        if failed_ckpt_path is not None:
            failed_p = Path(failed_ckpt_path)
            if failed_p.exists():
                quarantine_name = failed_p.with_name(f"{failed_p.stem}_QUARANTINED_COLLAPSE.pt")
                try:
                    failed_p.rename(quarantine_name)
                    logger.info("Quarantined failed checkpoint to %s", quarantine_name.name)
                except Exception as exc:
                    logger.warning("Could not rename failed checkpoint: %s", exc)

        return self.last_known_good_ckpt

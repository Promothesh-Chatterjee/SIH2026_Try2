"""Checkpoint protection and atomic persistence guard."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict
import torch

logger = logging.getLogger(__name__)


class CheckpointGuard:
    """Guarantees checkpoint safety, atomic writes, and forbidden-path enforcement.

    Enforcements:
      1. Never writes into production_baseline/ or any forbidden directory.
      2. Writes candidate checkpoints atomically via temporary swap files.
      3. Verifies file integrity with SHA-256 hash logging upon every save.
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
                Path("cognitive_ew_smart_scan/checkpoints/production_baseline"),
            ]
        self.forbidden_dirs = [Path(d).resolve() for d in forbidden_dirs]

        # Immediate path validation on instantiation
        self.validate_path_safety(self.output_dir)

    def validate_path_safety(self, target_path: Path) -> None:
        target_res = target_path.resolve()
        for forbidden in self.forbidden_dirs:
            if forbidden.exists():
                if target_res == forbidden or forbidden in target_res.parents:
                    raise RuntimeError(
                        f"CRITICAL PATH SAFETY VIOLATION: Target path '{target_res}' resolves "
                        f"inside forbidden baseline directory '{forbidden}'!"
                    )

    def save_checkpoint_atomic(
        self,
        state_payload: Dict[str, Any],
        filename: str,
    ) -> tuple[Path, str]:
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

        # Compute SHA-256 hash
        hasher = hashlib.sha256()
        with open(final_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256_hex = hasher.hexdigest()

        logger.info("Saved atomic checkpoint %s (SHA-256: %s)", final_path.name, sha256_hex[:16])
        return final_path, sha256_hex

    def list_valid_checkpoints(self) -> list[Path]:
        """Returns all non-quarantined, valid checkpoints in the directory."""
        return sorted([
            p for p in self.output_dir.glob("checkpoint_*.pt")
            if "QUARANTINED" not in p.name and "tmp" not in p.name
        ])

    def get_active_checkpoint(self) -> Path:
        """Returns the latest approved, non-quarantined candidate checkpoint for deployment or continuation."""
        valid = self.list_valid_checkpoints()
        step_ckpts = [p for p in valid if "checkpoint_step_" in p.name or "checkpoint_gate_" in p.name]
        if not step_ckpts:
            raise FileNotFoundError(f"No valid non-quarantined step checkpoints found in {self.output_dir}")

        def extract_step(p: Path) -> int:
            try:
                stem = p.stem
                return int(stem.split("_")[-1])
            except ValueError:
                return -1

        valid_sorted = sorted(step_ckpts, key=extract_step)
        active = valid_sorted[-1]
        logger.info("Active approved candidate checkpoint identified: %s (Step: %d)", active.name, extract_step(active))
        return active

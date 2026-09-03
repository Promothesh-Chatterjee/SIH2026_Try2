"""Reproducible experiment run manager.

Creates a ``runs/<run_id>/`` directory per training/evaluation run with:

  metadata.json       environment + config snapshot for reproducibility
  telemetry.jsonl     append-only stream of scalar/vector metric events
  checkpoints/        (optional) model snapshots
  normalization.json  (optional) preprocessing statistics
  git_revision.txt    (optional) commit SHA at run start

The purpose is scientific reproducibility: every produced model/result can be
traced back to the exact seed, config, code revision, dataset split, and the
raw telemetry stream that produced it.
"""

from __future__ import annotations

import json
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TextIO


class RunManager:
    """Owns the on-disk artifacts of a single reproducible experiment run.

    Args:
        root: Parent directory that will contain the ``runs/<run_id>`` folder.
        config: Optional full configuration dict (model + training) to snapshot
            in ``metadata.json`` for reproducibility.
        extras: Optional extra metadata (e.g. ``{"split": "train"}``).
    """

    def __init__(
        self,
        root: str | Path = "runs",
        config: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.run_id: str = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.root = Path(root)
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self._meta_written = False

        # Normalize scalar types so metadata serializes cleanly.
        def _jsonable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": time.time(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": self._torch_version(),
            "config": _jsonable(config or {}),
        }
        if extras:
            self.metadata.update(_jsonable(extras))
        self._write_metadata()

        # Sub-directories
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _torch_version() -> str:
        try:
            import torch  # noqa: F401

            return torch.__version__
        except Exception:
            return "unknown"

    def _write_metadata(self) -> None:
        with self.lock:
            (self.dir / "metadata.json").write_text(
                json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            self._meta_written = True

    def write_git_revision(self) -> str:
        """Snapshot the current git commit SHA into the run dir.

        Returns the SHA string (or empty string if git is unavailable).
        """
        sha = ""
        try:
            import subprocess

            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            (self.dir / "git_revision.txt").write_text(sha + "\n", encoding="utf-8")
        except Exception:
            return ""
        return sha

    def write_normalization(self, stats: dict[str, Any]) -> None:
        """Persist preprocessing statistics (train-only) to ``normalization.json``."""
        with self.lock:
            (self.dir / "normalization.json").write_text(
                json.dumps(stats, indent=2, default=str), encoding="utf-8"
            )

    @property
    def telemetry_path(self) -> Path:
        return self.dir / "telemetry.jsonl"

    def open_telemetry(self) -> TextIO:
        """Open (append) the telemetry stream file handle.

        Callers should close the handle when finished. A fresh append-mode open
        is safe because writes are line-buffered JSON documents.
        """
        return self.telemetry_path.open("a", encoding="utf-8")

    def emit(self, timestamp: float | None = None, **fields: Any) -> None:
        """Append one telemetry record (a JSON object line) to the stream.

        Args:
            timestamp: Epoch seconds; defaults to now.
            **fields: Arbitrary named metrics (scalars or JSON-serializable).
        """
        record = {"ts": timestamp if timestamp is not None else time.time()}
        record.update(fields)
        line = json.dumps(record, default=str) + "\n"
        with self.lock:
            with self.telemetry_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def snapshot(self) -> dict[str, Any]:
        """Return the persisted non-telemetry metadata (for REST/api)."""
        return self.metadata

    def summary_paths(self) -> dict[str, Path]:
        """Map of well-known artifacts to their on-disk paths."""
        return {
            "metadata": self.dir / "metadata.json",
            "telemetry": self.telemetry_path,
            "checkpoints": self.dir / "checkpoints",
            "git_revision": self.dir / "git_revision.txt",
        }
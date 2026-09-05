"""Machine-readable provenance manifests for training and evaluation runs."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "numpy", "scipy", "pandas", "h5py", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def build_experiment_manifest(
    *,
    dataset_fingerprint: Any,
    dataset_root: str | Path,
    dataset_mode: str,
    split: str,
    seed: int,
    model_configuration: dict[str, Any],
    training_configuration: dict[str, Any],
    normalization_stats_hash: str | None,
    checkpoint_metadata: dict[str, Any] | None,
    device: str,
    metrics: dict[str, Any],
    git_revision_value: str | None = None,
) -> dict[str, Any]:
    """Build the canonical reproducibility manifest payload."""
    return _jsonable(
        {
            "manifest_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_revision": git_revision_value or git_revision(),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_root": Path(dataset_root),
            "dataset_mode": dataset_mode,
            "split": split,
            "seed": int(seed),
            "model_configuration": model_configuration,
            "training_configuration": training_configuration,
            "normalization_stats_hash": normalization_stats_hash,
            "checkpoint_metadata": checkpoint_metadata or {},
            "device": str(device),
            "software_versions": software_versions(),
            "metrics": metrics,
            "process": {"pid": os.getpid(), "hostname": platform.node()},
        }
    )


def write_experiment_manifest(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write and return a canonical experiment manifest."""
    payload = build_experiment_manifest(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload

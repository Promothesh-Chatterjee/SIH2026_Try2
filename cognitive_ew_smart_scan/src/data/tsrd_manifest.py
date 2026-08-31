"""TSRD validation and manifest utilities.

This module standardizes dataset discovery across official TSRD layouts and
creates a machine-readable manifest for reproducible scientific runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def resolve_split_dirs(data_root: str | Path, mode: str | None = None) -> dict[str, Path]:
    """Resolve train/validation/val/test directories across TSRD layouts.

    Returns a dict with canonical keys: train, val, test.
    """
    root = Path(data_root)
    mode_root = root / str(mode) if mode else root
    candidates = {
        "train": [mode_root / "train", root / "train", root / "stare" / "train", root / "scan" / "train"],
        "val": [mode_root / "val", mode_root / "validation", root / "val", root / "validation"],
        "test": [mode_root / "test", root / "test"],
    }
    resolved: dict[str, Path] = {}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists():
                resolved[key] = p
                break
        if key not in resolved:
            resolved[key] = mode_root / key if mode else root / key
    return resolved


def discover_h5_files(data_root: str | Path, mode: str | None = None) -> list[Path]:
    """Recursively discover .h5 files under the dataset root."""
    root = Path(data_root)
    if mode is not None:
        candidates = [root / mode, root]
    else:
        candidates = [root]
    files: list[Path] = []
    seen: set[Path] = set()
    for base in candidates:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.h5")):
            norm = path.resolve()
            if norm not in seen:
                seen.add(norm)
                files.append(path)
    return files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_root: str | Path, output_path: str | Path | None = None) -> dict:
    """Build a manifest describing the dataset contents and checksums."""
    root = Path(data_root)
    files = discover_h5_files(root)
    records = []
    for path in files:
        rel = path.relative_to(root)
        records.append({
            "path": str(rel).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    manifest = {
        "data_root": str(root),
        "file_count": len(records),
        "files": records,
        "split_map": {mode: str(resolve_split_dirs(root, mode)["train"]) for mode in ["stare", "scan"] if (root / mode).exists()},
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_dataset(data_root: str | Path) -> dict:
    """Validate that the dataset has a usable train/validation/test split."""
    root = Path(data_root)
    manifest = build_manifest(root)
    result = {
        "valid": True,
        "errors": [],
        "summary": {"file_count": manifest["file_count"]},
    }
    for mode in ["stare", "scan"]:
        mode_root = root / mode
        if not mode_root.exists():
            continue
        split_dirs = resolve_split_dirs(root, mode)
        for split_name, split_dir in split_dirs.items():
            if split_name in {"train", "val", "test"} and not split_dir.exists():
                result["errors"].append(f"Missing {mode}/{split_name} directory")
        if split_dirs["train"].exists() and not any(split_dirs["train"].glob("*.h5")):
            result["errors"].append(f"No .h5 files in {split_dirs['train']}")
      
    if result["errors"]:
        result["valid"] = False
    return result

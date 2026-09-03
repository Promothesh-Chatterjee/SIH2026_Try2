"""TSRD validation and manifest utilities.

Standardizes dataset discovery across official TSRD layouts, enforces the
TSRD Data Contract (P0-2), and creates machine-readable manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_FEATURES = ["ToA_us", "CF_MHz", "PW_us", "AoA_deg", "Amplitude_dB"]


def resolve_split_dirs(data_root: str | Path, mode: str | None = None) -> dict[str, Path]:
    """Resolve train/validation/test directories across TSRD layouts.

    Supports:
      1. Official TSRD layout: <mode>/train_<mode>, <mode>/val_<mode>, <mode>/test_<mode>
      2. Conventional layout: <mode>/train, <mode>/val, <mode>/test
      3. Archive layout: archive/train, archive/validation, archive/test
      4. Flat / nested root fallbacks

    Returns a dict with canonical keys: 'train', 'val', 'test'.
    """
    root = Path(data_root)
    m = str(mode) if mode else "scan"
    mode_root = root / m if mode else root

    candidates = {
        "train": [
            mode_root / f"train_{m}",
            mode_root / "train",
            root / f"train_{m}",
            root / "train",
            root / m / f"train_{m}",
            root / m / "train",
            root / "scan" / "train_scan",
            root / "stare" / "train_stare",
            root / "archive" / "train",
        ],
        "val": [
            mode_root / f"val_{m}",
            mode_root / f"validation_{m}",
            mode_root / "val",
            mode_root / "validation",
            root / f"val_{m}",
            root / "val",
            root / "validation",
            root / m / f"val_{m}",
            root / m / "val",
            root / "scan" / "val_scan",
            root / "stare" / "val_stare",
            root / "archive" / "validation",
        ],
        "test": [
            mode_root / f"test_{m}",
            mode_root / "test",
            root / f"test_{m}",
            root / "test",
            root / m / f"test_{m}",
            root / m / "test",
            root / "scan" / "test_scan",
            root / "stare" / "test_stare",
            root / "archive" / "test",
        ],
    }

    resolved: dict[str, Path] = {}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists() and p.is_dir():
                resolved[key] = p
                break
        if key not in resolved:
            # Default to mode_root / key as fallback
            resolved[key] = mode_root / f"{key}_{m}" if (mode_root / f"{key}_{m}").exists() else mode_root / key

    return resolved


class TSRDValidator:
    """Validates HDF5 pulse trains against the canonical TSRD Data Contract (P0-2)."""

    def __init__(
        self,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 40000.0,
        max_duration_s: float = 3600.0,
    ) -> None:
        self.freq_min_mhz = freq_min_mhz
        self.freq_max_mhz = freq_max_mhz
        self.max_duration_s = max_duration_s

    def validate_file(self, file_path: Path | str) -> dict[str, Any]:
        """Validate an individual HDF5 file against the TSRD contract."""
        path = Path(file_path)
        result: dict[str, Any] = {
            "file": str(path),
            "valid": True,
            "errors": [],
            "warnings": [],
            "num_pulses": 0,
            "num_emitters": 0,
            "duration_s": 0.0,
            "features": CANONICAL_FEATURES,
        }

        if not path.exists():
            result["valid"] = False
            result["errors"].append(f"File not found: {path}")
            return result

        if path.suffix != ".h5" and path.suffix != ".hdf5":
            result["valid"] = False
            result["errors"].append(f"Invalid extension: {path.suffix}")
            return result

        try:
            with h5py.File(str(path), "r") as handle:
                # 1. Dataset presence
                if "data" not in handle:
                    result["valid"] = False
                    result["errors"].append("Missing 'data' dataset")
                    return result
                if "labels" not in handle:
                    result["valid"] = False
                    result["errors"].append("Missing 'labels' dataset")
                    return result

                data = np.asarray(handle["data"])
                labels = np.asarray(handle["labels"]).reshape(-1)

                # 2. Shape check: (N, 5)
                if data.ndim != 2 or data.shape[1] != 5:
                    result["valid"] = False
                    result["errors"].append(f"Data shape must be (N, 5), got {data.shape}")
                    return result

                num_pulses = len(data)
                result["num_pulses"] = num_pulses
                if num_pulses == 0:
                    result["valid"] = False
                    result["errors"].append("Dataset contains 0 pulses")
                    return result

                # 3. Label count match
                if len(labels) != num_pulses:
                    result["valid"] = False
                    result["errors"].append(f"Labels length {len(labels)} does not match data length {num_pulses}")

                # 4. Finiteness
                if not np.all(np.isfinite(data)):
                    result["valid"] = False
                    result["errors"].append("Data contains non-finite (NaN/Inf) values")

                # 5. Time ordering (monotonic ToA)
                toas = data[:, 0]
                diffs = np.diff(toas)
                if np.any(diffs < 0):
                    result["valid"] = False
                    result["errors"].append("ToA is not monotonically non-decreasing")

                duration_us = float(np.ptp(toas)) if len(toas) > 0 else 0.0
                result["duration_s"] = duration_us / 1e6
                if result["duration_s"] > self.max_duration_s:
                    result["warnings"].append(f"Duration {result['duration_s']}s exceeds threshold {self.max_duration_s}s")

                # 6. Physical feature ranges
                freqs = data[:, 1]
                if np.any(freqs < self.freq_min_mhz) or np.any(freqs > self.freq_max_mhz):
                    result["warnings"].append(f"Frequencies outside standard range [{self.freq_min_mhz}, {self.freq_max_mhz}] MHz")

                pws = data[:, 2]
                if np.any(pws <= 0):
                    result["warnings"].append("Pulse width contains non-positive values")

                aoas = data[:, 3]
                if np.any(aoas < -180.0) or np.any(aoas > 360.0):
                    result["warnings"].append("AoA values outside [-180, 360] degrees")

                # 7. Emitter labels
                unique_emitters = np.unique(labels)
                result["num_emitters"] = int(len(unique_emitters))

                # Check metadata group if present
                if "metadata" in handle and hasattr(handle["metadata"], "attrs"):
                    meta_attrs = dict(handle["metadata"].attrs)
                    result["metadata_attrs"] = {k: str(v) for k, v in meta_attrs.items()}

        except Exception as exc:
            result["valid"] = False
            result["errors"].append(f"HDF5 reading error: {exc}")

        return result


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


def build_manifest(data_root: str | Path, output_path: str | Path | None = None, mode: str = "scan", max_files: int | None = None) -> dict:
    """Build a comprehensive manifest describing split files, pulse counts, and validation status."""
    root = Path(data_root)
    validator = TSRDValidator()
    split_dirs = resolve_split_dirs(root, mode)

    manifest: dict[str, Any] = {
        "data_root": str(root),
        "mode": mode,
        "splits": {},
        "summary": {"total_files": 0, "total_pulses": 0},
    }

    for split_name, split_dir in split_dirs.items():
        split_files = sorted(split_dir.glob("*.h5")) if split_dir.exists() else []
        if max_files and len(split_files) > max_files:
            split_files = split_files[:max_files]

        records = []
        split_pulses = 0
        for fp in split_files:
            v = validator.validate_file(fp)
            records.append({
                "path": str(fp.relative_to(root)).replace("\\", "/") if root in fp.parents else str(fp),
                "filename": fp.name,
                "size_bytes": fp.stat().st_size,
                "num_pulses": v["num_pulses"],
                "num_emitters": v["num_emitters"],
                "duration_s": round(v["duration_s"], 3),
                "valid": v["valid"],
            })
            split_pulses += v["num_pulses"]

        manifest["splits"][split_name] = {
            "directory": str(split_dir),
            "file_count": len(records),
            "total_pulses": split_pulses,
            "files": records,
        }
        manifest["summary"]["total_files"] += len(records)
        manifest["summary"]["total_pulses"] += split_pulses

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Saved TSRD manifest to %s (%d files, %d pulses)", out, manifest["summary"]["total_files"], manifest["summary"]["total_pulses"])

    return manifest


def validate_dataset(data_root: str | Path) -> dict:
    """Validate dataset split availability without full file traversal."""
    root = Path(data_root)
    result = {
        "valid": True,
        "errors": [],
        "splits": {},
    }

    for mode in ["scan", "stare"]:
        split_dirs = resolve_split_dirs(root, mode)
        mode_valid = True
        for split_name, split_dir in split_dirs.items():
            exists = split_dir.exists()
            h5_count = len(list(split_dir.glob("*.h5"))) if exists else 0
            result["splits"][f"{mode}/{split_name}"] = {
                "dir": str(split_dir),
                "exists": exists,
                "h5_count": h5_count,
            }
            if not exists or h5_count == 0:
                result["errors"].append(f"Missing or empty {mode}/{split_name} at {split_dir}")
                mode_valid = False

    if result["errors"]:
        # Check if at least one mode is valid
        scan_train = result["splits"].get("scan/train", {}).get("h5_count", 0)
        if scan_train > 0:
            result["valid"] = True
        else:
            result["valid"] = False
    return result


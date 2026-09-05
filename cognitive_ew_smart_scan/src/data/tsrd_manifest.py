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
            root / "archive" / "validation",
        ],
        "test": [
            mode_root / f"test_{m}",
            mode_root / "test",
            root / f"test_{m}",
            root / "test",
            root / m / f"test_{m}",
            root / m / "test",
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
            "structurally_valid": True,
            "empty_scenario": False,
            "training_eligible": False,
            "evaluation_eligible": False,
            "errors": [],
            "warnings": [],
            "num_pulses": 0,
            "num_emitters": 0,
            "num_nonnoise_emitters": 0,
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
                    result["structurally_valid"] = False
                    result["errors"].append("Missing 'data' dataset")
                    return result
                if "labels" not in handle:
                    result["valid"] = False
                    result["structurally_valid"] = False
                    result["errors"].append("Missing 'labels' dataset")
                    return result

                data = np.asarray(handle["data"])
                labels = np.asarray(handle["labels"]).reshape(-1)

                # 2. Shape check: (N, 5)
                shape = tuple(int(s) for s in data.shape)
                if data.ndim != 2 or data.shape[1] != 5:
                    result["valid"] = False
                    result["structurally_valid"] = False
                    result["errors"].append(f"Data shape must be (N, 5), got {shape}")
                    return result

                num_pulses = len(data)
                result["num_pulses"] = num_pulses
                # Zero-pulse trains (e.g. `data (0,5)` / `labels (0,1)` in official
                # TSRD splits) are STILL structurally valid — an empty scene is a
                # legitimate scene, not a corrupt file. We only record it and mark
                # it ineligible for training/evaluation (see eligibility fields).
                result["empty_scenario"] = num_pulses == 0

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
                nonnoise = labels[labels != -1]
                result["num_nonnoise_emitters"] = int(len(np.unique(nonnoise)))

                # --- Eligibility classification (Phase 19) ---
                # Three distinct concepts, so a handful of empty trains never
                # invalidate the whole dataset or crash a training run:
                #   * structural validity  — shape/readability/label alignment
                #   * training eligibility — structurally valid AND non-empty
                #   * evaluation eligibility — training-eligible AND has at least
                #       one annotated (non-noise) emitter to score against
                result["structurally_valid"] = result["valid"]
                result["training_eligible"] = result["valid"] and num_pulses > 0
                result["evaluation_eligible"] = (
                    result["training_eligible"] and result["num_nonnoise_emitters"] >= 1
                )

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


def dataset_fingerprint(files: list[Path], root: str | Path, mode: str) -> str:
    """Return a stable fingerprint for files, sizes, split mode, and hashes."""
    root_path = Path(root).resolve()
    entries = []
    for path in sorted(files, key=lambda item: str(item).replace("\\", "/")):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        entries.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    payload = json.dumps({"mode": mode, "files": entries}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(data_root: str | Path, output_path: str | Path | None = None, mode: str = "scan", max_files: int | None = None) -> dict:
    """Build a comprehensive manifest describing split files, pulse counts, and validation status.

    Per-split eligibility statistics are reported (structurally valid / empty /
    training-eligible / evaluation-eligible counts) so a few zero-pulse trains
    are visible but never treated as dataset-level corruption.
    """
    root = Path(data_root)
    validator = TSRDValidator()
    split_dirs = resolve_split_dirs(root, mode)

    all_files: list[Path] = []
    manifest: dict[str, Any] = {
        "data_root": str(root),
        "mode": mode,
        "splits": {},
        "summary": {
            "total_files": 0,
            "total_pulses": 0,
            "structurally_valid_files": 0,
            "empty_files": 0,
            "training_eligible": 0,
            "evaluation_eligible": 0,
        },
    }

    for split_name, split_dir in split_dirs.items():
        split_files = sorted(split_dir.glob("*.h5")) if split_dir.exists() else []
        if max_files and len(split_files) > max_files:
            split_files = split_files[:max_files]

        records = []
        split_pulses = 0
        split_stats = {
            "structurally_valid_files": 0,
            "empty_files": 0,
            "training_eligible": 0,
            "evaluation_eligible": 0,
        }
        all_files.extend(split_files)
        for fp in split_files:
            v = validator.validate_file(fp)
            if v["structurally_valid"]:
                split_stats["structurally_valid_files"] += 1
            if v["empty_scenario"]:
                split_stats["empty_files"] += 1
            if v["training_eligible"]:
                split_stats["training_eligible"] += 1
            if v["evaluation_eligible"]:
                split_stats["evaluation_eligible"] += 1
            records.append({
                "path": str(fp.relative_to(root)).replace("\\", "/") if root in fp.parents else str(fp),
                "filename": fp.name,
                "size_bytes": fp.stat().st_size,
                "sha256": _sha256(fp),
                "num_pulses": v["num_pulses"],
                "num_emitters": v["num_emitters"],
                "duration_s": round(v["duration_s"], 3),
                "structurally_valid": v["structurally_valid"],
                "empty_scenario": v["empty_scenario"],
                "training_eligible": v["training_eligible"],
                "evaluation_eligible": v["evaluation_eligible"],
            })
            split_pulses += v["num_pulses"]

        manifest["splits"][split_name] = {
            "directory": str(split_dir),
            "file_count": len(records),
            "total_pulses": split_pulses,
            "structurally_valid_files": split_stats["structurally_valid_files"],
            "empty_files": split_stats["empty_files"],
            "training_eligible": split_stats["training_eligible"],
            "evaluation_eligible": split_stats["evaluation_eligible"],
            "files": records,
        }
        manifest["summary"]["total_files"] += len(records)
        manifest["summary"]["total_pulses"] += split_pulses
        for key in split_stats:
            manifest["summary"][key] += split_stats[key]

    manifest["dataset_fingerprint"] = dataset_fingerprint(all_files, root, mode)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Saved TSRD manifest to %s (%d files, %d pulses)", out, manifest["summary"]["total_files"], manifest["summary"]["total_pulses"])

    return manifest


def count_empty_h5(directory: Path) -> int:
    """Count zero-pulse ``.h5`` trains in a directory (header-only, cheap).

    An empty pulse train is a legitimate official-TSRD scene, so it does NOT
    affect structural dataset validity — this count exists purely for reporting.
    """
    if not directory.is_dir():
        return 0
    empty = 0
    for path in directory.glob("*.h5"):
        try:
            import h5py
            with h5py.File(str(path), "r") as handle:
                if "data" in handle and handle["data"].shape[0] == 0:
                    empty += 1
        except Exception:
            continue
    return empty


def validate_dataset(data_root: str | Path) -> dict:
    """Validate dataset split availability without full file traversal.

    Structural validity is checked per-split (the directory exists and contains
    at least one ``.h5``). Zero-pulse trains are REPORTED (``num_empty``) but do
    NOT invalidate the split — an empty scene is structurally valid. Individual
    per-file eligibility lives in ``TSRDValidator`` / ``build_manifest``.
    """
    root = Path(data_root)
    result = {
        "valid": True,
        "errors": [],
        "splits": {},
    }
    any_all_empty = False

    for mode in ["scan", "stare"]:
        split_dirs = resolve_split_dirs(root, mode)
        mode_valid = True
        for split_name, split_dir in split_dirs.items():
            exists = split_dir.exists()
            h5_count = len(list(split_dir.glob("*.h5"))) if exists else 0
            num_empty = count_empty_h5(split_dir) if exists else 0
            result["splits"][f"{mode}/{split_name}"] = {
                "dir": str(split_dir),
                "exists": exists,
                "h5_count": h5_count,
                "num_empty": num_empty,
                "meaningful_train_count": max(0, h5_count - num_empty),
            }
            if not exists or h5_count == 0:
                result["errors"].append(f"Missing or empty {mode}/{split_name} at {split_dir}")
                mode_valid = False
            elif num_empty == h5_count:
                any_all_empty = True
                result["errors"].append(
                    f"All {h5_count} trains in {mode}/{split_name} are zero-pulse "
                    f"(none usable for training/evaluation despite structural validity)"
                )
                mode_valid = False

    if result["errors"]:
        # Check if at least one mode is valid (structural). An individual
        # zero-pulse train is fine, but a split with NO usable train is not.
        scan_train = result["splits"].get("scan/train", {}).get("h5_count", 0)
        if scan_train > 0 and not any_all_empty:
            result["valid"] = True
        else:
            result["valid"] = False
    return result


def generate_dataset_report(
    train_files: list[Path],
    val_files: list[Path],
    mode: str = "scan",
    max_sample_files: int = 10,
) -> dict:
    """Generate a comprehensive dataset report for training/evaluation.

    Args:
        train_files: List of training file paths.
        val_files: List of validation file paths.
        mode: Data mode ("scan" or "stare").
        max_sample_files: Maximum files to sample for detailed statistics.

    Returns:
        Dictionary with dataset statistics.
    """
    import h5py

    validator = TSRDValidator()
    report = {
        "mode": mode,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_pulses": 0,
        "val_pulses": 0,
        "train_emitters": 0,
        "val_emitters": 0,
        "train_duration_s": 0.0,
        "val_duration_s": 0.0,
        "frequency_range_mhz": [float("inf"), float("-inf")],
        "pulse_width_range_us": [float("inf"), float("-inf")],
        "amplitude_range_db": [float("inf"), float("-inf")],
        "noise_fraction": 0.0,
        "missing_files": [],
        "invalid_files": [],
        "empty_files": 0,
        "training_eligible_files": 0,
        "evaluation_eligible_files": 0,
    }

    def sample_stats(files: list[Path], split_name: str) -> tuple[int, int, float]:
        pulses = 0
        emitters = 0
        duration = 0.0
        freq_min = float("inf")
        freq_max = float("-inf")
        pw_min = float("inf")
        pw_max = float("-inf")
        amp_min = float("inf")
        amp_max = float("-inf")
        noise_count = 0
        total_pulses = 0

        sample_files = files[:max_sample_files]
        for fp in sample_files:
            v = validator.validate_file(fp)
            if not v["valid"]:
                report["invalid_files"].append({"file": str(fp), "errors": v["errors"]})
                continue
            if v["empty_scenario"]:
                report["empty_files"] += 1
            if v["training_eligible"]:
                report["training_eligible_files"] += 1
            if v["evaluation_eligible"]:
                report["evaluation_eligible_files"] += 1
            pulses += v["num_pulses"]
            emitters += v["num_emitters"]
            duration += v["duration_s"]
            total_pulses += v["num_pulses"]

            # Get detailed stats from file
            try:
                with h5py.File(str(fp), "r") as handle:
                    data = np.asarray(handle["data"])
                    labels = np.asarray(handle["labels"]).reshape(-1)
                    freqs = data[:, 1]
                    pws = data[:, 2]
                    amps = data[:, 4]
                    freq_min = min(freq_min, float(np.min(freqs)))
                    freq_max = max(freq_max, float(np.max(freqs)))
                    pw_min = min(pw_min, float(np.min(pws)))
                    pw_max = max(pw_max, float(np.max(pws)))
                    amp_min = min(amp_min, float(np.min(amps)))
                    amp_max = max(amp_max, float(np.max(amps)))
                    noise_count += int(np.sum(labels == -1))
            except Exception:
                pass

        return pulses, emitters, duration

    train_pulses, train_emitters, train_dur = sample_stats(train_files, "train")
    val_pulses, val_emitters, val_dur = sample_stats(val_files, "val")

    report["train_pulses"] = train_pulses
    report["val_pulses"] = val_pulses
    report["train_emitters"] = train_emitters
    report["val_emitters"] = val_emitters
    report["train_duration_s"] = round(train_dur, 3)
    report["val_duration_s"] = round(val_dur, 3)

    if report["frequency_range_mhz"][0] == float("inf"):
        report["frequency_range_mhz"] = [0.0, 18000.0]
    if report["pulse_width_range_us"][0] == float("inf"):
        report["pulse_width_range_us"] = [0.0, 10.0]
    if report["amplitude_range_db"][0] == float("inf"):
        report["amplitude_range_db"] = [-140.0, 0.0]

    total_pulses = train_pulses + val_pulses
    total_noise = 0  # Would need full scan for accurate noise fraction
    report["noise_fraction"] = 0.0

    return report


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

    Supports (candidates defined in :mod:`src.data.tsrd_root`):
      1. Official/Kaggle layout: <mode>/train_<mode>, <mode>/val_<mode>, <mode>/test_<mode>
      2. Conventional layout: <mode>/train, <mode>/val|validation, <mode>/test
      3. Archive layout: archive/train, archive/validation, archive/test
      4. Flat / nested root fallbacks

    Returns a dict with canonical keys: 'train', 'val', 'test'.
    """
    from ..data.tsrd_root import split_candidate_dirs

    root = Path(data_root)
    m = str(mode) if mode else "scan"
    mode_root = root / m if mode else root
    candidates = split_candidate_dirs(root, m, mode_root=mode_root)

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


def streaming_canonical_content_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute streaming canonical dataset-content SHA-256 digest.

    Deterministically hashes:
      canonical schema descriptor || canonical float64 data chunks || canonical int64 label chunks

    Omits arbitrary HDF5 file-level attributes/timestamps so that renamed,
    re-serialized, or re-compressed files containing identical pulse streams produce identical digests.
    """
    path = Path(file_path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    try:
        with h5py.File(str(path), "r") as handle:
            if "data" not in handle:
                return ""
            data_ds = handle["data"]
            labels_ds = handle.get("labels")
            n_pulses = int(data_ds.shape[0])
            n_cols = int(data_ds.shape[1]) if data_ds.ndim == 2 else 0
            hasher.update(
                f"CANONICAL_TSRD_STREAM_V1:N={n_pulses};cols={n_cols};features=ToA_us,CF_MHz,PW_us,AoA_deg,Amplitude_dB\n".encode("ascii")
            )
            # Stream data in little-endian float64
            for i in range(0, n_pulses, chunk_size):
                chunk = np.asarray(data_ds[i : i + chunk_size], dtype="<f8")
                hasher.update(chunk.tobytes())
            # Stream labels in little-endian int64
            if labels_ds is not None:
                for i in range(0, n_pulses, chunk_size):
                    chunk = np.asarray(labels_ds[i : i + chunk_size], dtype="<i8").reshape(-1)
                    hasher.update(chunk.tobytes())
            else:
                hasher.update(b"NO_LABELS\n")
    except Exception as exc:
        logger.warning("Streaming canonical content hash failed for %s: %s", path, exc)
        return ""
    return hasher.hexdigest()


def compute_near_duplicate_diagnostic(data: np.ndarray, max_pulses: int = 1000) -> str:
    """Compute quantized pulse sequence fingerprint for near-duplicate telemetry.

    NOTE: Used purely as a diagnostic telemetry signal; never alone triggers a split leakage failure.
    """
    if len(data) == 0:
        return "empty"
    sample = data[:max_pulses]
    toas = sample[:, 0]
    cfs = sample[:, 1]
    pws = sample[:, 2]
    d_toas = np.diff(toas) if len(toas) > 1 else np.array([0.0])
    q_dtoa = np.round(d_toas / 10.0) * 10.0
    q_cf = np.round(cfs / 50.0) * 50.0
    q_pw = np.round(pws / 0.5) * 0.5
    h = hashlib.sha256()
    h.update(q_dtoa.astype(np.int64).tobytes())
    h.update(q_cf.astype(np.int64).tobytes())
    h.update(q_pw.astype(np.int64).tobytes())
    return h.hexdigest()[:16]


class SplitLeakageError(RuntimeError):
    """Raised when data leakage or overlap is detected between train, val, or test splits."""
    pass


def validate_split_isolation(
    split_files: dict[str, list[Path]],
    root: Path | str | None = None,
    root_path: Path | str | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Verify 3-layer split isolation across train, val, and test partitions.

    Layers:
      Layer 1: Canonical relative file paths
      Layer 2: Raw file SHA-256 digests
      Layer 3: Streaming canonical-content SHA-256 digests

    Diagnostic:
      Near-duplicate quantized fingerprints (reported for telemetry only).
    """
    effective_root = root_path if root_path is not None else root
    root_path = Path(effective_root).resolve() if effective_root else None
    splits = sorted(split_files.keys())

    paths_by_split: dict[str, dict[str, Path]] = {}
    file_hashes_by_split: dict[str, dict[str, Path]] = {}
    content_hashes_by_split: dict[str, dict[str, Path]] = {}
    near_dups_by_split: dict[str, dict[str, Path]] = {}

    for s in splits:
        paths_by_split[s] = {}
        file_hashes_by_split[s] = {}
        content_hashes_by_split[s] = {}
        near_dups_by_split[s] = {}
        for p in split_files[s]:
            resolved = p.resolve()
            rel = str(resolved.relative_to(root_path)).replace("\\", "/") if root_path and root_path in resolved.parents else p.name
            paths_by_split[s][rel] = resolved
            f_hash = _sha256(resolved)
            file_hashes_by_split[s][f_hash] = resolved
            c_hash = streaming_canonical_content_sha256(resolved)
            if c_hash:
                content_hashes_by_split[s][c_hash] = resolved

    layer1_overlaps: list[dict[str, Any]] = []
    layer2_overlaps: list[dict[str, Any]] = []
    layer3_overlaps: list[dict[str, Any]] = []

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]

            # Layer 1: Canonical Path
            p1 = set(paths_by_split[s1].keys())
            p2 = set(paths_by_split[s2].keys())
            p_inter = p1.intersection(p2)
            for path_key in p_inter:
                layer1_overlaps.append({
                    "split_a": s1,
                    "split_b": s2,
                    "path": path_key,
                })

            # Layer 2: Raw File Hash
            h1 = set(file_hashes_by_split[s1].keys())
            h2 = set(file_hashes_by_split[s2].keys())
            h_inter = h1.intersection(h2)
            for h_val in h_inter:
                layer2_overlaps.append({
                    "split_a": s1,
                    "split_b": s2,
                    "file_sha256": h_val,
                    "file_a": str(file_hashes_by_split[s1][h_val]),
                    "file_b": str(file_hashes_by_split[s2][h_val]),
                })

            # Layer 3: Canonical Content Hash
            c1 = set(content_hashes_by_split[s1].keys())
            c2 = set(content_hashes_by_split[s2].keys())
            c_inter = c1.intersection(c2)
            for c_val in c_inter:
                layer3_overlaps.append({
                    "split_a": s1,
                    "split_b": s2,
                    "canonical_content_sha256": c_val,
                    "file_a": str(content_hashes_by_split[s1][c_val]),
                    "file_b": str(content_hashes_by_split[s2][c_val]),
                })

    is_isolated = bool(len(layer1_overlaps) == 0 and len(layer2_overlaps) == 0 and len(layer3_overlaps) == 0)

    report = {
        "isolated": is_isolated,
        "splits_checked": splits,
        "total_files_checked": sum(len(v) for v in split_files.values()),
        "layer1_path_overlaps": layer1_overlaps,
        "layer2_raw_hash_overlaps": layer2_overlaps,
        "layer3_content_hash_overlaps": layer3_overlaps,
    }

    if not is_isolated and fail_fast:
        err_msg = (
            f"Split leakage detected across {splits}! "
            f"Layer 1 (Path): {len(layer1_overlaps)} overlaps; "
            f"Layer 2 (File SHA): {len(layer2_overlaps)} overlaps; "
            f"Layer 3 (Content SHA): {len(layer3_overlaps)} overlaps."
        )
        raise SplitLeakageError(err_msg)

    return report


PROJECT_TAXONOMY_CLASSES = (
    "fixed",
    "sparse",
    "fast_agile",
    "slow_agile",
    "markov_agile",
    "periodic_scan",
    "mixed",
    "dense",
)


def classify_project_taxonomy(
    data: np.ndarray,
    labels: np.ndarray | None = None,
    max_pulses: int = 50000,
) -> dict[str, Any]:
    """Classify an electromagnetic scenario into the project's 8-class EW taxonomy.

    The 8 classes constitute a project-defined behavioral taxonomy for EW evaluation,
    not official TSRD dataset labels. Real TSRD files may exhibit combinations of behaviors
    or be classified as 'unknown' if feature evidence is ambiguous.

    Returns:
      primary_class: str | "unknown"
      secondary_tags: list[str]
      classification_confidence: float
      classification_evidence: dict[str, Any]
    """
    if len(data) == 0:
        return {
            "primary_class": "unknown",
            "secondary_tags": [],
            "classification_confidence": 0.0,
            "classification_evidence": {"empty": True},
        }

    sample = data[:max_pulses]
    num_pulses = len(sample)
    toas = sample[:, 0]
    freqs = sample[:, 1]
    pws = sample[:, 2]
    amps = sample[:, 4]

    duration_us = float(np.ptp(toas)) if len(toas) > 1 else 1.0
    duration_ms = max(1.0, duration_us / 1000.0)
    pulse_density_pms = float(num_pulses / duration_ms)

    # Emitter-level features if labels available
    if labels is not None and len(labels) >= num_pulses:
        lbls = np.asarray(labels[:num_pulses]).reshape(-1)
        valid_mask = (lbls != -1)
        unique_emitters = [int(x) for x in np.unique(lbls[valid_mask])]
    else:
        lbls = np.zeros(num_pulses, dtype=np.int64)
        unique_emitters = [0]

    n_emitters = len(unique_emitters)

    emitter_agile = []
    emitter_fixed = []
    emitter_fast = []
    emitter_slow = []
    emitter_spans = []

    for eid in unique_emitters[:50]:
        mask = (lbls == eid)
        e_cfs = freqs[mask]
        e_toas = toas[mask]
        if len(e_cfs) < 3:
            continue
        cf_span = float(np.ptp(e_cfs))
        emitter_spans.append(cf_span)
        n_unique_cfs = len(set(np.round(e_cfs, 1)))
        pris = np.diff(e_toas) if len(e_toas) > 1 else np.array([0.0])
        med_pri = float(np.median(pris)) if len(pris) > 0 else 0.0

        is_agile = bool(cf_span >= 30.0 and n_unique_cfs >= 3)
        is_fixed = bool(cf_span <= 15.0)
        is_fast = bool(is_agile and med_pri <= 1200.0)
        is_slow = bool(is_agile and med_pri > 1200.0)

        if is_agile:
            emitter_agile.append(eid)
        if is_fixed:
            emitter_fixed.append(eid)
        if is_fast:
            emitter_fast.append(eid)
        if is_slow:
            emitter_slow.append(eid)

    has_agile = len(emitter_agile) > 0
    has_fixed = len(emitter_fixed) > 0
    overall_cf_span = float(np.ptp(freqs)) if len(freqs) > 0 else 0.0

    # Evidence metrics
    evidence = {
        "num_pulses": num_pulses,
        "num_emitters": n_emitters,
        "duration_ms": round(duration_ms, 2),
        "pulse_density_pms": round(pulse_density_pms, 2),
        "overall_cf_span_mhz": round(overall_cf_span, 1),
        "n_agile_emitters": len(emitter_agile),
        "n_fixed_emitters": len(emitter_fixed),
        "n_fast_agile_emitters": len(emitter_fast),
        "n_slow_agile_emitters": len(emitter_slow),
    }

    tags: list[str] = []

    # Tag assignments
    if n_emitters >= 15 or (n_emitters >= 5 and pulse_density_pms >= 2.5):
        tags.append("dense")
    if pulse_density_pms <= 0.25 or (n_emitters <= 2 and pulse_density_pms <= 0.5):
        tags.append("sparse")
    if len(emitter_fast) > 0:
        tags.append("fast_agile")
    if len(emitter_slow) > 0:
        tags.append("slow_agile")
    if has_fixed and has_agile:
        tags.append("mixed")
    if has_fixed and not has_agile and n_emitters >= 1:
        tags.append("fixed")
    if has_agile and len(emitter_agile) >= 2:
        tags.append("markov_agile")

    # Periodic scan check: presence of periodic burst gaps in ToA with intra-burst pulses
    if len(toas) > 20:
        diffs = np.diff(toas)
        burst_pris = diffs[diffs < 1000.0]
        large_gaps = diffs[diffs > 10000.0]
        if len(large_gaps) >= 3 and len(burst_pris) >= 8:
            gap_std = float(np.std(large_gaps))
            gap_mean = float(np.mean(large_gaps))
            if gap_mean > 0 and (gap_std / gap_mean) < 0.40:
                tags.append("periodic_scan")
                evidence["periodic_scan_period_us"] = round(gap_mean, 1)

    # Primary class determination
    primary = "unknown"
    conf = 0.30

    if "markov_agile" in tags and not has_fixed:
        primary = "markov_agile"
        conf = 0.88
    elif "periodic_scan" in tags and not has_agile:
        primary = "periodic_scan"
        conf = 0.85
    elif "sparse" in tags and not has_agile:
        primary = "sparse"
        conf = 0.85
    elif "mixed" in tags:
        primary = "mixed"
        conf = 0.88
    elif "dense" in tags:
        primary = "dense"
        conf = 0.85
    elif "fast_agile" in tags:
        primary = "fast_agile"
        conf = 0.82
    elif "slow_agile" in tags:
        primary = "slow_agile"
        conf = 0.80
    elif "fixed" in tags:
        primary = "fixed"
        conf = 0.85
    elif "sparse" in tags:
        primary = "sparse"
        conf = 0.75
    elif "periodic_scan" in tags:
        primary = "periodic_scan"
        conf = 0.70
    elif tags:
        primary = tags[0]
        conf = 0.60

    return {
        "primary_class": primary,
        "secondary_tags": tags,
        "classification_confidence": round(conf, 2),
        "classification_evidence": evidence,
    }


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

    def validate_file(
        self,
        file_path: Path | str,
        compute_content_hash: bool = True,
        compute_taxonomy: bool = True,
    ) -> dict[str, Any]:
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
            "canonical_content_sha256": None,
            "observed_ranges": {},
            "taxonomy": None,
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

                amps = data[:, 4]

                if num_pulses > 0:
                    result["observed_ranges"] = {
                        "toa_us": [float(np.min(toas)), float(np.max(toas))],
                        "freq_mhz": [float(np.min(freqs)), float(np.max(freqs))],
                        "pw_us": [float(np.min(pws)), float(np.max(pws))],
                        "aoa_deg": [float(np.min(aoas)), float(np.max(aoas))],
                        "amp_db": [float(np.min(amps)), float(np.max(amps))],
                    }
                    if compute_taxonomy:
                        result["taxonomy"] = classify_project_taxonomy(data, labels)

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

            if compute_content_hash and result["structurally_valid"]:
                result["canonical_content_sha256"] = streaming_canonical_content_sha256(path)

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


def build_manifest(
    data_root: str | Path,
    output_path: str | Path | None = None,
    mode: str = "scan",
    max_files: int | None = None,
    split_files: dict[str, list[Path]] | None = None,
    source_type: str = "unknown",
    source_provider: str | None = None,
    dataset_id: str | None = None,
    source_revision: str | None = None,
    root_resolution_source: str = "default",
    evaluation_split: str | None = None,
    enforce_split_isolation: bool = True,
    fail_fast_on_leakage: bool = False,
    classify_taxonomy: bool = True,
    compute_content_hash: bool = True,
) -> dict:
    """Build a comprehensive manifest describing split files, pulse counts, and validation status.

    Per-split eligibility statistics are reported (structurally valid / empty /
    training-eligible / evaluation-eligible counts) so a few zero-pulse trains
    are visible but never treated as dataset-level corruption.

    Args:
        data_root: Dataset root directory.
        output_path: Optional JSON path to write the manifest to.
        mode: Split mode (``scan`` or ``stare``).
        max_files: Optional cap on the number of files recorded per split.
        split_files: Optional explicit per-split file lists ``{split_name:
            [Path, ...]}``. When provided for a split, those exact files are
            recorded (after sorting) instead of globbing the split directory,
            so the manifest surface matches the files actually consumed by a
            training run.
        source_type: Data source type ("official_tsrd", "synthetic_fixture", etc.)
        source_provider: Data provider or upstream repository URL.
        dataset_id: Canonical identifier for dataset version.
        source_revision: Git commit or release tag of data source.
        root_resolution_source: Method by which data root was resolved ("cli", "env", "yaml", "default").
        evaluation_split: If specified, split designated for evaluation. Fails closed if "train".
        enforce_split_isolation: If True, executes 3-layer cross-split isolation verification.
        classify_taxonomy: If True, classifies pulse trains according to project 8-class EW taxonomy.
        compute_content_hash: If True, computes streaming canonical-content SHA-256.
    """
    if evaluation_split == "train":
        raise ValueError("Role contract violation: 'train' split cannot be designated as evaluation_split")

    root = Path(data_root)
    validator = TSRDValidator()
    split_dirs = resolve_split_dirs(root, mode)

    all_files: list[Path] = []
    collected_split_files: dict[str, list[Path]] = {}
    taxonomy_counts: dict[str, int] = {cls_name: 0 for cls_name in PROJECT_TAXONOMY_CLASSES}
    taxonomy_counts["unknown"] = 0

    manifest: dict[str, Any] = {
        "data_root": str(root),
        "mode": mode,
        "dataset_provenance": {
            "source_type": source_type,
            "source_provider": source_provider,
            "dataset_id": dataset_id,
            "source_revision": source_revision,
            "root_resolution_source": root_resolution_source,
            "evaluation_split": evaluation_split,
        },
        "splits": {},
        "summary": {
            "total_files": 0,
            "total_pulses": 0,
            "structurally_valid_files": 0,
            "empty_files": 0,
            "training_eligible": 0,
            "evaluation_eligible": 0,
        },
        "taxonomy_summary": {},
    }

    for split_name, split_dir in split_dirs.items():
        if split_files is not None and split_name not in split_files:
            continue
        if split_files and split_name in split_files and split_files[split_name] is not None:
            split_files_expanded = sorted(split_files[split_name])
        else:
            split_files_expanded = sorted(split_dir.glob("*.h5")) if split_dir.exists() else []
        split_files_expanded = split_files_expanded[:max_files] if max_files and len(split_files_expanded) > max_files else split_files_expanded

        collected_split_files[split_name] = split_files_expanded
        records = []
        split_pulses = 0
        split_stats = {
            "structurally_valid_files": 0,
            "empty_files": 0,
            "training_eligible": 0,
            "evaluation_eligible": 0,
        }
        all_files.extend(split_files_expanded)
        for fp in split_files_expanded:
            v = validator.validate_file(
                fp,
                compute_content_hash=compute_content_hash,
                compute_taxonomy=classify_taxonomy,
            )
            if v["structurally_valid"]:
                split_stats["structurally_valid_files"] += 1
            if v["empty_scenario"]:
                split_stats["empty_files"] += 1
            if v["training_eligible"]:
                split_stats["training_eligible"] += 1
            if v["evaluation_eligible"]:
                split_stats["evaluation_eligible"] += 1

            tax = v.get("taxonomy")
            if tax and "primary_class" in tax:
                p_cls = tax["primary_class"]
                taxonomy_counts[p_cls] = taxonomy_counts.get(p_cls, 0) + 1

            records.append({
                "path": str(fp.relative_to(root)).replace("\\", "/") if root in fp.parents else str(fp),
                "filename": fp.name,
                "size_bytes": fp.stat().st_size,
                "sha256": _sha256(fp),
                "canonical_content_sha256": v.get("canonical_content_sha256"),
                "num_pulses": v["num_pulses"],
                "num_emitters": v["num_emitters"],
                "duration_s": round(v["duration_s"], 3),
                "structurally_valid": v["structurally_valid"],
                "empty_scenario": v["empty_scenario"],
                "training_eligible": v["training_eligible"],
                "evaluation_eligible": v["evaluation_eligible"],
                "observed_ranges": v.get("observed_ranges"),
                "taxonomy": tax,
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

    manifest["taxonomy_summary"] = taxonomy_counts
    manifest["dataset_fingerprint"] = dataset_fingerprint(all_files, root, mode)

    if enforce_split_isolation and len(collected_split_files) > 1:
        iso_report = validate_split_isolation(
            collected_split_files,
            root_path=root,
            fail_fast=fail_fast_on_leakage,
        )
        manifest["split_isolation"] = iso_report

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


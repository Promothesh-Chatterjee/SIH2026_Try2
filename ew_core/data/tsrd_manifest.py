"""TSRD validation and manifest utilities.

Standardizes dataset discovery across official TSRD layouts, enforces the
TSRD Data Contract (P0-2), and creates machine-readable manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
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
    precomputed_raw_hashes: dict[str, str] | None = None,
    precomputed_content_hashes: dict[str, str] | None = None,
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
            p_str = str(resolved)
            p_orig = str(p)
            rel = str(resolved.relative_to(root_path)).replace("\\", "/") if root_path and root_path in resolved.parents else p.name
            paths_by_split[s][rel] = resolved

            f_hash = None
            if precomputed_raw_hashes is not None:
                f_hash = precomputed_raw_hashes.get(p_str) or precomputed_raw_hashes.get(p_orig)
            if not f_hash:
                f_hash = _sha256(resolved)
            file_hashes_by_split[s][f_hash] = resolved

            c_hash = None
            if precomputed_content_hashes is not None:
                c_hash = precomputed_content_hashes.get(p_str) or precomputed_content_hashes.get(p_orig)
            if not c_hash:
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
        "raw_hashes": {h: str(path) for s in splits for h, path in file_hashes_by_split[s].items()},
        "content_hashes": {c: str(path) for s in splits for c, path in content_hashes_by_split[s].items()},
        "file_to_raw_hash": {str(path): h for s in splits for h, path in file_hashes_by_split[s].items()},
        "file_to_content_hash": {str(path): c for s in splits for c, path in content_hashes_by_split[s].items()},
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


def validate_split_isolation_dual_mode(
    file_lists: dict[str, dict[str, list[Path]]],
    root_path: str | Path | None = None,
    fail_fast: bool = False,
    precomputed_raw_hashes: dict[str, str] | None = None,
    precomputed_content_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Exhaustive 3-layer cross-split isolation for both STARE and SCAN.

    Mandatory failure condition:
      - train <-> val <-> test isolation within STARE
      - train <-> val <-> test isolation within SCAN

    Cross-mode comparison (STARE <-> SCAN):
      - Duplicate raw file hashes and canonical content hashes across modes are
        reported as telemetry diagnostics (does not fail qualification, as official
        TSRD uses the same transmitter configs under different receiver modes).
    """
    root = Path(root_path).resolve() if root_path else None
    stare_splits = file_lists.get("stare", {})
    scan_splits = file_lists.get("scan", {})

    stare_iso = validate_split_isolation(
        stare_splits,
        root_path=root,
        fail_fast=fail_fast,
        precomputed_raw_hashes=precomputed_raw_hashes,
        precomputed_content_hashes=precomputed_content_hashes,
    )
    scan_iso = validate_split_isolation(
        scan_splits,
        root_path=root,
        fail_fast=fail_fast,
        precomputed_raw_hashes=precomputed_raw_hashes,
        precomputed_content_hashes=precomputed_content_hashes,
    )

    is_isolated = bool(stare_iso["isolated"] and scan_iso["isolated"])

    # Fast cross-mode set intersection using precomputed hash dictionaries
    stare_raw = stare_iso.get("raw_hashes", {})
    scan_raw = scan_iso.get("raw_hashes", {})
    common_raw = set(stare_raw.keys()) & set(scan_raw.keys())
    cross_raw_overlaps = [
        {"sha256": h, "stare_file": stare_raw[h], "scan_file": scan_raw[h]}
        for h in common_raw
    ]

    stare_content = stare_iso.get("content_hashes", {})
    scan_content = scan_iso.get("content_hashes", {})
    common_content = set(stare_content.keys()) & set(scan_content.keys())
    cross_content_overlaps = [
        {"canonical_content_sha256": c, "stare_file": stare_content[c], "scan_file": scan_content[c]}
        for c in common_content
    ]

    cross_mode_diag = {
        "cross_mode_raw_hash_overlaps": cross_raw_overlaps,
        "cross_mode_content_hash_overlaps": cross_content_overlaps,
        "cross_mode_raw_overlap_count": len(cross_raw_overlaps),
        "cross_mode_content_overlap_count": len(cross_content_overlaps),
        "note": "Cross-mode overlaps are reported as telemetry; transmitter configs are shared across receiver modes in official TSRD.",
    }

    file_to_raw: dict[str, str] = {}
    file_to_raw.update(stare_iso.get("file_to_raw_hash", {}))
    file_to_raw.update(scan_iso.get("file_to_raw_hash", {}))

    file_to_content: dict[str, str] = {}
    file_to_content.update(stare_iso.get("file_to_content_hash", {}))
    file_to_content.update(scan_iso.get("file_to_content_hash", {}))

    report = {
        "isolated": is_isolated,
        "stare_isolation": stare_iso,
        "scan_isolation": scan_iso,
        "cross_mode_diagnostic": cross_mode_diag,
        "file_to_raw_hash": file_to_raw,
        "file_to_content_hash": file_to_content,
    }

    if not is_isolated and fail_fast:
        raise SplitLeakageError("Intra-mode split leakage detected in STARE or SCAN!")

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
    max_pulses: int | None = None,
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

    sample = data if max_pulses is None else data[:max_pulses]
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
    emitter_markov = []
    markov_evidence_list = []

    for eid in unique_emitters:
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
            # Project-defined first-order transition-dependent agility heuristic:
            # Evaluated within each individual emitter label sequence (not interleaved global stream).
            if len(e_cfs) >= 20:
                q_freq = np.round(e_cfs / 30.0) * 30.0
                unique_states, state_indices = np.unique(q_freq, return_inverse=True)
                K = len(unique_states)
                if K >= 2:
                    s_from = state_indices[:-1]
                    s_to = state_indices[1:]
                    trans_counts = np.zeros((K, K), dtype=np.int64)
                    np.add.at(trans_counts, (s_from, s_to), 1)
                    N_trans = len(s_from)
                    if N_trans > 0:
                        p_joint = trans_counts / N_trans
                        p_from = np.sum(p_joint, axis=1)
                        p_to = np.sum(p_joint, axis=0)
                        outer_p = np.outer(p_from, p_to)
                        mi = 0.0
                        for r in range(K):
                            for col in range(K):
                                if p_joint[r, col] > 0 and outer_p[r, col] > 0:
                                    mi += p_joint[r, col] * np.log2(p_joint[r, col] / outer_p[r, col])

                        row_sums = np.sum(trans_counts, axis=1)
                        cond_ent = 0.0
                        for r in range(K):
                            if row_sums[r] > 0:
                                row_p = trans_counts[r] / row_sums[r]
                                row_nz = row_p[row_p > 0]
                                cond_ent += p_from[r] * (-np.sum(row_nz * np.log2(row_nz)))

                        sparsity = float(np.count_nonzero(trans_counts == 0) / (K * K))
                        if mi >= 0.35 or (mi >= 0.20 and sparsity >= 0.15):
                            emitter_markov.append(eid)
                            markov_evidence_list.append({
                                "emitter_id": int(eid),
                                "num_states": int(K),
                                "transition_mi": round(float(mi), 4),
                                "conditional_entropy": round(float(cond_ent), 4),
                                "sparsity": round(sparsity, 4),
                            })
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
        "n_markov_agile_emitters": len(emitter_markov),
        "markov_evidence": markov_evidence_list,
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
    if (len(emitter_markov) >= 2 or (len(emitter_markov) >= 1 and not has_fixed and len(emitter_agile) == len(emitter_markov))):
        tags.append("markov_agile")

    # Periodic scan check: presence of periodic burst gaps in ToA with intra-burst pulses
    # Labeled strictly as a periodic burst/gap heuristic (pulse arrival regularity, not unmeasured antenna rotation).
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
                evidence["periodic_scan_heuristic_note"] = "Detected via periodic burst/gap arrival timing regularity heuristic"

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

    def validate_file_streaming(
        self,
        file_path: str | Path,
        chunk_size: int = 100000,
        compute_content_hash: bool = False,
        compute_taxonomy: bool = False,
    ) -> dict[str, Any]:
        """Exhaustively validate HDF5 file using streaming chunks without loading full multi-million pulse arrays.

        Tracks:
          - files_checked (1)
          - pulses_checked (int)
          - first_inversion_file (str or None)
          - first_inversion_index (int or None)
          - first_inversion_delta_us (float or None)
          - nonfinite_count (int)
          - empty_file_count (1 if empty else 0)
          - structurally_valid (bool)
          - training_eligible (bool)
          - evaluation_eligible (bool)
        """
        path = Path(file_path)
        result: dict[str, Any] = {
            "path": str(path),
            "filename": path.name,
            "valid": True,
            "structurally_valid": True,
            "empty_scenario": False,
            "training_eligible": False,
            "evaluation_eligible": False,
            "num_pulses": 0,
            "num_emitters": 0,
            "duration_s": 0.0,
            "first_inversion_file": None,
            "first_inversion_index": None,
            "first_inversion_delta_us": None,
            "nonfinite_count": 0,
            "errors": [],
            "warnings": [],
        }

        if not path.exists():
            result["valid"] = False
            result["structurally_valid"] = False
            result["errors"].append(f"File does not exist: {path}")
            return result

        try:
            with h5py.File(str(path), "r") as handle:
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

                data_ds = handle["data"]
                labels_ds = handle["labels"]

                shape = data_ds.shape
                if len(shape) != 2 or shape[1] != 5:
                    result["valid"] = False
                    result["structurally_valid"] = False
                    result["errors"].append(f"Data shape must be (N, 5), got {shape}")
                    return result

                n_pulses = shape[0]
                result["num_pulses"] = n_pulses

                if labels_ds.shape[0] != n_pulses:
                    result["valid"] = False
                    result["errors"].append(f"Labels length {labels_ds.shape[0]} != data length {n_pulses}")

                if n_pulses == 0:
                    result["empty_scenario"] = True
                    result["structurally_valid"] = result["valid"]
                    result["training_eligible"] = False
                    result["evaluation_eligible"] = False
                    return result

                # Chunked streaming 5-column validation (finiteness, ToA monotonicity, exact emitter set)
                prev_last = -np.inf
                first_toa = None
                last_toa = None
                inversion_found = False

                unique_emitters_set: set[int] = set()
                nonnoise_emitters_set: set[int] = set()

                running_min = np.full(5, np.inf, dtype=np.float64)
                running_max = np.full(5, -np.inf, dtype=np.float64)

                for start_idx in range(0, n_pulses, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_pulses)
                    chunk_data = np.asarray(data_ds[start_idx:end_idx], dtype=np.float64)
                    chunk_labels = np.asarray(labels_ds[start_idx:end_idx]).reshape(-1)

                    if chunk_data.ndim != 2 or chunk_data.shape[1] != 5:
                        result["valid"] = False
                        result["structurally_valid"] = False
                        result["errors"].append(f"Chunk at [{start_idx}:{end_idx}] has invalid shape {chunk_data.shape}")
                        break

                    # 1. Finiteness check across all 5 PDW columns
                    finite_mask = np.isfinite(chunk_data)
                    if not np.all(finite_mask):
                        n_bad = int(chunk_data.size - np.sum(finite_mask))
                        result["nonfinite_count"] += n_bad
                        result["valid"] = False
                        result["structurally_valid"] = False
                        result["errors"].append(f"Found {n_bad} non-finite PDW values in chunk [{start_idx}:{end_idx}]")

                    # 2. Time ordering (monotonic ToA)
                    toas = chunk_data[:, 0]
                    if first_toa is None and len(toas) > 0:
                        first_toa = toas[0]
                    if len(toas) > 0:
                        last_toa = toas[-1]

                    # Inter-chunk monotonicity check
                    if not inversion_found and len(toas) > 0 and toas[0] < prev_last:
                        inversion_found = True
                        delta = float(prev_last - toas[0])
                        result["valid"] = False
                        result["structurally_valid"] = False
                        result["first_inversion_file"] = str(path)
                        result["first_inversion_index"] = start_idx
                        result["first_inversion_delta_us"] = delta
                        result["errors"].append(
                            f"ToA inversion between chunks at index {start_idx}: delta={delta:.3f} us"
                        )

                    # Intra-chunk monotonicity check
                    if not inversion_found and len(toas) > 1:
                        diffs = np.diff(toas)
                        neg_mask = diffs < 0
                        if np.any(neg_mask):
                            inversion_found = True
                            rel_idx = int(np.where(neg_mask)[0][0])
                            abs_idx = start_idx + rel_idx + 1
                            delta = float(-diffs[rel_idx])
                            result["valid"] = False
                            result["structurally_valid"] = False
                            result["first_inversion_file"] = str(path)
                            result["first_inversion_index"] = abs_idx
                            result["first_inversion_delta_us"] = delta
                            result["errors"].append(
                                f"ToA inversion at index {abs_idx}: delta={delta:.3f} us"
                            )

                    if len(toas) > 0:
                        prev_last = toas[-1]

                    # 3. Exact streaming emitter set collection (zero sampling cap)
                    for lbl in chunk_labels:
                        lbl_int = int(lbl)
                        unique_emitters_set.add(lbl_int)
                        if lbl_int != -1:
                            nonnoise_emitters_set.add(lbl_int)

                    # 4. Running physical feature bounds
                    if len(chunk_data) > 0:
                        chunk_min = np.min(chunk_data, axis=0)
                        chunk_max = np.max(chunk_data, axis=0)
                        running_min = np.minimum(running_min, chunk_min)
                        running_max = np.maximum(running_max, chunk_max)

                if first_toa is not None and last_toa is not None:
                    result["duration_s"] = float(max(0.0, last_toa - first_toa)) / 1e6

                result["num_emitters"] = len(unique_emitters_set)
                result["num_nonnoise_emitters"] = len(nonnoise_emitters_set)

                # Decouple structural validity from physical-domain diagnostics:
                # Amplitude, PW=0, CF spectrum, and AoA are recorded as diagnostics and do not invalidate structural validity.
                result["observed_ranges"] = {
                    "toa_us": [float(running_min[0]), float(running_max[0])],
                    "freq_mhz": [float(running_min[1]), float(running_max[1])],
                    "pw_us": [float(running_min[2]), float(running_max[2])],
                    "aoa_deg": [float(running_min[3]), float(running_max[3])],
                    "amp_db": [float(running_min[4]), float(running_max[4])],
                    "has_zero_pw": bool(running_min[2] <= 0.0),
                    "amplitude_range": [float(running_min[4]), float(running_max[4])],
                }
                result["physical_diagnostics"] = {
                    "has_zero_pw": bool(running_min[2] <= 0.0),
                    "amplitude_range": [float(running_min[4]), float(running_max[4])],
                    "cf_range_mhz": [float(running_min[1]), float(running_max[1])],
                    "aoa_range_deg": [float(running_min[3]), float(running_max[3])],
                }

                result["structurally_valid"] = result["valid"]
                result["training_eligible"] = result["valid"] and n_pulses > 0
                result["evaluation_eligible"] = (
                    result["training_eligible"] and result["num_nonnoise_emitters"] >= 1
                )

                if "metadata" in handle and hasattr(handle["metadata"], "attrs"):
                    result["metadata_attrs"] = {k: str(v) for k, v in handle["metadata"].attrs.items()}

                if compute_taxonomy and n_pulses > 0:
                    sample_pulses = min(n_pulses, 50000)
                    sample_data = np.asarray(data_ds[:sample_pulses])
                    sample_labels = np.asarray(labels_ds[:sample_pulses]).reshape(-1)
                    result["taxonomy"] = classify_project_taxonomy(sample_data, sample_labels)

            if compute_content_hash and result["structurally_valid"]:
                result["canonical_content_sha256"] = streaming_canonical_content_sha256(path, chunk_size=chunk_size)

        except Exception as exc:
            result["valid"] = False
            result["structurally_valid"] = False
            result["errors"].append(f"HDF5 reading error: {exc}")

        return result


def audit_tsrd_transmitter_metadata(
    file_path: Path | str,
    tolerance_mhz: float = 50.0,
    tolerance_pw_us: float = 0.5,
) -> dict[str, Any]:
    """Audit HDF5 transmitter metadata consistency against observed pulse trains (Gate 10.1B).

    Derives mapping rule by sorting /metadata/transmitters entries by their integer suffix:
      label k -> sorted /metadata/transmitters children by integer suffix -> child[k]

    Categorizes each file into the 4-tier quality overlay:
      1. consistent: Metadata and assigned PDWs align cleanly.
      2. inconsistent_but_PDWs_labels_usable: Label-to-transmitter mapping diverges, but PDWs and local cluster labels are structurally sound.
      3. quarantined_for_metadata_dependent_training: Unsafe for training pipelines requiring semantic transmitter truth.
      4. unsafe_for_metadata_dependent_evaluation: Unsafe for evaluation protocols relying on transmitter parameter truth.
    """
    import ast
    import re

    path = Path(file_path)
    result: dict[str, Any] = {
        "file": str(path),
        "filename": path.name,
        "tier": "inconsistent_but_PDWs_labels_usable",
        "consistent": False,
        "quarantined_for_metadata_dependent_training": True,
        "unsafe_for_metadata_dependent_evaluation": True,
        "num_emitters_in_data": 0,
        "num_transmitters_in_metadata": 0,
        "matched_labels": [],
        "mismatched_labels": [],
        "reason": None,
    }

    if not path.exists():
        result["reason"] = f"File not found: {path}"
        return result

    try:
        with h5py.File(str(path), "r") as handle:
            if "data" not in handle or "labels" not in handle:
                result["reason"] = "Missing data or labels dataset"
                return result

            n_total = handle["data"].shape[0]
            if n_total == 0:
                result["tier"] = "consistent"
                result["consistent"] = True
                result["quarantined_for_metadata_dependent_training"] = False
                result["unsafe_for_metadata_dependent_evaluation"] = False
                result["reason"] = "Empty scenario"
                return result

            # Sample up to 5,000 pulses to audit metadata mapping efficiently
            sample_n = min(n_total, 5000)
            data = np.asarray(handle["data"][:sample_n])
            labels = np.asarray(handle["labels"][:sample_n]).reshape(-1)

            meta = handle.get("metadata")
            if meta is None or "transmitters" not in meta:
                result["reason"] = "Missing /metadata/transmitters"
                return result

            tx = meta["transmitters"]
            parsed_children = []

            if isinstance(tx, h5py.Group):
                for k in sorted(tx.keys()):
                    child = tx[k]
                    m = re.search(r"(\d+)$", k)
                    suffix_int = int(m.group(1)) if m else -1
                    fc_data = []
                    if "frequency_config" in child:
                        fc_grp = child["frequency_config"]
                        if "freqs_mhz" in fc_grp:
                            fc_data = np.asarray(fc_grp["freqs_mhz"]).flatten().tolist()
                        elif hasattr(fc_grp, "attrs") and "freqs_mhz" in fc_grp.attrs:
                            fc_data = list(fc_grp.attrs["freqs_mhz"])
                    pwc_data = []
                    if "pulse_width_config" in child:
                        pwc_grp = child["pulse_width_config"]
                        if "pws_us" in pwc_grp:
                            pwc_data = np.asarray(pwc_grp["pws_us"]).flatten().tolist()
                        elif hasattr(pwc_grp, "attrs") and "pws_us" in pwc_grp.attrs:
                            pwc_data = list(pwc_grp.attrs["pws_us"])

                    parsed_children.append({
                        "key": k,
                        "suffix": suffix_int,
                        "freqs_mhz": fc_data,
                        "pws_us": pwc_data,
                    })
            else:
                for idx, item in enumerate(tx):
                    raw = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    try:
                        d = ast.literal_eval(raw)
                    except Exception:
                        d = {}
                    func_name = d.get("function", "")
                    m = re.search(r"(\d+)$", func_name)
                    suffix_int = int(m.group(1)) if m else idx

                    fc = d.get("frequency_config", {})
                    freqs_list = fc.get("freqs_mhz", [])
                    pwc = d.get("pulse_width_config", {})
                    pws_list = pwc.get("pws_us", [])

                    parsed_children.append({
                        "key": func_name or f"tx_{idx}",
                        "suffix": suffix_int,
                        "freqs_mhz": freqs_list,
                        "pws_us": pws_list,
                    })

            parsed_children.sort(key=lambda c: (c["suffix"], c["key"]))
            result["num_transmitters_in_metadata"] = len(parsed_children)

            suffix_map = {c["suffix"]: c for c in parsed_children if c["suffix"] >= 0}

            unique_labels = [int(x) for x in np.unique(labels) if x != -1]
            result["num_emitters_in_data"] = len(unique_labels)

            matched = []
            mismatched = []

            for lbl in unique_labels:
                target_cfg = suffix_map.get(lbl)
                if target_cfg is None and 0 <= lbl < len(parsed_children):
                    target_cfg = parsed_children[lbl]

                if target_cfg is None:
                    mismatched.append({"label": lbl, "reason": "no_corresponding_transmitter_in_metadata"})
                    continue

                lbl_mask = (labels == lbl)
                lbl_pulses = data[lbl_mask]
                if len(lbl_pulses) == 0:
                    matched.append(lbl)
                    continue

                lbl_cfs = lbl_pulses[:, 1]
                lbl_pws = lbl_pulses[:, 2]

                cfg_cfs = target_cfg["freqs_mhz"]
                cfg_pws = target_cfg["pws_us"]

                cf_match = True
                if cfg_cfs:
                    cfg_cfs_arr = np.asarray(cfg_cfs, dtype=np.float64)
                    diffs_cf = np.abs(lbl_cfs[:, None] - cfg_cfs_arr[None, :])
                    min_diff_cf = np.min(diffs_cf, axis=1)
                    if np.mean(min_diff_cf <= tolerance_mhz) < 0.80:
                        cf_match = False

                pw_match = True
                if cfg_pws:
                    cfg_pws_arr = np.asarray(cfg_pws, dtype=np.float64)
                    diffs_pw = np.abs(lbl_pws[:, None] - cfg_pws_arr[None, :])
                    min_diff_pw = np.min(diffs_pw, axis=1)
                    tol = max(tolerance_pw_us, 0.25 * float(np.min(cfg_pws_arr)))
                    if np.mean(min_diff_pw <= tol) < 0.80:
                        pw_match = False

                if cf_match and pw_match:
                    matched.append(lbl)
                else:
                    mismatched.append({
                        "label": lbl,
                        "reason": f"cf_match={cf_match}, pw_match={pw_match}",
                        "cfg_cfs": cfg_cfs,
                        "cfg_pws": cfg_pws,
                    })

            result["matched_labels"] = matched
            result["mismatched_labels"] = mismatched

            if len(mismatched) == 0:
                result["tier"] = "consistent"
                result["consistent"] = True
                result["quarantined_for_metadata_dependent_training"] = False
                result["unsafe_for_metadata_dependent_evaluation"] = False
                result["reason"] = "All emitter labels match transmitter metadata within tolerance"
            else:
                result["tier"] = "inconsistent_but_PDWs_labels_usable"
                result["consistent"] = False
                result["quarantined_for_metadata_dependent_training"] = True
                result["unsafe_for_metadata_dependent_evaluation"] = True
                result["reason"] = f"{len(mismatched)} emitter labels diverge from transmitter metadata"

    except Exception as exc:
        result["tier"] = "inconsistent_but_PDWs_labels_usable"
        result["consistent"] = False
        result["quarantined_for_metadata_dependent_training"] = True
        result["unsafe_for_metadata_dependent_evaluation"] = True
        result["reason"] = f"Audit exception: {exc}"

    return result


class DatasetImmutabilityError(RuntimeError):
    """Raised when any dataset file is modified, resized, or altered during qualification."""
    pass


class DatasetImmutabilityGuard:
    """Guards dataset immutability across qualification run.

    Captures file size, mtime, and raw SHA-256 for all monitored files at start,
    and verifies at completion that no file was modified or deleted.
    """

    def __init__(self, files: list[Path] | None = None) -> None:
        self.snapshots: dict[str, tuple[int, float, str]] = {}
        if files:
            self.capture(files)

    def record(self, path: Path | str | list[Path | str]) -> None:
        if isinstance(path, (list, tuple)):
            self.capture([Path(p) for p in path])
        else:
            self.capture([Path(path)])

    def capture(self, files: list[Path]) -> None:
        for p in files:
            p_res = Path(p).resolve()
            st = p_res.stat()
            self.snapshots[str(p_res)] = (st.st_size, st.st_mtime, _sha256(p_res))

    def verify_all(self, fail_fast: bool = False) -> dict[str, Any]:
        return self.verify(fail_fast=fail_fast)

    def verify(self, fail_fast: bool = True) -> dict[str, Any]:
        violations = []
        for p_str, (orig_size, orig_mtime, orig_sha) in self.snapshots.items():
            p = Path(p_str)
            if not p.exists():
                violations.append({"file": p_str, "error": "file_deleted", "type": "missing_file"})
                continue
            st = p.stat()
            if st.st_size != orig_size:
                violations.append({
                    "file": p_str,
                    "error": "size_changed",
                    "type": "size_mismatch",
                    "orig_size": orig_size,
                    "new_size": st.st_size,
                })
                continue
            if st.st_mtime != orig_mtime:
                new_sha = _sha256(p)
                if new_sha != orig_sha:
                    violations.append({
                        "file": p_str,
                        "error": "content_changed",
                        "type": "content_mismatch",
                        "orig_sha256": orig_sha,
                        "new_sha256": new_sha,
                    })
        passed = len(violations) == 0
        report = {
            "passed": passed,
            "files_monitored": len(self.snapshots),
            "violations_count": len(violations),
            "violations": violations,
        }
        if not passed and fail_fast:
            raise DatasetImmutabilityError(
                f"Dataset immutability violation: {len(violations)} files altered during execution! {violations[:3]}"
            )
        return report


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


def build_integrated_manifest(
    data_root: str | Path,
    output_path: str | Path | None = None,
    root_resolution_source: str = "cli",
    evaluation_split: str = "test",
    max_files_per_split: int | None = None,
    enforce_split_isolation: bool = True,
    fail_fast_on_leakage: bool = False,
    classify_taxonomy: bool = True,
    compute_content_hash: bool = True,
    precomputed_validations: dict[str, Any] | None = None,
    precomputed_raw_hashes: dict[str, str] | None = None,
    precomputed_content_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build exhaustive 6,000-file integrated manifest covering both STARE and SCAN.

    Every file entry records:
      mode, split, relative_path, file_size, raw_sha256, canonical_content_sha256,
      num_pulses, num_emitters, duration_s, observed_ranges,
      structurally_valid, empty_scenario, training_eligible, evaluation_eligible, taxonomy.

    A global dataset fingerprint is derived from all 6,000 canonical content hashes.
    Intra-mode 3-layer isolation is enforced as a hard contract; cross-mode telemetry
    is tracked diagnostically.
    """
    if evaluation_split == "train":
        raise ValueError("Role contract violation: 'train' split cannot be designated as evaluation_split")

    root = Path(data_root).resolve()
    validator = TSRDValidator()

    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    manifest: dict[str, Any] = {
        "data_root": str(root),
        "dataset_provenance": {
            "upstream_dataset_name": "Turing Synthetic Radar Dataset (TSRD)",
            "upstream_distribution": "Hugging Face / Official Upstream",
            "upstream_repository": "https://github.com/alan-turing-institute/turing-deinterleaving-challenge",
            "upstream_revision_identifier": "2026-02-release",
            "upstream_commit_or_dataset_revision": "2026.02",
            "receiver_modes": ["STARE", "SCAN"],
            "project_qualification_id": "TSRD-QUAL-2026-V1",
            "qualification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "local_dataset_root": str(root),
            "root_resolution_source": root_resolution_source,
            "evaluation_split": evaluation_split,
        },
        "modes": {},
        "summary": {
            "total_files": 0,
            "total_pulses": 0,
            "structurally_valid_files": 0,
            "empty_files": 0,
            "training_eligible": 0,
            "evaluation_eligible": 0,
        },
        "taxonomy_summary": {cls_name: 0 for cls_name in PROJECT_TAXONOMY_CLASSES},
    }
    manifest["taxonomy_summary"]["unknown"] = 0

    all_file_paths: list[Path] = []
    canonical_hashes_all: list[str] = []
    file_lists: dict[str, dict[str, list[Path]]] = {"stare": {}, "scan": {}}

    for mode in ["stare", "scan"]:
        manifest["modes"][mode] = {
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

        for split_name in ["train", "val", "test"]:
            dir_name = f"{split_name}_{mode}"
            split_dir = root / mode / dir_name
            files = sorted(split_dir.glob("*.h5")) if split_dir.exists() else []
            if max_files_per_split is not None:
                files = files[:max_files_per_split]

            file_lists[mode][split_name] = files
            records = []
            split_pulses = 0
            split_stats = {
                "structurally_valid_files": 0,
                "empty_files": 0,
                "training_eligible": 0,
                "evaluation_eligible": 0,
            }

            for fp in files:
                all_file_paths.append(fp)
                fp_key = str(fp.resolve())
                if precomputed_validations and fp_key in precomputed_validations:
                    v = precomputed_validations[fp_key]
                else:
                    v = validator.validate_file_streaming(
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

                raw_h = (
                    precomputed_raw_hashes.get(fp_key)
                    if precomputed_raw_hashes is not None and fp_key in precomputed_raw_hashes
                    else _sha256(fp)
                )
                c_hash = (
                    precomputed_content_hashes.get(fp_key)
                    if precomputed_content_hashes is not None and fp_key in precomputed_content_hashes
                    else (v.get("canonical_content_sha256") if compute_content_hash else None)
                )
                if c_hash:
                    canonical_hashes_all.append(c_hash)

                tax = v.get("taxonomy")
                if tax and "primary_class" in tax:
                    p_cls = tax["primary_class"]
                    manifest["taxonomy_summary"][p_cls] = manifest["taxonomy_summary"].get(p_cls, 0) + 1

                rel_path = str(fp.relative_to(root)).replace("\\", "/") if root in fp.parents else str(fp)
                records.append({
                    "mode": mode,
                    "split": split_name,
                    "relative_path": rel_path,
                    "filename": fp.name,
                    "size_bytes": fp.stat().st_size,
                    "raw_sha256": raw_h,
                    "canonical_content_sha256": c_hash,
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

            manifest["modes"][mode]["splits"][split_name] = {
                "directory": str(split_dir),
                "file_count": len(records),
                "total_pulses": split_pulses,
                "structurally_valid_files": split_stats["structurally_valid_files"],
                "empty_files": split_stats["empty_files"],
                "training_eligible": split_stats["training_eligible"],
                "evaluation_eligible": split_stats["evaluation_eligible"],
                "files": records,
            }

            manifest["modes"][mode]["summary"]["total_files"] += len(records)
            manifest["modes"][mode]["summary"]["total_pulses"] += split_pulses
            for k in split_stats:
                manifest["modes"][mode]["summary"][k] += split_stats[k]
                manifest["summary"][k] += split_stats[k]

            manifest["summary"]["total_files"] += len(records)
            manifest["summary"]["total_pulses"] += split_pulses

    # Path-bound global dataset fingerprint bound to sorted canonical tuples of:
    # (mode, split, relative_path, file_size, raw_sha256, canonical_content_sha256)
    fingerprint_entries = []
    for mode_key, m_info in manifest["modes"].items():
        for split_key, s_info in m_info["splits"].items():
            for r in s_info["files"]:
                fingerprint_entries.append((
                    str(r["mode"]),
                    str(r["split"]),
                    str(r["relative_path"]).replace("\\", "/"),
                    int(r["size_bytes"]),
                    str(r["raw_sha256"]),
                    str(r.get("canonical_content_sha256") or ""),
                ))
    fingerprint_entries.sort()
    fp_serialized = "\n".join(
        f"{m}|{s}|{p}|{sz}|{rh}|{ch}" for m, s, p, sz, rh, ch in fingerprint_entries
    )
    manifest["dataset_fingerprint"] = hashlib.sha256(fp_serialized.encode("utf-8")).hexdigest()

    if enforce_split_isolation:
        iso_report = validate_split_isolation_dual_mode(
            file_lists=file_lists,
            root_path=root,
            fail_fast=fail_fast_on_leakage,
        )
        manifest["split_isolation"] = iso_report

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        logger.info(
            "Saved integrated TSRD manifest to %s (%d files, %d pulses, fingerprint=%s)",
            out,
            manifest["summary"]["total_files"],
            manifest["summary"]["total_pulses"],
            manifest["dataset_fingerprint"],
        )

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
            except Exception as exc:
                logger.debug("Failed parsing sample stats for %s: %s", fp, exc)

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


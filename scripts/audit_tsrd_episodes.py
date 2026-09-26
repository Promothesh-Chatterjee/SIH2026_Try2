"""
Audit TSRD Episodes: Measure temporal diversity, pulse retention, and emitter preservation.

Verifies whether truncating at max_pulses=50,000 retains the complete temporal
and spectral diversity of each TSRD scenario.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_file(
    h5_path: Path,
    max_pulses: int = 50000,
    n_bands: int = 36,
    freq_min: float = 0.0,
    freq_max: float = 18000.0,
) -> dict[str, Any]:
    """Audit a single TSRD H5 file.

    Raw H5 data are never modified in place.
    Raw temporal ordering is inspected for monotonicity and duplicate timestamps.
    A local copy is sorted by ToA using stable sort so that retained pulses and
    all temporal/spectral statistics are computed from a strictly chronological sequence.
    """
    file_sha256 = _sha256_file(h5_path) if h5_path.is_file() else ""

    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f["data"][:], dtype=np.float64) if "data" in f else np.zeros((0, 5), dtype=np.float64)
        labels = np.asarray(f["labels"][:], dtype=np.int64).flatten() if "labels" in f else np.zeros(len(data), dtype=np.int64)

    total_pulses = len(data)
    retained_pulses = min(total_pulses, max_pulses)
    discarded_pulses = max(0, total_pulses - max_pulses)
    retention_pct = (retained_pulses / total_pulses * 100.0) if total_pulses > 0 else 0.0

    # 1. Raw temporal ordering inspection (before any sorting)
    if total_pulses > 1:
        raw_toas = data[:, 0]
        raw_diffs = np.diff(raw_toas)
        raw_negative_delta_count = int(np.sum(raw_diffs < 0))
        raw_duplicate_toa_count = int(np.sum(raw_diffs == 0))
        raw_toa_monotonic = bool(raw_negative_delta_count == 0)
    else:
        raw_negative_delta_count = 0
        raw_duplicate_toa_count = 0
        raw_toa_monotonic = True

    # 2. Stable chronological sort of local copy
    if total_pulses > 0:
        order = np.argsort(data[:, 0], kind="stable")
        data_sorted = data[order]
        labels_sorted = labels[order]
    else:
        data_sorted = data.copy()
        labels_sorted = labels.copy()

    # Retained pulses must be taken from the chronologically sorted sequence
    retained = data_sorted[:retained_pulses]
    retained_labels = labels_sorted[:retained_pulses]

    # 3. Total and retained scenario duration (computed strictly on sorted sequence)
    total_duration_us = float(data_sorted[-1, 0] - data_sorted[0, 0]) if total_pulses > 1 else 0.0
    retained_duration_us = float(retained[-1, 0] - retained[0, 0]) if retained_pulses > 1 else 0.0
    duration_coverage_pct = (retained_duration_us / total_duration_us * 100.0) if total_duration_us > 0 else 100.0

    # 4. Emitter retention (computed from sorted labels)
    all_emitters = set(int(x) for x in labels_sorted if int(x) >= 0)
    retained_emitters = set(int(x) for x in retained_labels if int(x) >= 0)
    emitter_retention_pct = (len(retained_emitters) / len(all_emitters) * 100.0) if all_emitters else 100.0

    # 5. Band distribution (computed from sorted sequence)
    band_width = (freq_max - freq_min) / float(n_bands)
    cfs_all = data_sorted[:, 1] if total_pulses > 0 else np.array([], dtype=np.float64)
    cfs_retained = retained[:, 1] if retained_pulses > 0 else np.array([], dtype=np.float64)
    bands_all = np.clip(((cfs_all - freq_min) / band_width).astype(int), 0, n_bands - 1) if total_pulses > 0 else np.array([], dtype=int)
    bands_retained = np.clip(((cfs_retained - freq_min) / band_width).astype(int), 0, n_bands - 1) if retained_pulses > 0 else np.array([], dtype=int)

    distinct_bands_all = int(len(set(bands_all)))
    distinct_bands_retained = int(len(set(bands_retained)))

    # 6. PRI statistics (computed strictly from positive deltas of retained sorted sequence)
    toas_retained = retained[:, 0] if retained_pulses > 0 else np.array([], dtype=np.float64)
    pris = np.diff(toas_retained) if len(toas_retained) > 1 else np.array([], dtype=np.float64)
    pos_pris = pris[pris > 0]

    if len(pos_pris) > 0:
        pri_mean = float(np.mean(pos_pris))
        pri_p50 = float(np.median(pos_pris))
        pri_p90 = float(np.percentile(pos_pris, 90))
        pri_sample_count = int(len(pos_pris))
    else:
        pri_mean = 0.0
        pri_p50 = 0.0
        pri_p90 = 0.0
        pri_sample_count = 0

    return {
        "file": h5_path.name,
        "file_path": str(h5_path).replace("\\", "/"),
        "file_sha256": file_sha256,
        "total_pulses": total_pulses,
        "retained_pulses": retained_pulses,
        "discarded_pulses": discarded_pulses,
        "pulse_retention_pct": round(retention_pct, 2),
        "total_duration_us": round(total_duration_us, 1),
        "retained_duration_us": round(retained_duration_us, 1),
        "duration_coverage_pct": round(duration_coverage_pct, 2),
        "all_emitters_count": len(all_emitters),
        "retained_emitters_count": len(retained_emitters),
        "emitter_retention_pct": round(emitter_retention_pct, 2),
        "distinct_bands_all": distinct_bands_all,
        "distinct_bands_retained": distinct_bands_retained,
        "pri_mean_us": round(pri_mean, 2),
        "pri_p50_us": round(pri_p50, 2),
        "pri_p90_us": round(pri_p90, 2),
        "pri_sample_count": pri_sample_count,
        "raw_toa_monotonic": raw_toa_monotonic,
        "raw_negative_delta_count": raw_negative_delta_count,
        "raw_duplicate_toa_count": raw_duplicate_toa_count,
        "sorted_for_temporal_stats": True,
    }


def main():
    default_data_dir = "D:/TSRD" if Path("D:/TSRD").is_dir() else "data"
    parser = argparse.ArgumentParser(description="Audit TSRD H5 scenario temporal & spectral diversity")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Path to directory with TSRD H5 files")
    parser.add_argument("--max-pulses", type=int, default=50000, help="Cap used in training episodes")
    parser.add_argument("--out", type=str, default="tsrd_audit_report.json", help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files audited")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h5_files = sorted(list(data_dir.rglob("*.h5")))
    if not h5_files:
        logger.warning("No .h5 files found in %s", data_dir)
        return

    if args.limit:
        h5_files = h5_files[:args.limit]

    logger.info("Found %d H5 files in %s. Auditing with max_pulses=%d...", len(h5_files), data_dir, args.max_pulses)
    reports = []
    seen_paths: set[str] = set()
    hash_to_paths: dict[str, list[str]] = {}

    print("\n" + "=" * 145)
    print(f"{'File':<25} | {'Pulses (Ret/Tot)':<18} | {'Ret %':<7} | {'Duration (Ret/Tot µs)':<24} | {'Dur %':<7} | {'PRI p50 (µs)':<12} | {'Mono':<5} | {'Emitters'}")
    print("-" * 145)

    for p in h5_files:
        try:
            rep = audit_file(p, max_pulses=args.max_pulses)
            reports.append(rep)
            p_key = rep["file_path"]
            seen_paths.add(p_key)
            if rep["file_sha256"]:
                hash_to_paths.setdefault(rep["file_sha256"], []).append(p_key)

            p_str = f"{rep['retained_pulses']}/{rep['total_pulses']}"
            d_str = f"{rep['retained_duration_us']:.0f}/{rep['total_duration_us']:.0f}"
            e_str = f"{rep['retained_emitters_count']}/{rep['all_emitters_count']}"
            mono_str = "YES" if rep["raw_toa_monotonic"] else "NO"
            print(f"{rep['file'][:25]:<25} | {p_str:<18} | {rep['pulse_retention_pct']:5.1f}% | {d_str:<24} | {rep['duration_coverage_pct']:5.1f}% | {rep['pri_p50_us']:12.2f} | {mono_str:<5} | {e_str}")
        except Exception as exc:
            logger.error("Error auditing %s: %s", p, exc)

    print("=" * 145 + "\n")

    # Current commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "UNKNOWN"

    duplicate_hash_groups = sum(1 for paths in hash_to_paths.values() if len(paths) > 1)
    temporal_order_violations = sum(1 for r in reports if not r["raw_toa_monotonic"])

    envelope = {
        "dataset_root": str(data_dir).replace("\\", "/"),
        "dataset_kind": "TSRD" if "tsrd" in str(data_dir).lower() else "LOCAL_FIXTURE",
        "git_commit": git_commit,
        "audit_run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "unique_files_audited": len(seen_paths),
        "report_entries": len(reports),
        "duplicate_report_entries": len(reports) - len(seen_paths),
        "duplicate_content_hash_groups": duplicate_hash_groups,
        "temporal_order_violations": temporal_order_violations,
        "reports": reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)
    logger.info("Saved audit report to %s (%d entries, %d hash duplicates, %d temporal violations)",
                out_path, len(reports), duplicate_hash_groups, temporal_order_violations)


if __name__ == "__main__":
    main()

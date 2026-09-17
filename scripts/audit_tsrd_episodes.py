"""
Audit TSRD Episodes: Measure temporal diversity, pulse retention, and emitter preservation.

Verifies whether truncating at max_pulses=50,000 retains the complete temporal
and spectral diversity of each TSRD scenario.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def audit_file(h5_path: Path, max_pulses: int = 50000, n_bands: int = 36, freq_min: float = 0.0, freq_max: float = 18000.0) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f["data"][:], dtype=np.float64)
        labels = np.asarray(f["labels"][:], dtype=np.int64) if "labels" in f else np.zeros(len(data), dtype=np.int64)

    total_pulses = len(data)
    retained_pulses = min(total_pulses, max_pulses)
    discarded_pulses = max(0, total_pulses - max_pulses)
    retention_pct = (retained_pulses / total_pulses * 100.0) if total_pulses > 0 else 0.0

    # Total scenario duration
    total_duration_us = float(data[-1, 0] - data[0, 0]) if total_pulses > 1 else 0.0
    retained_duration_us = float(data[retained_pulses - 1, 0] - data[0, 0]) if retained_pulses > 1 else 0.0
    duration_coverage_pct = (retained_duration_us / total_duration_us * 100.0) if total_duration_us > 0 else 100.0

    # Emitter retention
    all_emitters = set(int(x) for x in labels if int(x) >= 0)
    retained_emitters = set(int(x) for x in labels[:retained_pulses] if int(x) >= 0)
    emitter_retention_pct = (len(retained_emitters) / len(all_emitters) * 100.0) if all_emitters else 100.0

    # Band distribution
    band_width = (freq_max - freq_min) / float(n_bands)
    cfs_all = data[:, 1]
    cfs_retained = data[:retained_pulses, 1]
    bands_all = np.clip(((cfs_all - freq_min) / band_width).astype(int), 0, n_bands - 1)
    bands_retained = np.clip(((cfs_retained - freq_min) / band_width).astype(int), 0, n_bands - 1)

    distinct_bands_all = int(len(set(bands_all)))
    distinct_bands_retained = int(len(set(bands_retained)))

    # Burst distribution (PRI statistics)
    toas_retained = data[:retained_pulses, 0]
    pris = np.diff(toas_retained) if len(toas_retained) > 1 else np.array([0.0])
    pri_mean = float(np.mean(pris))
    pri_p50 = float(np.median(pris))
    pri_p90 = float(np.percentile(pris, 90)) if len(pris) > 0 else 0.0

    return {
        "file": h5_path.name,
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
    }


def main():
    parser = argparse.ArgumentParser(description="Audit TSRD H5 scenario temporal & spectral diversity")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to directory with TSRD H5 files")
    parser.add_argument("--max-pulses", type=int, default=50000, help="Cap used in training episodes")
    parser.add_argument("--out", type=str, default="tsrd_audit_report.json", help="Output JSON report path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h5_files = sorted(list(data_dir.rglob("*.h5")))
    if not h5_files:
        logger.warning("No .h5 files found in %s", data_dir)
        return

    logger.info("Found %d H5 files in %s. Auditing with max_pulses=%d...", len(h5_files), data_dir, args.max_pulses)
    reports = []
    print("\n" + "=" * 135)
    print(f"{'File':<25} | {'Pulses (Ret/Tot)':<18} | {'Ret %':<7} | {'Duration (Ret/Tot µs)':<24} | {'Dur %':<7} | {'Emitters':<10} | {'Bands'}")
    print("-" * 135)

    for p in h5_files:
        try:
            rep = audit_file(p, max_pulses=args.max_pulses)
            reports.append(rep)
            p_str = f"{rep['retained_pulses']}/{rep['total_pulses']}"
            d_str = f"{rep['retained_duration_us']:.0f}/{rep['total_duration_us']:.0f}"
            e_str = f"{rep['retained_emitters_count']}/{rep['all_emitters_count']}"
            b_str = f"{rep['distinct_bands_retained']}/{rep['distinct_bands_all']}"
            print(f"{rep['file'][:25]:<25} | {p_str:<18} | {rep['pulse_retention_pct']:5.1f}% | {d_str:<24} | {rep['duration_coverage_pct']:5.1f}% | {e_str:<10} | {b_str}")
        except Exception as exc:
            logger.error("Error auditing %s: %s", p, exc)

    print("=" * 135 + "\n")
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    logger.info("Saved audit report to %s", out_path)


if __name__ == "__main__":
    main()

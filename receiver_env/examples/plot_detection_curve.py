"""Empirical Detection Performance Curve Generator (Pd and Pfa vs SNR).

Sweeps SNR across: +15 dB, +12 dB, +10 dB, +8 dB, +6 dB, +4 dB, +3 dB, +2 dB, 0 dB, -2 dB, -3 dB, -6 dB.
Generates an empirical ROC/detection curve and performance scorecard for SIH presentations.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from receiver_env.config import ReceiverConfig
from receiver_env.validation.metrics import evaluate_detections
from receiver_env.validation.validation_runner import (
    SyntheticSignalGenerator,
    ValidationRunner,
)


def run_snr_sweep(
    snr_points_db: List[float],
    pulses_per_snr: int = 50,
    pri_us: float = 60.0,
    pw_us: float = 8.0,
    sample_rate_hz: float = 20_000_000.0,
    center_frequency_hz: float = 3_000_000_000.0,
    seed: int = 2026,
) -> List[Dict[str, float]]:
    """Execute streaming detection evaluation across SNR sweep."""
    config = ReceiverConfig(
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        bandwidth_hz=10_000_000.0,
        snr_margin_db=8.0,
        hysteresis_db=3.0,
        min_pulse_samples=10,
        min_gap_samples=15,
    )
    runner = ValidationRunner(config=config, chunk_size=2048)

    results = []
    print("\nStarting Empirical Detection Curve Evaluation...")
    print("=" * 80)
    print(f" {'SNR (dB)':>8s} | {'Pd (%)':>8s} | {'Pfa (%)':>8s} | {'Matched':>7s} / {'Total':>5s} | {'Latency (spl)':>13s} | {'Status':>10s}")
    print("-" * 80)

    for snr in snr_points_db:
        # Use a distinct seed per SNR for statistically independent noise realization
        snr_seed = seed + int(round(snr * 10))
        gen = SyntheticSignalGenerator(
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,
            seed=snr_seed,
        )

        iq, truth = gen.generate_pulse_train(
            num_pulses=pulses_per_snr,
            pri_us=pri_us,
            pw_us=pw_us,
            snr_db=snr,
            initial_delay_us=30.0,
            freq_offset_hz=1.0e6,
        )

        res = runner.run_scenario(iq, truth, tolerance_samples=30, min_iou=0.25)
        m = res.metrics

        pd_pct = m.pd * 100.0
        pfa_pct = m.pfa * 100.0
        status = "EXCELLENT" if pd_pct >= 95.0 else ("GOOD" if pd_pct >= 80.0 else ("DEGRADED" if pd_pct >= 50.0 else "UNRELIABLE"))

        print(f" {snr:+8.1f} | {pd_pct:7.1f}% | {pfa_pct:7.2f}% | {m.matched_truth_count:7d} / {m.total_truth_pulses:5d} | {m.mean_start_latency:13.2f} | {status:>10s}")

        results.append({
            "snr_db": snr,
            "pd": m.pd,
            "pd_pct": pd_pct,
            "pfa": m.pfa,
            "pfa_pct": pfa_pct,
            "matched": m.matched_truth_count,
            "total_truth": m.total_truth_pulses,
            "total_detected": m.total_detected_pulses,
            "false_alarms": m.false_alarm_count,
            "mean_latency": m.mean_start_latency,
            "split_count": m.split_pulse_count,
            "merge_count": m.merged_pulse_count,
        })

    print("=" * 80)
    return results


def plot_detection_curve(results: List[Dict[str, float]], output_path: Path) -> None:
    """Plot high-resolution publication-quality Pd vs SNR and latency curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snrs = [r["snr_db"] for r in results]
    pds = [r["pd_pct"] for r in results]
    pfas = [r["pfa_pct"] for r in results]
    latencies = [r["mean_latency"] for r in results]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)

    # 1. Detection Probability (Pd) & False Alarm Rate (Pfa)
    ax1.plot(snrs, pds, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Probability of Detection ($P_d$)")
    ax1.axhline(95.0, color="#d62728", linestyle="--", linewidth=1.5, label=r"Phase 1 Target ($P_d \geq 95\%$)")
    ax1.axhline(50.0, color="#7f7f7f", linestyle=":", linewidth=1.0, label="50% Detection Threshold")

    # Highlighting critical operational regions
    ax1.axvspan(6.0, max(snrs), color="#2ca02c", alpha=0.12, label=r"High-Confidence Operational Region ($\geq 6$ dB)")
    ax1.axvspan(0.0, 6.0, color="#ff7f0e", alpha=0.12, label="Sub-Nyquist / Marginal Region (0 to 6 dB)")
    ax1.axvspan(min(snrs), 0.0, color="#d62728", alpha=0.12, label="Below Noise Floor ($< 0$ dB)")

    for s, p in zip(snrs, pds):
        ax1.annotate(
            f"{p:.1f}%",
            (s, p),
            textcoords="offset points",
            xytext=(0, 10 if p < 95 else -15),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#0f3b5c",
        )

    ax1.set_ylabel("Detection Probability $P_d$ (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Cognitive EW Receiver (Phase 1): Empirical Detection Curve ($P_d$ vs SNR)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Mean Boundary Latency
    valid_lat_snrs = [s for s, p, l in zip(snrs, pds, latencies) if p > 0]
    valid_latencies = [l for p, l in zip(pds, latencies) if p > 0]
    ax2.plot(valid_lat_snrs, valid_latencies, "s-", color="#2ca02c", linewidth=2.0, markersize=7, label="Mean Boundary Latency (Samples)")
    # Sample rate is 20 MS/s -> 1 sample = 50 ns
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Leading Edge Latency (Samples)", fontsize=12, fontweight="bold")
    ax2.set_title("Pulse Edge Estimation Latency vs SNR (1 sample = 50 ns)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(valid_latencies + [10]) + 5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Secondary axis for nanoseconds
    secax = ax2.secondary_yaxis("right", functions=(lambda x: x * 50.0, lambda x: x / 50.0))
    secax.set_ylabel("Latency (nanoseconds)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] High-resolution presentation curve saved to: {output_path}")


def main() -> None:
    # Explicit points requested by user: 15, 10, 8, 6, 3, 0, -3 (plus intermediate for smoother curve)
    snr_sweep = [15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 0.0, -2.0, -3.0, -6.0]
    results = run_snr_sweep(snr_sweep, pulses_per_snr=50)

    # Save outputs
    output_dir = workspace_root / "receiver_env" / "reports"
    img_path = output_dir / "detection_curve_pd_vs_snr.png"
    plot_detection_curve(results, img_path)

    # Also save CSV for SIH presentation slides / spreadsheets
    csv_path = output_dir / "detection_benchmark_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("snr_db,pd_pct,pfa_pct,matched,total,mean_latency_samples,latency_ns\n")
        for r in results:
            f.write(f"{r['snr_db']},{r['pd_pct']:.2f},{r['pfa_pct']:.2f},{r['matched']},{r['total_truth']},{r['mean_latency']:.2f},{r['mean_latency']*50.0:.1f}\n")
    print(f"[+] Empirical CSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()

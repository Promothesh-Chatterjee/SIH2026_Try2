"""
Operational Telemetry Visualization & SIH Demonstration Suite.

Generates high-resolution publication-quality figures from operational telemetry:
1. RF Spectrum Waterfall & Cognitive Receiver Aperture Tracking (Time-Frequency).
2. Cognitive Dwell Mode Selection & Adaptive Duration Dynamics.
3. Pulse Intercept Latency & Detection Temporal Precision.
4. Cumulative Interceptions & Multi-Emitter Track Persistence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Path resolution
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from src.environment.scenario_generator import load_h5_records
from src.data.tsrd_root import resolve_tsrd_root
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_operational_telemetry")


def generate_demonstration_dashboard(
    telemetry_path: Path,
    scenario_h5_path: Path,
    output_path: Path,
    max_time_ms: float = 60.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load telemetry frames
    with open(telemetry_path, "r", encoding="utf-8") as f:
        telemetry: List[Dict[str, Any]] = json.load(f)

    # 2. Load ground-truth scenario pulses for RF waterfall background
    records = load_h5_records(
        scenario_h5_path,
        freq_min_mhz=0.0,
        freq_max_mhz=18000.0,
        time_horizon_us=10_000_000.0,
        max_pulses=50000,
    )

    # Filter pulses within visual window
    max_time_us = max_time_ms * 1000.0
    def _get_t(p: Any) -> float:
        return float(getattr(p, "time_us", getattr(p, "toa_us", 0.0)) if not isinstance(p, dict) else p.get("time_us", p.get("toa_us", 0.0)))

    def _get_f(p: Any) -> float:
        return float(getattr(p, "frequency_mhz", getattr(p, "freq_mhz", 0.0)) if not isinstance(p, dict) else p.get("frequency_mhz", p.get("freq_mhz", 0.0)))

    vis_pulses = [p for p in records if _get_t(p) <= max_time_us]
    pulse_times_ms = [_get_t(p) / 1000.0 for p in vis_pulses]
    pulse_freqs_ghz = [_get_f(p) / 1000.0 for p in vis_pulses]

    # Filter telemetry dwells within visual window
    vis_telemetry = [f for f in telemetry if f["dwell_start_us"] <= max_time_us]

    # Set styling
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12), dpi=200)

    # Create grid: Top large waterfall, bottom 4 diagnostic panels
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1.0, 1.0], hspace=0.32, wspace=0.22)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: Time-Frequency RF Spectrum Waterfall & Cognitive Receiver Tuning
    # ──────────────────────────────────────────────────────────────────────────
    ax_waterfall = fig.add_subplot(gs[0, :])
    ax_waterfall.set_title(
        f"Cognitive EW Smart Scan: Autonomous Frequency Tracking & Dwell Apertures ({scenario_h5_path.stem})",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # Plot incident RF pulses
    ax_waterfall.scatter(
        pulse_times_ms,
        pulse_freqs_ghz,
        s=12,
        c="#38bdf8",
        alpha=0.65,
        edgecolors="none",
        label="Incident RF Pulses (TSRD Radar)",
        zorder=2,
    )

    # Draw receiver dwell aperture boxes
    # Blue/Green for Hit, Gray/Red for Miss
    for f in vis_telemetry:
        t_start_ms = f["dwell_start_us"] / 1000.0
        t_duration_ms = f["dwell_duration_us"] / 1000.0
        f_center_ghz = f["center_frequency_mhz"] / 1000.0
        f_bw_ghz = f["bandwidth_mhz"] / 1000.0
        f_low_ghz = f_center_ghz - f_bw_ghz / 2.0

        hit = f["hit"]
        edge_color = "#10b981" if hit else "#ef4444"
        face_color = "#10b981" if hit else "#64748b"
        alpha = 0.30 if hit else 0.12
        lw = 1.5 if hit else 0.8

        rect = patches.Rectangle(
            (t_start_ms, f_low_ghz),
            t_duration_ms,
            f_bw_ghz,
            linewidth=lw,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=alpha,
            zorder=3 if hit else 1,
        )
        ax_waterfall.add_patch(rect)

    # Add custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Incident RF Pulses", markerfacecolor="#38bdf8", markersize=8),
        patches.Patch(facecolor="#10b981", edgecolor="#10b981", alpha=0.4, label="Receiver Aperture: HIT (Detected)"),
        patches.Patch(facecolor="#64748b", edgecolor="#ef4444", alpha=0.2, label="Receiver Aperture: MISS / Search"),
    ]
    ax_waterfall.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=10)
    ax_waterfall.set_xlabel("Mission Elapsed Time (ms)", fontsize=11, fontweight="semibold")
    ax_waterfall.set_ylabel("Frequency (GHz)", fontsize=11, fontweight="semibold")
    ax_waterfall.set_xlim(0, max_time_ms)
    ax_waterfall.set_ylim(0, 18.0)
    ax_waterfall.grid(True, linestyle="--", alpha=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: Dwell Mode Distribution
    # ──────────────────────────────────────────────────────────────────────────
    ax_mode = fig.add_subplot(gs[1, 0])
    mode_names = list(DWELL_MODES)
    mode_counts = [sum(1 for f in telemetry if f["selected_mode"] == m) for m in range(CANONICAL_N_MODES)]
    mode_colors = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899"]

    bars = ax_mode.bar(mode_names, mode_counts, color=mode_colors, edgecolor="black", alpha=0.85, width=0.55)
    ax_mode.set_title("Cognitive Dwell Duration Allocations", fontsize=12, fontweight="bold")
    ax_mode.set_ylabel("Dwell Count", fontsize=10, fontweight="semibold")
    ax_mode.grid(True, linestyle="--", alpha=0.5)
    for bar, count in zip(bars, mode_counts):
        pct = (count / len(telemetry)) * 100.0 if telemetry else 0.0
        ax_mode.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: Cumulative Interceptions Over Time
    # ──────────────────────────────────────────────────────────────────────────
    ax_cum = fig.add_subplot(gs[1, 1])
    times_ms = [f["timestamp_us"] / 1000.0 for f in telemetry]
    cum_hits = np.cumsum([1 if f["hit"] else 0 for f in telemetry])
    rolling_ir = [h / (i + 1) * 100.0 for i, h in enumerate(cum_hits)]

    ax_cum.plot(times_ms, cum_hits, color="#10b981", linewidth=2.2, label="Cumulative Dwell Hits")
    ax_cum.set_title("Interception Velocity & Track Continuity", fontsize=12, fontweight="bold")
    ax_cum.set_xlabel("Mission Elapsed Time (ms)", fontsize=10, fontweight="semibold")
    ax_cum.set_ylabel("Cumulative Interceptions", fontsize=10, fontweight="semibold", color="#10b981")
    ax_cum.tick_params(axis="y", labelcolor="#10b981")
    ax_cum.grid(True, linestyle="--", alpha=0.5)

    ax_ir = ax_cum.twinx()
    ax_ir.plot(times_ms, rolling_ir, color="#f59e0b", linestyle="--", linewidth=1.8, label="Rolling Hit Rate (%)")
    ax_ir.set_ylabel("Rolling Hit Rate (%)", fontsize=10, fontweight="semibold", color="#f59e0b")
    ax_ir.tick_params(axis="y", labelcolor="#f59e0b")
    ax_ir.set_ylim(0, 100)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: Intercept Latency Distribution (Sub-millisecond Precision)
    # ──────────────────────────────────────────────────────────────────────────
    ax_lat = fig.add_subplot(gs[2, 0])
    latencies = [f["intercept_time_us"] for f in telemetry if f["intercept_time_us"] is not None]
    if latencies:
        ax_lat.hist(latencies, bins=25, color="#8b5cf6", edgecolor="black", alpha=0.8, density=True)
        median_lat = float(np.median(latencies))
        mean_lat = float(np.mean(latencies))
        ax_lat.axvline(median_lat, color="#ef4444", linestyle="--", linewidth=2.0, label=f"Median: {median_lat:.1f} µs")
        ax_lat.axvline(mean_lat, color="#06b6d4", linestyle=":", linewidth=2.0, label=f"Mean: {mean_lat:.1f} µs")
        ax_lat.legend(loc="upper right", fontsize=9)
    ax_lat.set_title("Intercept Arrival Latency (Within Dwell)", fontsize=12, fontweight="bold")
    ax_lat.set_xlabel("Pulse Arrival Relative to Dwell Start (µs)", fontsize=10, fontweight="semibold")
    ax_lat.set_ylabel("Probability Density", fontsize=10, fontweight="semibold")
    ax_lat.grid(True, linestyle="--", alpha=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 5: Spectrum Exploration & Distinct Band Visit Distribution
    # ──────────────────────────────────────────────────────────────────────────
    ax_bands = fig.add_subplot(gs[2, 1])
    band_counts = np.zeros(CANONICAL_N_BANDS, dtype=int)
    for f in telemetry:
        band_counts[f["selected_band"]] += 1
    bands = np.arange(CANONICAL_N_BANDS)

    ax_bands.bar(bands, band_counts, color="#3b82f6", edgecolor="black", alpha=0.8)
    ax_bands.set_title("Full Spectrum Activity & Search Balance (36 Bands)", fontsize=12, fontweight="bold")
    ax_bands.set_xlabel("Band Index (0 - 35, 500 MHz each)", fontsize=10, fontweight="semibold")
    ax_bands.set_ylabel("Visits", fontsize=10, fontweight="semibold")
    ax_bands.set_xlim(-0.5, 35.5)
    ax_bands.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=250)
    plt.close()
    logger.info("Operational Telemetry Dashboard generated successfully at %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Operational Telemetry Demonstration Dashboard")
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=BASE_DIR / "reports" / "operational_validation" / "telemetry_config_29.json",
        help="Path to telemetry JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "reports" / "operational_validation" / "operational_telemetry_dashboard_config_29.png",
        help="Path to save output dashboard PNG",
    )
    parser.add_argument("--max_time_ms", type=float, default=60.0, help="Waterfall visualization time window in ms")
    args = parser.parse_args()

    # Find corresponding scenario h5 file
    train_cfg_path = BASE_DIR / "configs" / "training_config.yaml"
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    data_dir = resolve_tsrd_root(None, train_cfg)
    scenario_file = data_dir / "stare" / "val_stare" / "config_29.h5"
    if not scenario_file.exists():
        scenario_file = data_dir / "config_29.h5"

    generate_demonstration_dashboard(
        telemetry_path=args.telemetry,
        scenario_h5_path=scenario_file,
        output_path=args.output,
        max_time_ms=args.max_time_ms,
    )

    # Also generate for config_117
    telemetry_117 = BASE_DIR / "reports" / "operational_validation" / "telemetry_config_117.json"
    output_117 = BASE_DIR / "reports" / "operational_validation" / "operational_telemetry_dashboard_config_117.png"
    scenario_117 = data_dir / "stare" / "val_stare" / "config_117.h5"
    if not scenario_117.exists():
        scenario_117 = data_dir / "config_117.h5"

    if telemetry_117.exists() and scenario_117.exists():
        generate_demonstration_dashboard(
            telemetry_path=telemetry_117,
            scenario_h5_path=scenario_117,
            output_path=output_117,
            max_time_ms=args.max_time_ms,
        )


if __name__ == "__main__":
    main()

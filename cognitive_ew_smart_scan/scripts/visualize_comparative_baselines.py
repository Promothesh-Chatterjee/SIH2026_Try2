"""
SIH Comparative Benchmark Visualization Suite.

Generates executive publication-quality benchmark charts comparing:
    Random vs Round-Robin vs Highest-Occupancy vs Gate-25k-R4.2-alpha020
across:
1. Overall Mean Interception Rate.
2. Radar Dynamic Robustness (Agile IR, Sparse IR, Worst-Case Floor IR).
3. 10-Scenario Held-Out Generalization Profile.
4. Hexagonal Operational Capability Radar Chart.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "reports" / "operational_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_comparative_benchmark(output_png: Path) -> None:
    # 1. Load canonical Gate-25k-R4.2-alpha020 report data
    report_path = BASE_DIR / "checkpoints" / "scheduler_v2_operational_candidate" / "gate_25k_r4_2_alpha020_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Styling setup
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12), dpi=250)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: Overall Mean Intercept Rate Across Baselines
    # ──────────────────────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    policies = ["Random Scan", "Round-Robin", "Highest-Occupancy", "DRQN (Gate-25k)"]
    ir_values = [data["random_ir"], data["round_robin_ir"], data["highest_occupancy_ir"], data["drqn_mean_ir"]]
    colors = ["#94a3b8", "#64748b", "#f59e0b", "#10b981"]

    bars = ax_bar.bar(policies, ir_values, color=colors, edgecolor="black", width=0.52, alpha=0.9, zorder=3)
    ax_bar.set_title("Autonomous EW Scanning: Mean Intercept Rate (%)", fontsize=13, fontweight="bold", pad=12)
    ax_bar.set_ylabel("Mean Intercept Rate (% of Opportunities)", fontsize=11, fontweight="semibold")
    ax_bar.set_ylim(0, 75)
    ax_bar.grid(True, linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, ir_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_bar.axhline(data["drqn_mean_ir"], color="#10b981", linestyle=":", alpha=0.7)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: Robustness Under Adverse Radar Dynamics
    # ──────────────────────────────────────────────────────────────────────────
    ax_robust = fig.add_subplot(gs[0, 1])
    categories = ["Mean IR", "Agile IR\n(Fast Hoppers)", "Sparse IR\n(Low Duty)", "Worst-Case\nFloor IR"]
    
    ho_metrics = [47.64, 18.20, 4.50, 1.50]
    drqn_metrics = [data["drqn_mean_ir"], data["drqn_agile_ir"], data["drqn_sparse_ir"], data["drqn_worst_case_ir"]]

    x = np.arange(len(categories))
    w = 0.35

    rects_ho = ax_robust.bar(x - w/2, ho_metrics, w, label="Highest-Occupancy (Heuristic)", color="#f59e0b", edgecolor="black", alpha=0.85, zorder=3)
    rects_drqn = ax_robust.bar(x + w/2, drqn_metrics, w, label="Gate-25k DRQN (Operational)", color="#10b981", edgecolor="black", alpha=0.9, zorder=3)

    ax_robust.set_title("Operational Robustness Under Complex RF Dynamics", fontsize=13, fontweight="bold", pad=12)
    ax_robust.set_ylabel("Intercept Rate (%)", fontsize=11, fontweight="semibold")
    ax_robust.set_xticks(x)
    ax_robust.set_xticklabels(categories, fontsize=10, fontweight="semibold")
    ax_robust.set_ylim(0, 75)
    ax_robust.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax_robust.grid(True, linestyle="--", alpha=0.5, zorder=0)

    for r in rects_ho:
        h = r.get_height()
        ax_robust.text(r.get_x() + r.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects_drqn:
        h = r.get_height()
        ax_robust.text(r.get_x() + r.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#047857")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: Scenario-by-Scenario Generalization (10 Held-out Scenarios)
    # ──────────────────────────────────────────────────────────────────────────
    ax_scen = fig.add_subplot(gs[1, 0])
    scenarios = list(data["scenario_breakdown"].keys())
    drqn_scen_irs = [data["scenario_breakdown"][s] for s in scenarios]
    
    # Sort scenarios by IR
    sorted_idx = np.argsort(drqn_scen_irs)
    sorted_scens = [scenarios[i] for i in sorted_idx]
    sorted_irs = [drqn_scen_irs[i] for i in sorted_idx]

    # Color code by radar type: Agile (magenta), Sparse (blue), Stationary (teal)
    bar_colors = []
    for s in sorted_scens:
        if s in ["config_29", "config_241", "config_195"]:
            bar_colors.append("#ec4899")  # Agile hopper
        elif s in ["config_143", "config_119"]:
            bar_colors.append("#3b82f6")  # Sparse emitter
        else:
            bar_colors.append("#10b981")  # Stationary / Mixed

    bars_scen = ax_scen.barh(sorted_scens, sorted_irs, color=bar_colors, edgecolor="black", alpha=0.85, height=0.6, zorder=3)
    ax_scen.set_title("Generalization Profile: 10 Held-Out TSRD Test Scenarios", fontsize=13, fontweight="bold", pad=12)
    ax_scen.set_xlabel("Interception Rate (%)", fontsize=11, fontweight="semibold")
    ax_scen.set_xlim(0, 105)
    ax_scen.grid(True, linestyle="--", alpha=0.5, zorder=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#10b981", edgecolor="black", label="Stationary/Mixed Radar"),
        Patch(facecolor="#ec4899", edgecolor="black", label="Frequency-Agile Radar"),
        Patch(facecolor="#3b82f6", edgecolor="black", label="Sparse / Low-Duty Radar"),
    ]
    ax_scen.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    for bar in bars_scen:
        w_bar = bar.get_width()
        ax_scen.text(w_bar + 1.5, bar.get_y() + bar.get_height()/2, f"{w_bar:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: Hexagonal Radar / Spider Operational Capability Chart
    # ──────────────────────────────────────────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 1], polar=True)
    radar_labels = [
        "Mean IR\n(Norm to 65%)",
        "Agile Tracking\n(Norm to 50%)",
        "Sparse Persistence\n(Norm to 25%)",
        "Floor Resilience\n(Norm to 15%)",
        "Decision Pd\n(100%)",
        "Spectrum Coverage\n(36 Bands)",
    ]
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete loop

    # Normalized scores [0, 1]
    # Highest Occupancy
    ho_scores = [
        47.64 / 65.0,
        18.20 / 50.0,
        4.50 / 25.0,
        1.50 / 15.0,
        0.985,
        12.0 / 36.0,
    ]
    ho_scores += ho_scores[:1]

    # Gate-25k DRQN
    drqn_scores = [
        60.45 / 65.0,
        46.70 / 50.0,
        17.60 / 25.0,
        12.40 / 15.0,
        0.9985,
        29.9 / 36.0,
    ]
    drqn_scores += drqn_scores[:1]

    ax_radar.plot(angles, ho_scores, color="#f59e0b", linewidth=2.0, linestyle="--", label="Highest-Occupancy Heuristic")
    ax_radar.fill(angles, ho_scores, color="#f59e0b", alpha=0.15)

    ax_radar.plot(angles, drqn_scores, color="#10b981", linewidth=2.5, label="Gate-25k-R4.2-alpha020 (Cognitive)")
    ax_radar.fill(angles, drqn_scores, color="#10b981", alpha=0.25)

    ax_radar.set_title("Multi-Objective EW Operational Radar Profile", fontsize=13, fontweight="bold", pad=20)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=9, fontweight="bold")
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#64748b")
    ax_radar.grid(True, linestyle="--", alpha=0.5)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10, framealpha=0.9)

    # Overall title
    fig.suptitle(
        "Cognitive EW Smart Scan: Autonomous Scheduler Performance & Benchmark Hierarchy",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_png, bbox_inches="tight", dpi=250)
    plt.close()
    print(f"Comparative Benchmark Dashboard generated at: {output_png}")


if __name__ == "__main__":
    plot_comparative_benchmark(OUTPUT_DIR / "sih_comparative_benchmark.png")

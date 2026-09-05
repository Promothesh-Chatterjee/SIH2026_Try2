"""Full evaluation pipeline: deinterleaving + scheduler over test pulse trains.

Produces results.csv, aggregate_metrics.json, experiment_metadata.json,
dataset_fingerprint.json, roc_curve.pdf, deinterleaving_performance.pdf,
controller_comparison.pdf and formatted summary table.
"""

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..models.baseline_schedulers import (
    HighestOccupancyScheduler,
    HighestUncertaintyScheduler,
    RoundRobinScheduler,
)
from ..models.baseline_suite import FixedPeriodicScan, RevisitHeuristic, SequentialSweep
from ..models.random_scheduler import RandomScheduler

logger = logging.getLogger(__name__)

ALL_BASELINE_NAMES: tuple[str, ...] = (
    "sequential_sweep",
    "round_robin",
    "random",
    "fixed_periodic_scan",
    "highest_occupancy",
    "highest_uncertainty",
    "revisit_heuristic",
)


def _build_baseline(baseline: str, n_bands: int, n_modes: int | None = None, seed: int = 42):
    """Construct a comparison scheduler for the given baseline name."""
    name = baseline.lower().strip()
    baseline_modes = 1 if n_modes is None else int(n_modes)
    if name == "random":
        return RandomScheduler(n_bands=n_bands, n_modes=baseline_modes, seed=seed)
    if name == "round_robin":
        return RoundRobinScheduler(n_bands=n_bands, n_modes=baseline_modes)
    if name == "sequential_sweep":
        return SequentialSweep(n_bands=n_bands, n_modes=baseline_modes)
    if name == "fixed_periodic_scan":
        return FixedPeriodicScan(n_bands=n_bands, n_modes=baseline_modes)
    if name == "revisit_heuristic":
        return RevisitHeuristic(n_bands=n_bands, n_modes=baseline_modes)
    if name == "highest_occupancy":
        return HighestOccupancyScheduler(n_bands=n_bands, n_modes=baseline_modes)
    if name == "highest_uncertainty":
        return HighestUncertaintyScheduler(n_bands=n_bands, n_modes=baseline_modes)
    if name == "none":
        return None
    raise ValueError(f"Unknown baseline '{baseline}'")


def _raw_pulse_count(path: str | Path) -> int:
    """Pulse count from the H5 header only (never loads the data)."""
    try:
        import h5py
        with h5py.File(str(path), "r") as handle:
            if "data" not in handle:
                return -1
            return int(handle["data"].shape[0])
    except Exception:
        return -1


def _compute_dataset_fingerprint(files: list[Path], n_empty: int, n_unusable: int) -> dict:
    """Compute deterministic dataset fingerprint metadata."""
    manifest = []
    total_pulses = 0
    for f in files:
        size = f.stat().st_size if f.exists() else 0
        pulses = _raw_pulse_count(f)
        if pulses > 0:
            total_pulses += pulses
        h = hashlib.sha256()
        try:
            with open(f, "rb") as fh:
                h.update(fh.read(1024 * 1024))
            file_hash = h.hexdigest()[:16]
        except Exception:
            file_hash = "unreadable"
        manifest.append({
            "file_name": f.name,
            "file_path": str(f),
            "size_bytes": size,
            "pulse_count": pulses,
            "hash_sha256": file_hash,
        })
    return {
        "n_files_discovered": len(files),
        "n_evaluated_files": max(0, len(files) - n_empty - n_unusable),
        "n_empty_scenarios": n_empty,
        "n_unusable_scenarios": n_unusable,
        "total_pulses": total_pulses,
        "files_manifest": manifest,
    }


def _get_experiment_metadata(
    seed: int,
    mode: str,
    config_path: Path,
    test_dir: Path,
    output_dir: Path,
    norm_stats_path: Path | None,
    deinterleaver_ckpt: str | Path | None,
    scheduler_ckpt: str | Path | None,
    evaluated_controllers: list[str],
) -> dict:
    """Generate provenance metadata dict."""
    import torch
    try:
        from ..telemetry.run_manager import get_git_revision
        git_hash = get_git_revision()
    except Exception:
        git_hash = "unknown"

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_revision": git_hash,
        "seed": seed,
        "mode": mode,
        "config_path": str(config_path),
        "test_dir": str(test_dir),
        "output_dir": str(output_dir),
        "norm_stats_path": str(norm_stats_path) if norm_stats_path else None,
        "deinterleaver_ckpt": str(deinterleaver_ckpt) if deinterleaver_ckpt else None,
        "scheduler_ckpt": str(scheduler_ckpt) if scheduler_ckpt else None,
        "evaluated_controllers": evaluated_controllers,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }


def _plot_controller_comparison(
    controllers_summary: dict[str, dict[str, float]], save_path: Path
) -> None:
    """Side-by-side bar chart comparing all controllers across key metrics."""
    names = list(controllers_summary.keys())
    metrics_to_plot = [
        ("Pd", "Pd (Probability of Detection)"),
        ("Pfa", "Pfa (Probability of False Alarm)"),
        ("avg_intercept_rate", "Avg Intercept Rate"),
        ("avg_reward", "Avg Reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes = axes.flatten()

    for i, (m_key, m_title) in enumerate(metrics_to_plot):
        vals = [controllers_summary[name].get(m_key, 0.0) for name in names]
        colors = ["#1f77b4" if "learned" in name.lower() or "moe" in name.lower() else "#7f8c8d" for name in names]
        bars = axes[i].bar(names, vals, color=colors, edgecolor="black", alpha=0.85)
        axes[i].set_title(m_title, fontsize=11, fontweight="bold")
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                axes[i].annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.suptitle("SmartScan Benchmark Controller Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.savefig(save_path.with_suffix(".png"), format="png")
    plt.close()


def run_full_evaluation(
    deinterleaver_ckpt: str | Path | None,
    scheduler_ckpt: str | Path | None,
    config_path: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    mode: str = "scan",
    baseline: str = "all",
    norm_stats: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """Run full reproducible evaluation over test pulse trains."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
    from ..environment.scenario_generator import load_h5_records
    from ..evaluation.metrics import FiguresOfMerit
    from ..telemetry.publisher import TelemetryPublisher
    from ..telemetry.run_manager import RunManager

    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    drqn_cfg = model_cfg.get("drqn_scheduler", {})
    reward_cfg = model_cfg.get("reward", {})
    env_cfg = model_cfg.get("environment", {})

    from ..preprocessing.normalise import load_normalization_stats

    norm_stats_path: Path | None = None
    if norm_stats:
        norm_stats_path = Path(norm_stats)
    if norm_stats_path is None or not norm_stats_path.exists():
        if deinterleaver_ckpt:
            cand = Path(deinterleaver_ckpt).parent / "normalization_stats.json"
            if cand.exists():
                norm_stats_path = cand
    if norm_stats_path is None or not norm_stats_path.exists():
        cand = Path("configs/normalization_stats.json")
        if cand.exists():
            norm_stats_path = cand

    train_stats = None
    if deinterleaver_ckpt:
        if norm_stats_path is None or not norm_stats_path.exists():
            raise FileNotFoundError(
                "Deinterleaver evaluation requires train-fitted normalization stats "
                "but none were found. Fit them on TRAIN data first."
            )
        train_stats = load_normalization_stats(norm_stats_path)
        logger.info("Loaded train normalization stats from %s", norm_stats_path)

    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(test_dir.glob("*.h5"))
    if not files and test_dir.exists():
        files = sorted(test_dir.rglob("*.h5"))
    if test_dir.is_file():
        files = [test_dir]
    if not files:
        for cand in [Path("data/test") / mode, Path("data/scan/test"), Path("data/stare/test"), Path("data/test")]:
            if cand.exists():
                files = sorted(cand.rglob("*.h5"))
                if files:
                    logger.info("Found test files via fallback %s", cand)
                    break
    if not files:
        raise FileNotFoundError(f"No test .h5 files in {test_dir} (mode={mode})")
    files = files[:250]
    logger.info("Evaluating %d test files (mode=%s, seed=%d)", len(files), mode, seed)

    # Load Deinterleaver
    deinterleaver = None
    if deinterleaver_ckpt and Path(deinterleaver_ckpt).exists():
        try:
            from ..models.deinterleaver import PDWTransformerEncoder
            ckpt_path = Path(deinterleaver_ckpt)
            d_cfg = model_cfg.get("deinterleaver", {})
            deinterleaver = PDWTransformerEncoder(
                pdw_dim=d_cfg.get("pdw_dim", 6),
                d_model=d_cfg.get("d_model", 128),
                nhead=d_cfg.get("nhead", 8),
                num_layers=d_cfg.get("num_layers", 4),
                dim_feedforward=d_cfg.get("dim_feedforward", 512),
                dropout=d_cfg.get("dropout", 0.1),
                embed_dim=d_cfg.get("embed_dim", 64),
            )
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            deinterleaver.load_state_dict(state, strict=False)
            deinterleaver.eval()
            logger.info("Loaded deinterleaver %s", ckpt_path)
        except Exception as exc:
            logger.warning("Failed to load deinterleaver %s: %s", deinterleaver_ckpt, exc)
            deinterleaver = None
    else:
        logger.warning("Deinterleaver ckpt not found: %s — skipping deinterleave metrics", deinterleaver_ckpt)

    # Load Learned Scheduler
    scheduler = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scheduler_ckpt and Path(scheduler_ckpt).exists():
        try:
            from ..models.drqn_scheduler import DRQNScheduler
            from ..models.smartscan_moe import SmartScanMoE as MoE

            band_features = int(env_cfg.get("band_features", 10))
            n_bands = int(drqn_cfg.get("n_bands", env_cfg.get("n_bands", 36)))
            n_modes = int(drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5)))
            n_actions = int(drqn_cfg.get("n_actions", n_bands * n_modes))
            obs_dim = int(drqn_cfg.get("obs_dim", n_bands * band_features))

            scheduler_drqn = DRQNScheduler(
                obs_dim=obs_dim,
                n_bands=n_bands,
                n_actions=n_actions,
                lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
                lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
            )
            state = torch.load(str(scheduler_ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            scheduler_drqn.load_state_dict(state, strict=False)
            scheduler_drqn.eval()

            moe_cfg = model_cfg.get("smartscan_moe", {})
            scheduler = MoE(
                scheduler_drqn,
                {
                    **moe_cfg,
                    "n_bands": n_bands,
                    "n_modes": n_modes,
                    "n_actions": n_actions,
                    "device": str(device),
                },
            )
            logger.info("Loaded learned scheduler %s", scheduler_ckpt)
        except Exception as exc:
            logger.warning("Failed to load scheduler %s: %s", scheduler_ckpt, exc)
            scheduler = None

    env_config = {**drqn_cfg, **reward_cfg, "n_bands": drqn_cfg.get("n_bands", 36), "n_modes": drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5))}
    env_config["n_actions"] = int(env_config["n_bands"]) * int(env_config["n_modes"])

    try:
        from turing_deinterleaving_challenge import PulseTrain
        has_pt = True
    except ImportError:
        has_pt = False

    # Resolve baseline list
    n_bands = int(drqn_cfg.get("n_bands", 36))
    n_modes = int(env_config.get("n_modes", 5))

    baseline_names_to_run = []
    b_arg = str(baseline).lower().strip()
    if b_arg == "all":
        baseline_names_to_run = list(ALL_BASELINE_NAMES)
    elif b_arg != "none":
        for b_sub in b_arg.split(","):
            b_clean = b_sub.strip()
            if b_clean and b_clean != "none":
                baseline_names_to_run.append(b_clean)

    baseline_controllers = {}
    baseline_foms = {}
    for b_name in baseline_names_to_run:
        ctrl = _build_baseline(b_name, n_bands=n_bands, n_modes=n_modes, seed=seed)
        if ctrl is not None:
            baseline_controllers[b_name] = ctrl
            baseline_foms[b_name] = FiguresOfMerit(n_bands=n_bands)

    global_fom = FiguresOfMerit(n_bands=n_bands)
    rows: list[dict] = []
    deinter_metrics: list[dict] = []

    run = RunManager(
        root="runs",
        config={
            "mode": mode,
            "n_files": len(files),
            "scheduler": str(scheduler_ckpt),
            "deinterleaver": str(deinterleaver_ckpt),
            "baseline": baseline,
            "seed": seed,
        },
        extras={"split": "test", "mode": mode, "device": str(device)},
    )
    run.write_git_revision()
    telemetry = TelemetryPublisher(run=run)

    n_empty_scenarios = 0
    n_skipped_scheduler = 0

    for idx, fpath in enumerate(files):
        per_file: dict = {"file": str(fpath), "mode": mode}
        raw_rows = _raw_pulse_count(fpath)

        # Deinterleaving evaluation
        v_measure = ami = ari = homogeneity = completeness = float("nan")
        pairwise_mcc = pairwise_f1 = float("nan")
        latency_ms = float("nan")
        n_clusters = noise_fraction = float("nan")

        if has_pt:
            try:
                pt = PulseTrain.load(str(fpath))
                pdws = pt.data
                labels_true = pt.labels
                if pdws is not None and len(pdws) > 0 and deinterleaver is not None:
                    from ..models.deinterleaver import windowed_cluster_deinterleave
                    from ..preprocessing.normalise import normalise_pdws

                    d_cfg = model_cfg.get("deinterleaver", {})
                    pdws_norm, _ = normalise_pdws(pdws, train_stats)
                    t0 = time.perf_counter()
                    res = windowed_cluster_deinterleave(
                        deinterleaver,
                        pdws_norm,
                        window_size=int(d_cfg.get("window_size", 2048)),
                        stride=int(d_cfg.get("window_stride", 1024)),
                        device=str(device),
                        min_cluster_size=int(d_cfg.get("min_cluster_size", 10)),
                        min_samples=int(d_cfg.get("min_samples", 5)),
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(pdws))
                    labels_pred = res["labels"]

                    from ..evaluation.metrics import deinterleaver_train_metrics
                    m = deinterleaver_train_metrics(labels_true, labels_pred, with_pairwise=True)
                    v_measure = m.get("v_measure", float("nan"))
                    ami = m.get("ami", float("nan"))
                    ari = m.get("ari", float("nan"))
                    homogeneity = m.get("homogeneity", float("nan"))
                    completeness = m.get("completeness", float("nan"))
                    pairwise_mcc = m.get("pairwise_mcc", float("nan"))
                    pairwise_f1 = m.get("pairwise_f1", float("nan"))
                    n_clusters = m.get("n_clusters_predicted", float("nan"))
                    noise_fraction = m.get("noise_fraction", float("nan"))
            except Exception as exc:
                logger.warning("Failed to evaluate deinterleave %s: %s", fpath, exc)

        per_file.update({
            "v_measure": v_measure,
            "ami": ami,
            "ari": ari,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "pairwise_mcc": pairwise_mcc,
            "pairwise_f1": pairwise_f1,
            "latency_ms_per_pulse": latency_ms,
            "n_clusters_predicted": n_clusters,
            "noise_fraction": noise_fraction,
        })
        deinter_metrics.append({
            "v_measure": v_measure, "ami": ami, "ari": ari,
            "homogeneity": homogeneity, "completeness": completeness,
            "pairwise_mcc": pairwise_mcc, "pairwise_f1": pairwise_f1,
        })

        # Scheduler Evaluation on identical scenario
        scheduled = False
        try:
            records = load_h5_records(
                Path(fpath),
                freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
                freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
                max_pulses=int(env_cfg.get("max_pulses", 50000)),
            )
            if raw_rows == 0 or not records:
                per_file["empty_scenario"] = raw_rows == 0
                per_file["skipped_reason"] = (
                    "empty_scenario_zero_pulses" if raw_rows == 0
                    else "no_records_after_filter"
                )
                per_file.update({"sched_Pd": float("nan"), "sched_Pfa": float("nan"), "sched_avg_reward": float("nan")})
                if raw_rows == 0:
                    n_empty_scenarios += 1
                else:
                    n_skipped_scheduler += 1
            else:
                scheduled = True
        except Exception as exc:
            logger.warning("Failed loading scenario for %s: %s", fpath, exc)

        if scheduled:
            # 1. Learned Scheduler episode
            if scheduler is not None:
                env = CognitiveRFScanEnv({**env_config, **env_cfg}, records=records, seed=seed)
                obs, _ = env.reset()
                done = False
                steps = 0
                hidden = None
                if hasattr(scheduler, "reset"):
                    try:
                        scheduler.reset()
                    except Exception:
                        pass
                while not done and steps < 5000:
                    if hasattr(scheduler, "set_periodic_urgency_vector") and hasattr(env, "belief"):
                        scheduler.set_periodic_urgency_vector(
                            np.asarray(getattr(env.belief, "periodic_urgency", np.zeros(36)), dtype=np.float32).reshape(-1)
                        )
                    if hasattr(scheduler, "select_action"):
                        action, hidden, attr = scheduler.select_action(obs, hidden)
                        mode_ctx = {"action_score": float(attr.get("action_score", 1.0)), "reason": str(attr.get("reason", "mode_preset"))}
                        obs, reward, terminated, truncated, info = env.step(action, mode_context=mode_ctx)
                    else:
                        bands, hidden, _ = scheduler.select_bands(obs, hidden)
                        action = int(bands[0])
                        obs, reward, terminated, truncated, info = env.step(action)

                    if hasattr(scheduler, "update"):
                        try:
                            scheduler.update(action)
                        except Exception:
                            pass
                    done = bool(terminated or truncated)
                    steps += 1
                    global_fom.update(
                        int(info["band_chosen"]) if "band_chosen" in info else action,
                        info["ground_truth_active"],
                        info["hit"],
                        float(info.get("intercept_time_error_us", float("nan"))),
                        float(reward),
                    )
                fom = env.get_fom()
                per_file.update({f"sched_{k}": v for k, v in fom.items()})

            # 2. Benchmark Baseline Episodes (Identical records and seed)
            for b_name, b_ctrl in baseline_controllers.items():
                benv = CognitiveRFScanEnv({**env_config, **env_cfg}, records=records, seed=seed)
                bobs, _ = benv.reset()
                if hasattr(b_ctrl, "reset"):
                    try:
                        b_ctrl.reset()
                    except Exception:
                        pass
                bdone = False
                bsteps = 0
                while not bdone and bsteps < 5000:
                    if hasattr(b_ctrl, "set_periodic_urgency_vector") and hasattr(benv, "belief"):
                        try:
                            urg = np.asarray(getattr(benv.belief, "periodic_urgency", np.zeros(n_bands)), dtype=np.float32)
                            b_ctrl.set_periodic_urgency_vector(urg)
                        except Exception:
                            pass
                    try:
                        if hasattr(b_ctrl, "act"):
                            action, _ = b_ctrl.act(bobs)
                        else:
                            action = int(b_ctrl.step(bobs))
                    except Exception:
                        action = int(benv.action_space.sample())

                    bobs, reward, terminated, truncated, info = benv.step(int(action))
                    if hasattr(b_ctrl, "update"):
                        try:
                            b_ctrl.update(int(action))
                        except Exception:
                            pass
                    bdone = bool(terminated or truncated)
                    bsteps += 1
                    baseline_foms[b_name].update(
                        int(info["band_chosen"]) if "band_chosen" in info else action,
                        info["ground_truth_active"],
                        info["hit"],
                        float(info.get("intercept_time_error_us", float("nan"))),
                        float(reward),
                    )
                bfom = benv.get_fom()
                per_file.update({f"bl_{b_name}_{k}": v for k, v in bfom.items()})

        rows.append(per_file)
        if (idx + 1) % 10 == 0:
            logger.info("Evaluated %d/%d files", idx + 1, len(files))

    # Build DataFrame & Save CSV
    df = pd.DataFrame(rows)
    results_csv = output_dir / "results.csv"
    df.to_csv(results_csv, index=False)
    logger.info("Saved %s (%d rows)", results_csv, len(df))

    # Aggregates & Summaries
    from ..evaluation.metrics import aggregate_deinterleaver_metrics
    deint_agg = aggregate_deinterleaver_metrics(deinter_metrics)

    agg: dict = {key: float(val) for key, val in deint_agg.items()}
    for metric in ["v_measure", "ami", "ari", "homogeneity", "completeness", "pairwise_mcc", "pairwise_f1"]:
        agg[f"mean_{metric}"] = agg.get(f"{metric}_mean", float("nan"))
        agg[f"median_{metric}"] = agg.get(f"{metric}_median", float("nan"))

    if scheduler is not None:
        agg.update({f"sched_{k}": float(v) for k, v in global_fom.summary().items()})

    controllers_summary: dict[str, dict[str, float]] = {}
    if scheduler is not None:
        controllers_summary["Learned (SmartScanMoE)"] = global_fom.summary()

    for b_name, b_fom in baseline_foms.items():
        summary_dict = b_fom.summary()
        controllers_summary[b_name] = summary_dict
        for k, v in summary_dict.items():
            agg[f"bl_{b_name}_{k}"] = float(v)

    agg["n_files"] = len(files)
    agg["n_empty_scenarios"] = n_empty_scenarios
    agg["n_skipped_scheduler"] = n_skipped_scheduler
    agg["n_evaluated_files"] = max(0, len(files) - n_empty_scenarios - n_skipped_scheduler)
    agg["mode"] = mode
    agg["seed"] = seed

    # Save JSON files
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved aggregate_metrics.json")

    exp_meta = _get_experiment_metadata(
        seed=seed, mode=mode, config_path=Path(config_path),
        test_dir=test_dir, output_dir=output_dir, norm_stats_path=norm_stats_path,
        deinterleaver_ckpt=deinterleaver_ckpt, scheduler_ckpt=scheduler_ckpt,
        evaluated_controllers=list(controllers_summary.keys()),
    )
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(exp_meta, f, indent=2)
    logger.info("Saved experiment_metadata.json")

    ds_fp = _compute_dataset_fingerprint(files, n_empty_scenarios, n_skipped_scheduler)
    with open(output_dir / "dataset_fingerprint.json", "w") as f:
        json.dump(ds_fp, f, indent=2)
    logger.info("Saved dataset_fingerprint.json")

    # Generate Plots
    try:
        global_fom.plot_roc_curve(output_dir / "roc_curve.pdf")
    except Exception as exc:
        logger.warning("ROC plot failed: %s", exc)

    try:
        plt.figure(figsize=(6, 4), dpi=300)
        vals = pd.to_numeric(df["v_measure"], errors="coerce").dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=20, color="steelblue", edgecolor="white", alpha=0.9)
            plt.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.3f}")
            plt.xlabel("V-measure")
            plt.ylabel("Count")
            plt.title(f"Deinterleaving Performance ({mode})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "deinterleaving_performance.pdf", format="pdf")
            plt.close()
    except Exception as exc:
        logger.warning("Deinterleaving plot failed: %s", exc)

    try:
        if controllers_summary:
            _plot_controller_comparison(controllers_summary, output_dir / "controller_comparison.pdf")
            logger.info("Saved controller_comparison.pdf & png")
    except Exception as exc:
        logger.warning("Controller comparison plot failed: %s", exc)

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f" Evaluation Benchmark Summary — mode={mode} — {len(files)} files (seed={seed})")
    print("=" * 90)
    print(f" {'Controller':<24} {'Pd':<10} {'Pfa':<10} {'Sensitivity':<12} {'Int. Rate':<12} {'Time Err(us)':<14} {'Avg Reward':<12}")
    print("-" * 90)

    for ctrl_name, s_dict in controllers_summary.items():
        pd_val = s_dict.get("Pd", float("nan"))
        pfa_val = s_dict.get("Pfa", float("nan"))
        sens_val = s_dict.get("sensitivity", pd_val)
        rate_val = s_dict.get("avg_intercept_rate", float("nan"))
        err_val = s_dict.get("avg_intercept_time_error_us", float("nan"))
        rw_val = s_dict.get("avg_reward", float("nan"))

        print(
            f" {ctrl_name:<24} "
            f"{pd_val:<10.4f} "
            f"{pfa_val:<10.4f} "
            f"{sens_val:<12.4f} "
            f"{rate_val:<12.4f} "
            f"{err_val:<14.4f} "
            f"{rw_val:<12.4f}"
        )

    print("-" * 90)
    if deinterleaver is not None:
        vm = agg.get("mean_v_measure", float("nan"))
        ami_val = agg.get("mean_ami", float("nan"))
        ari_val = agg.get("mean_ari", float("nan"))
        mcc_val = agg.get("mean_pairwise_mcc", float("nan"))
        print(f" Deinterleaver: V-measure={vm:.4f}  AMI={ami_val:.4f}  ARI={ari_val:.4f}  Pairwise-MCC={mcc_val:.4f}")
    if n_empty_scenarios or n_skipped_scheduler:
        print(f" File Accounting: {len(files)} total, {agg['n_evaluated_files']} evaluated, {n_empty_scenarios} empty, {n_skipped_scheduler} unusable")
    print("=" * 90)
    print(f"Results CSV: {results_csv}")
    print(f"Output directory: {output_dir}\n")

    return {"dataframe": df, "aggregate": agg, "output_dir": output_dir}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Full evaluation & benchmark pipeline")
    parser.add_argument("--deinterleaver-ckpt", type=str, default="checkpoints/deinterleaver/best.pt")
    parser.add_argument("--scheduler-ckpt", type=str, default="checkpoints/scheduler/best.pt")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--mode", type=str, choices=["scan", "stare"], default="scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm-stats", type=str, default=None,
        help="Path to train-fitted normalization stats JSON (leakage-free test eval).")
    parser.add_argument(
        "--baseline",
        type=str,
        default="all",
        help="Baseline(s) to evaluate: 'all', 'none', or comma-separated list.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_full_evaluation(
        deinterleaver_ckpt=args.deinterleaver_ckpt,
        scheduler_ckpt=args.scheduler_ckpt,
        config_path=args.config,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        baseline=args.baseline,
        norm_stats=args.norm_stats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

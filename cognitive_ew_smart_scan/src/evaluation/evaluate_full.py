"""
Full evaluation pipeline: deinterleaving + scheduler over 250 test pulse trains.

Produces results.csv, aggregate_metrics.json, roc_curve.pdf, deinterleaving_performance.pdf
and prints formatted summary table vs baseline/targets.
"""

import argparse
import json
import logging
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


def _build_baseline(baseline: str, n_bands: int, n_modes: int | None = None, seed: int = 42):
    """Construct a comparison scheduler for the given baseline name.

    Args:
        baseline: One of "sequential_sweep", "round_robin", "random",
            "fixed_periodic_scan", "highest_occupancy", "highest_uncertainty",
            "revisit_heuristic"; returns None for "none".
        n_bands: Number of discrete bands.
            n_modes: Number of dwell modes (action = band*n_modes + mode). When
                omitted, preserve the legacy band-only helper behavior.
        seed: Seed for the random baseline.

    Returns:
        A scheduler instance exposing ``step(observation) -> int`` (and
        ``act``), or None if baseline is "none".

    Raises:
        ValueError: If baseline is not a recognised name.
    """
    name = baseline.lower()
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
    """Pulse count from the H5 header only (never loads the data).

    Returns ``-1`` for unreadable/unstructured files (never an empty-scenario
    claim). Zero is a genuine official-TSRD empty scene.
    """
    try:
        import h5py
        with h5py.File(str(path), "r") as handle:
            if "data" not in handle:
                return -1
            return int(handle["data"].shape[0])
    except Exception:
        return -1


def run_full_evaluation(
    deinterleaver_ckpt: str | Path | None,
    scheduler_ckpt: str | Path | None,
    config_path: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    mode: str = "scan",
    baseline: str = "none",
    norm_stats: str | Path | None = None,
) -> dict:
    """Run full evaluation over test pulse trains.

    Args:
        deinterleaver_ckpt: Path to Transformer checkpoint (or None → skip deinterleave).
        scheduler_ckpt: Path to DRQN checkpoint (or None → use random baseline).
        config_path: Path to model_config.yaml.
        test_dir: Directory containing test .h5 files (e.g., data/test/scan).
        output_dir: Output directory for results.
        mode: stare or scan (for reporting).
        baseline: Comparison controller to run alongside the learned scheduler:
            one of "none", "random", "round_robin", "highest_occupancy",
            "highest_uncertainty". When not "none", FoM is also collected for
            this baseline and reported in the summary table as "Baseline".
        norm_stats: Path to train-fitted normalization statistics JSON. When
            provided it is loaded and applied to test PDWs (leakage-free).
            If None, normalization stats are sought next to the deinterleaver
            checkpoint and then in ``configs/normalization_stats.json``.

    Returns:
        Dict with aggregate metrics and per-file DataFrame saved to disk.

    Raises:
        FileNotFoundError: If test_dir has no .h5 files, or normalization stats
            are required for deinterleaving but cannot be located.
    """
    import torch

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

    # Resolve train-fitted normalization statistics (leakage-safe test evaluation).
    from ..preprocessing.normalise import load_normalization_stats

    norm_stats_path: Path | None = None
    if norm_stats:
        norm_stats_path = Path(norm_stats)
    if norm_stats_path is None or not norm_stats_path.exists():
        # Fall back to a stats file stored beside the deinterleaver checkpoint.
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
                "but none were found at "
                f"{norm_stats} / {Path(deinterleaver_ckpt).parent / 'normalization_stats.json'} / "
                "configs/normalization_stats.json. Fit them on TRAIN data first."
            )
        train_stats = load_normalization_stats(norm_stats_path)
        logger.info("Loaded train normalization stats from %s", norm_stats_path)

    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover up to 250 test files
    files = sorted(test_dir.glob("*.h5"))
    if not files and test_dir.exists():
        # Try subdirectories
        files = sorted(test_dir.rglob("*.h5"))
    if test_dir.is_file():
        files = [test_dir]
    if not files:
        # Fallback: data/test/scan, data/scan/test etc.
        for cand in [Path("data/test") / mode, Path("data/scan/test"), Path("data/stare/test"), Path("data/test")]:
            if cand.exists():
                files = sorted(cand.rglob("*.h5"))
                if files:
                    logger.info("Found test files via fallback %s", cand)
                    break
    if not files:
        raise FileNotFoundError(f"No test .h5 files in {test_dir} (mode={mode})")
    files = files[:250]
    logger.info("Evaluating %d test files (mode=%s)", len(files), mode)

    # Load models if checkpoints exist
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
            # Handle nested state dicts
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

    scheduler = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scheduler_ckpt and Path(scheduler_ckpt).exists():
        try:
            from ..models.drqn_scheduler import DRQNScheduler
            from ..models.smartscan_moe import SmartScanMoE

            # Derive obs dim from the canonical observation contract
            # (n_bands * band_features) rather than a hardcoded literal.
            band_features = int(env_cfg.get("band_features", 10))
            n_bands = int(drqn_cfg.get("n_bands", env_cfg.get("n_bands", 36)))
            n_modes = int(drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5)))
            n_actions = int(drqn_cfg.get("n_actions", n_bands * n_modes))
            obs_dim = int(drqn_cfg.get("obs_dim", n_bands * band_features))

            scheduler = DRQNScheduler(
                obs_dim=obs_dim,
                n_bands=n_bands,
                n_actions=n_actions,
                lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
                lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
            )
            state = torch.load(str(scheduler_ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            scheduler.load_state_dict(state, strict=False)
            scheduler.eval()
            # Wrap in MoE for evaluation
            moe_cfg = model_cfg.get("smartscan_moe", {})
            from ..models.smartscan_moe import SmartScanMoE as MoE

            moe = MoE(
                scheduler,
                {
                    **moe_cfg,
                    "n_bands": n_bands,
                    "n_modes": n_modes,
                    "n_actions": n_actions,
                    "device": str(device),
                },
            )
            scheduler = moe  # use MoE for scheduling
            logger.info("Loaded scheduler %s", scheduler_ckpt)
        except Exception as exc:
            logger.warning("Failed to load scheduler %s: %s", scheduler_ckpt, exc)
            scheduler = None
    else:
        logger.warning("Scheduler ckpt not found: %s — using random baseline", scheduler_ckpt)

    # Env for scheduler metrics (use test dir as data_dir with subset="." workaround)
    # Create a lightweight env that we reset per file via manual pt loading
    # Instead, iterate files and simulate intercepts via RFScanEnv per file
    env_config = {**drqn_cfg, **reward_cfg, "n_bands": drqn_cfg.get("n_bands", 36), "n_modes": drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5))}
    env_config["n_actions"] = int(env_config["n_bands"]) * int(env_config["n_modes"])
    # Patch training_config environment keys if present
    try:
        with open("configs/training_config.yaml") as f:
            train_cfg = yaml.safe_load(f)
            env_config.update(train_cfg.get("environment", {}))
    except Exception:
        pass

    try:
        from turing_deinterleaving_challenge import PulseTrain  # type: ignore
        has_pt = True
    except ImportError:
        has_pt = False

    # For metrics aggregation
    rows: list[dict] = []
    global_fom = FiguresOfMerit()
    deinter_metrics: list[dict] = []
    device = torch.device("cuda" if torch.cuda.is_available() and scheduler is not None else "cpu")

    # Comparison baseline controller (same env seed for an apples-to-apples run).
    n_bands = int(drqn_cfg.get("n_bands", 36))
    n_modes = int(env_config.get("n_modes", 5))
    baseline_controller = _build_baseline(baseline, n_bands=n_bands, n_modes=n_modes)
    baseline_fom = FiguresOfMerit()

    # P0-9: reproducible evaluation run + real telemetry publisher.
    run = RunManager(
        root="runs",
        config={
            "mode": mode,
            "n_files": len(files),
            "scheduler": str(scheduler_ckpt),
            "deinterleaver": str(deinterleaver_ckpt),
            "baseline": baseline,
        },
        extras={"split": "test", "mode": mode, "device": str(device)},
    )
    run.write_git_revision()
    telemetry = TelemetryPublisher(run=run)
    logger.info("Evaluation run %s at %s", run.run_id, run.dir)

    # Explicit empty/unusable scenario accounting (Phase 19): zero-pulse scenes
    # are reported and skipped, never scored or allowed to pollute aggregates.
    n_empty_scenarios = 0
    n_skipped_scheduler = 0

    for idx, fpath in enumerate(files):
        per_file: dict = {"file": str(fpath), "mode": mode}
        raw_rows = _raw_pulse_count(fpath)
        # --- Deinterleaving (leakage-safe + windowed inference) ---
        v_measure = ami = ari = homogeneity = completeness = float("nan")
        pairwise_mcc = pairwise_f1 = float("nan")
        latency_ms = float("nan")
        n_clusters = noise_fraction = float("nan")
        if has_pt:
            try:
                pt = PulseTrain.load(str(fpath))
                pdws = pt.data  # (N,5)
                labels_true = pt.labels
                if pdws is not None and len(pdws) > 0 and deinterleaver is not None:
                    from ..models.deinterleaver import windowed_cluster_deinterleave
                    from ..preprocessing.normalise import normalise_pdws

                    d_cfg = model_cfg.get("deinterleaver", {})
                    window_size = int(d_cfg.get("window_size", 2048))
                    stride = int(d_cfg.get("window_stride", 1024))
                    mcs = int(d_cfg.get("min_cluster_size", 10))
                    ms = int(d_cfg.get("min_samples", 5))
                    pdws_norm, _ = normalise_pdws(pdws, train_stats)
                    t0 = time.perf_counter()
                    res = windowed_cluster_deinterleave(
                        deinterleaver,
                        pdws_norm,
                        window_size=window_size,
                        stride=stride,
                        device=str(device),
                        min_cluster_size=mcs,
                        min_samples=ms,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(pdws))  # ms per pulse avg
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
        per_file.update(
            {
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
            }
        )
        deinter_metrics.append(
            {
                "v_measure": v_measure,
                "ami": ami,
                "ari": ari,
                "homogeneity": homogeneity,
                "completeness": completeness,
                "pairwise_mcc": pairwise_mcc,
                "pairwise_f1": pairwise_f1,
            }
        )

        # --- Scheduler (run one episode per file via CognitiveRFScanEnv) ---
        scheduled = False
        try:
            # Build a cognitive env fed by this single test file's pulses.
            freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
            freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
            record_limit = int(env_cfg.get("max_pulses", 50000))
            records = load_h5_records(
                Path(fpath),
                freq_min_mhz=freq_min,
                freq_max_mhz=freq_max,
                max_pulses=record_limit,
            )
            if raw_rows == 0 or not records:
                # Explicit empty-scenario handling: never evaluate a zero-pulse
                # scene, and never let it corrupt the aggregate metrics.
                per_file["empty_scenario"] = raw_rows == 0
                per_file["skipped_reason"] = (
                    "empty_scenario_zero_pulses" if raw_rows == 0
                    else "no_records_after_filter_clipped_out_of_band"
                )
                per_file.update(
                    {
                        "sched_Pd": float("nan"), "sched_Pfa": float("nan"),
                        "sched_avg_reward": float("nan"),
                    }
                )
                if raw_rows == 0:
                    n_empty_scenarios += 1
                    logger.warning("Skipping empty scenario (0 pulses): %s", fpath)
                else:
                    n_skipped_scheduler += 1
                    logger.warning(
                        "Skipping unusable scenario (%d raw pulses, 0 after filter): %s",
                        raw_rows, fpath,
                    )
            else:
                scheduled = True
                env = CognitiveRFScanEnv({**env_config, **env_cfg}, records=records, seed=42)
                obs, _ = env.reset()
            if scheduled:
                # Achieved controller episode (learned MoE, or random fallback).
                done = False
                steps = 0
                hidden = None
                if scheduler is not None and hasattr(scheduler, "reset"):
                    try:
                        scheduler.reset()
                        if hasattr(scheduler, "eager_agent") and hasattr(scheduler.eager_agent, "hidden"):
                            drqn_inner = scheduler.eager_agent.drqn if hasattr(scheduler.eager_agent, "drqn") else None
                            if drqn_inner is not None:
                                hidden = drqn_inner.init_hidden(1, device)
                                scheduler.eager_agent.hidden = hidden
                    except Exception:
                        pass
                while not done and steps < 5000:
                    if scheduler is not None:
                        try:
                            if hasattr(scheduler, "set_periodic_urgency_vector"):
                                scheduler.set_periodic_urgency_vector(np.asarray(getattr(env, "belief", np.zeros(36)).periodic_urgency, dtype=np.float32).reshape(-1) if hasattr(getattr(env, "belief", None), "periodic_urgency") else np.zeros(36, dtype=np.float32))
                            if hasattr(scheduler, "select_action"):
                                action, hidden, attr = scheduler.select_action(obs, hidden)  # type: ignore
                                mode_ctx = {"action_score": float(attr.get("action_score", 1.0)), "reason": str(attr.get("reason", "mode_preset"))}
                                obs, reward, terminated, truncated, info = env.step(action, mode_context=mode_ctx)
                            else:
                                bands, hidden, _ = scheduler.select_bands(obs, hidden)  # type: ignore
                                action = int(bands[0])
                                obs, reward, terminated, truncated, info = env.step(action)
                        except Exception:
                            action = int(env.action_space.sample())
                            obs, reward, terminated, truncated, info = env.step(action)
                    else:
                        action = int(env.action_space.sample())
                        obs, reward, terminated, truncated, info = env.step(action)
                    if scheduler is not None and hasattr(scheduler, "update"):
                        try:
                            scheduler.update(action)  # type: ignore
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
                telemetry.update(step=idx, file=str(fpath), type="eval_file", pd=float(fom.get("Pd", 0.0)), pfa=float(fom.get("Pfa", 0.0)), avg_reward=float(fom.get("avg_reward", 0.0)))

                # Comparison baseline episode (same file, same env seed if requested).
                if baseline_controller is not None:
                    benv = CognitiveRFScanEnv({**env_config, **env_cfg}, records=records, seed=42)
                    bobs, _ = benv.reset()
                    done = False
                    steps = 0
                    while not done and steps < 5000:
                        try:
                            action = int(baseline_controller.step(bobs))
                        except Exception:
                            action = int(benv.action_space.sample())
                        bobs, reward, terminated, truncated, info = benv.step(action)
                        done = bool(terminated or truncated)
                        steps += 1
                        baseline_fom.update(
                            int(info["band_chosen"]) if "band_chosen" in info else action,
                            info["ground_truth_active"],
                            info["hit"],
                            float(info.get("intercept_time_error_us", float("nan"))),
                            float(reward),
                        )
                    bfom = benv.get_fom()
                    per_file.update({f"bl_sched_{k}": v for k, v in bfom.items()})
        except Exception as exc:
            logger.warning("Scheduler eval failed for %s: %s", fpath, exc)
            per_file.update({"sched_Pd": float("nan"), "sched_Pfa": float("nan")})

        rows.append(per_file)
        if (idx + 1) % 10 == 0:
            logger.info("Evaluated %d/%d", idx + 1, len(files))

    # Build DataFrame and save
    df = pd.DataFrame(rows)
    results_csv = output_dir / "results.csv"
    df.to_csv(results_csv, index=False)
    logger.info("Saved %s (%d rows)", results_csv, len(df))

    # Aggregate deinterleaver metrics (all official metrics + pairwise MCC/F1).
    from ..evaluation.metrics import aggregate_deinterleaver_metrics

    agg: dict = {}
    deint_agg = aggregate_deinterleaver_metrics(deinter_metrics)
    for key, value in deint_agg.items():
        agg[key] = float(value)
    # Backward-compatible scalar aggregate keys for the summary table.
    for metric in ["v_measure", "ami", "ari", "homogeneity", "completeness", "pairwise_mcc", "pairwise_f1"]:
        agg[f"mean_{metric}"] = agg.get(f"{metric}_mean", float("nan"))
        agg[f"median_{metric}"] = agg.get(f"{metric}_median", float("nan"))
    # Latency aggregate from per-row values.
    if "latency_ms_per_pulse" in df.columns:
        vals = pd.to_numeric(df["latency_ms_per_pulse"], errors="coerce").dropna()
        if len(vals) > 0:
            agg["mean_latency_ms_per_pulse"] = float(vals.mean())
            agg["median_latency_ms_per_pulse"] = float(vals.median())
    # Scheduler aggregates from global_fom (achieved controller).
    agg.update({f"sched_{k}": float(v) for k, v in global_fom.summary().items()})
    # Baseline scheduler aggregates (only when a real comparison baseline ran).
    if baseline_controller is not None:
        agg["baseline_name"] = baseline
        agg.update({f"bl_sched_{k}": float(v) for k, v in baseline_fom.summary().items()})
    else:
        agg["baseline_name"] = "none"
    agg["n_files"] = len(files)
    agg["n_empty_scenarios"] = n_empty_scenarios
    agg["n_skipped_scheduler"] = n_skipped_scheduler
    agg["n_scheduler_files"] = max(0, len(files) - n_empty_scenarios - n_skipped_scheduler)
    agg["mode"] = mode
    telemetry.update(step=len(files), type="done", n_files=len(files), **{f"sched_{k}": float(v) for k, v in global_fom.summary().items()})

    # Literature/static baseline (for titles) vs the measured comparison baseline
    # (when a real one ran, it is reported in the printed table as "Baseline").
    baseline = {"v_measure": 0.62, "Pd": 0.65, "Pfa": 0.12}
    targets = {"v_measure": 0.85, "Pd": 0.90, "Pfa": 0.05}
    agg["static_baseline"] = baseline
    agg["targets"] = targets
    if baseline_controller is not None:
        agg["baseline"] = {
            "Pd": float(baseline_fom.summary().get("Pd", float("nan"))),
            "Pfa": float(baseline_fom.summary().get("Pfa", float("nan"))),
        }

    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved aggregate_metrics.json")

    # ROC curve
    try:
        roc_path = output_dir / "roc_curve.pdf"
        global_fom.plot_roc_curve(roc_path)
    except Exception as exc:
        logger.warning("ROC plot failed: %s", exc)

    # Deinterleaving performance PDF (hist of V-measure)
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
            logger.info("Saved deinterleaving_performance.pdf")
    except Exception as exc:
        logger.warning("Deinterleaving plot failed: %s", exc)

    # Print summary table
    print("\n" + "=" * 72)
    print(f" Evaluation Summary — mode={mode} — {len(files)} files")
    print("=" * 72)
    basename = agg.get("baseline_name", "none")
    if basename != "none":
        print(f" Comparison baseline: {basename}")
    print(f" {'Metric':<28} {'Achieved':<12} {'Baseline':<12} {'Target':<12}")
    print("-" * 72)

    def _baseline_val(metric: str) -> float:
        """Prefer the measured comparison baseline, else the static book value."""
        if baseline_controller is not None:
            measured = agg.get(f"bl_sched_{metric}", float("nan"))
            if measured == measured:  # not NaN
                return float(measured)
        return float(baseline.get(metric, float("nan")))

    for metric in ["v_measure", "Pd", "Pfa"]:
        achieved = agg.get(f"mean_{metric}", agg.get(f"sched_{metric}", float("nan")))
        if metric == "Pd":
            achieved = agg.get("sched_Pd", float("nan"))
        if metric == "Pfa":
            achieved = agg.get("sched_Pfa", float("nan"))
        base = _baseline_val(metric)
        tgt = targets.get(metric, float("nan"))
        print(f" {metric:<28} {achieved:<12.4f} {base:<12.4f} {tgt:<12.4f}")
    if baseline_controller is None:
        print(" (Baseline column = static literature value; pass --baseline to measure one)")
    # Extra scheduler metrics
    for k in ["sched_avg_intercept_rate", "sched_avg_intercept_time_error_us", "sched_avg_reward"]:
        if k in agg:
            print(f" {k:<28} {agg[k]:<12.4f}")
    if baseline_controller is not None:
        for k in ["bl_sched_avg_intercept_rate", "bl_sched_avg_intercept_time_error_us", "bl_sched_avg_reward"]:
            if k in agg:
                print(f" {k:<28} {agg[k]:<12.4f}")
    if n_empty_scenarios or n_skipped_scheduler:
        print(f" Empty/unusable scenarios skipped: {n_empty_scenarios} empty, {n_skipped_scheduler} unusable "
              f"(evaluated {agg['n_scheduler_files']}/{len(files)})")
    print("=" * 72)
    print(f"Results: {results_csv}")
    print(f"Output dir: {output_dir}")
    print()

    return {"dataframe": df, "aggregate": agg, "output_dir": output_dir}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--deinterleaver-ckpt", type=str, default="checkpoints/deinterleaver/best.pt")
    parser.add_argument("--scheduler-ckpt", type=str, default="checkpoints/scheduler/best.pt")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--mode", type=str, choices=["scan", "stare"], default="scan")
    parser.add_argument("--norm-stats", type=str, default=None,
        help="Path to train-fitted normalization stats JSON (leakage-free test eval).")
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["none", "random", "round_robin", "sequential_sweep", "fixed_periodic_scan",
                 "highest_occupancy", "highest_uncertainty", "revisit_heuristic"],
        default="none",
        help="Comparison scheduler to run alongside the learned controller (none disables).",
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
    )


if __name__ == "__main__":
    main()

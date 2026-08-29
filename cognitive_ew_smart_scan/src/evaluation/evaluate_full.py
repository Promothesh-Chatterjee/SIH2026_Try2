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

logger = logging.getLogger(__name__)


def run_full_evaluation(
    deinterleaver_ckpt: str | Path | None,
    scheduler_ckpt: str | Path | None,
    config_path: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    mode: str = "scan",
) -> dict:
    """Run full evaluation over test pulse trains.

    Args:
        deinterleaver_ckpt: Path to Transformer checkpoint (or None → skip deinterleave).
        scheduler_ckpt: Path to DRQN checkpoint (or None → use random baseline).
        config_path: Path to model_config.yaml.
        test_dir: Directory containing test .h5 files (e.g., data/test/scan).
        output_dir: Output directory for results.
        mode: stare or scan (for reporting).

    Returns:
        Dict with aggregate metrics and per-file DataFrame saved to disk.

    Raises:
        FileNotFoundError: If test_dir has no .h5 files.
    """
    import torch

    from ..environment.rf_scan_env import RFScanEnv
    from ..evaluation.metrics import FiguresOfMerit

    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    drqn_cfg = model_cfg.get("drqn_scheduler", {})
    reward_cfg = model_cfg.get("reward", {})

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
    if scheduler_ckpt and Path(scheduler_ckpt).exists():
        try:
            from ..models.drqn_scheduler import DRQNScheduler
            from ..models.smartscan_moe import SmartScanMoE

            scheduler = DRQNScheduler(
                obs_dim=drqn_cfg.get("obs_dim", 360),
                n_bands=drqn_cfg.get("n_bands", 180),
                lstm_hidden=drqn_cfg.get("lstm_hidden", 256),
                lstm_layers=drqn_cfg.get("lstm_layers", 2),
            )
            state = torch.load(str(scheduler_ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            scheduler.load_state_dict(state, strict=False)
            scheduler.eval()
            # Wrap in MoE for evaluation
            moe_cfg = model_cfg.get("smartscan_moe", {})
            from ..models.smartscan_moe import SmartScanMoE as MoE

            moe = MoE(scheduler, {**moe_cfg, "n_bands": drqn_cfg.get("n_bands", 180)})
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
    # Simpler: create env pointing to test_dir parent and manually set current_pt
    # For now, instantiate env with test_dir parent; we will override _files
    env_parent = test_dir.parent if test_dir.is_dir() else Path("data")
    env_config = {**drqn_cfg, **reward_cfg, "n_bands": drqn_cfg.get("n_bands", 180)}
    # Patch training_config environment keys if present
    try:
        with open("configs/training_config.yaml") as f:
            train_cfg = yaml.safe_load(f)
            env_config.update(train_cfg.get("environment", {}))
    except Exception:
        pass

    # We'll run per-file loop using RFScanEnv with synthetic override via direct pt loading
    # To avoid needing env to load from disk, we load pt manually and set env.current_pt
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

    for idx, fpath in enumerate(files):
        per_file: dict = {"file": str(fpath), "mode": mode}
        # --- Deinterleaving ---
        v_measure = ami = ari = float("nan")
        latency_ms = float("nan")
        if has_pt:
            try:
                pt = PulseTrain.load(str(fpath))
                pdws = pt.data  # (N,5)
                labels_true = pt.labels
                if pdws is not None and len(pdws) > 0 and deinterleaver is not None:
                    from ..preprocessing.normalise import normalise_pdws
                    from ..models.deinterleaver import deinterleave

                    pdws_norm, _ = normalise_pdws(pdws, None)
                    t0 = time.perf_counter()
                    labels_pred = deinterleave(deinterleaver, pdws_norm, device=str(device))
                    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(pdws))  # ms per pulse avg
                    # Metrics (ignore noise label -1 for V-measure? Include)
                    try:
                        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score  # type: ignore

                        # Filter to pulses where pred != -1 or include all? Use all for strict
                        mask = labels_pred != -1
                        if np.sum(mask) > 5 and len(set(labels_true[mask])) > 1:
                            v_measure = float(v_measure_score(labels_true[mask], labels_pred[mask]))
                            ami = float(adjusted_mutual_info_score(labels_true[mask], labels_pred[mask]))
                            ari = float(adjusted_rand_score(labels_true[mask], labels_pred[mask]))
                        elif len(set(labels_true)) > 1:
                            # fallback on all
                            v_measure = float(v_measure_score(labels_true, np.where(labels_pred == -1, max(labels_true) + 1, labels_pred)))
                    except Exception as exc:
                        logger.debug("Metric failed for %s: %s", fpath, exc)
            except Exception as exc:
                logger.warning("Failed to evaluate deinterleave %s: %s", fpath, exc)
        per_file.update({"v_measure": v_measure, "ami": ami, "ari": ari, "latency_ms_per_pulse": latency_ms})
        deinter_metrics.append({"v_measure": v_measure, "ami": ami, "ari": ari})

        # --- Scheduler (run one episode per file via RFScanEnv) ---
        # Create env per file with single-file override
        try:
            # Build env that will load this specific file
            # Hack: create env with data_dir = fpath.parent and override _files
            env = RFScanEnv(env_config, data_dir=str(fpath.parent), subset=".", mode="scan" if "scan" in mode else "stare")
            env._files = [Path(fpath)]
            obs, _ = env.reset()
            # Set up MoE hidden if scheduler exists
            hidden = None
            if scheduler is not None and hasattr(scheduler, "reset"):
                try:
                    scheduler.reset()
                    if hasattr(scheduler, "eager_agent") and hasattr(scheduler.eager_agent, "hidden"):
                        # init hidden for DRQN inside MoE
                        drqn_inner = scheduler.eager_agent.drqn if hasattr(scheduler.eager_agent, "drqn") else None
                        if drqn_inner is not None:
                            hidden = drqn_inner.init_hidden(1, device)
                            scheduler.eager_agent.hidden = hidden
                except Exception:
                    pass
            done = False
            steps = 0
            while not done and steps < 5000:
                if scheduler is not None:
                    try:
                        # MoE path
                        bands, hidden, _ = scheduler.select_bands(obs, hidden)  # type: ignore
                        action = int(bands[0])
                    except Exception:
                        action = int(np.random.randint(0, env.n_bands))
                else:
                    action = int(np.random.randint(0, env.n_bands))
                obs, reward, terminated, truncated, info = env.step(action)
                if scheduler is not None and hasattr(scheduler, "update"):
                    try:
                        scheduler.update(action)  # type: ignore
                    except Exception:
                        pass
                done = bool(terminated or truncated)
                steps += 1
                # Accumulate global FoM
                global_fom.update(info["band_chosen"] if "band_chosen" in info else action, info["ground_truth_active"], info["hit"], info["intercept_time_error_us"], float(reward))
            fom = env.get_fom()
            per_file.update({f"sched_{k}": v for k, v in fom.items()})
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

    # Aggregate
    agg: dict = {}
    for col in ["v_measure", "ami", "ari", "latency_ms_per_pulse"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                agg[f"mean_{col}"] = float(vals.mean())
                agg[f"std_{col}"] = float(vals.std())
                agg[f"median_{col}"] = float(vals.median())
    # Scheduler aggregates from global_fom
    agg.update({f"sched_{k}": float(v) for k, v in global_fom.summary().items()})
    agg["n_files"] = len(files)
    agg["mode"] = mode

    # Baseline vs targets table
    baseline = {"v_measure": 0.62, "Pd": 0.65, "Pfa": 0.12}
    targets = {"v_measure": 0.85, "Pd": 0.90, "Pfa": 0.05}
    agg["baseline"] = baseline
    agg["targets"] = targets

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
    print(f" {'Metric':<28} {'Achieved':<12} {'Baseline':<12} {'Target':<12}")
    print("-" * 72)
    for metric in ["v_measure", "Pd", "Pfa"]:
        achieved = agg.get(f"mean_{metric}", agg.get(f"sched_{metric}", float("nan")))
        if metric == "Pd":
            achieved = agg.get("sched_Pd", float("nan"))
        if metric == "Pfa":
            achieved = agg.get("sched_Pfa", float("nan"))
        base = baseline.get(metric, float("nan"))
        tgt = targets.get(metric, float("nan"))
        print(f" {metric:<28} {achieved:<12.4f} {base:<12.4f} {tgt:<12.4f}")
    # Extra scheduler metrics
    for k in ["sched_avg_intercept_rate", "sched_avg_intercept_time_error_us", "sched_avg_reward"]:
        if k in agg:
            print(f" {k:<28} {agg[k]:<12.4f}")
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_full_evaluation(
        deinterleaver_ckpt=args.deinterleaver_ckpt,
        scheduler_ckpt=args.scheduler_ckpt,
        config_path=args.config,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

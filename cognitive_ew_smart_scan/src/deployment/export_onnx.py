"""
ONNX export for deinterleaver and DRQN scheduler (edge deployment on Jetson).

Handles CUDA/CPU fallback and dynamic axes for variable-length sequences.
"""

import argparse
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

NORM_STATS_FILENAME = "normalization_stats.json"


def _load_checkpoint_state(ckpt_path: Path, model, what: str) -> dict:
    """Load and STRICTLY apply a checkpoint to a model, failing fast.

    Phase 15: an ONNX export must never silently continue with random-initialized
    weights. Any of the following raises instead of exporting:
      * checkpoint missing                 -> FileNotFoundError
      * checkpoint corrupted/unreadable    -> RuntimeError
      * state dict not a tensor mapping    -> RuntimeError
      * architecture/hyperparameter mismatch (missing/unexpected keys, size
        mismatch)                         -> RuntimeError

    Returns:
        The checkpoint ``metadata`` dict (empty when the checkpoint is a bare
        state dict without metadata).
    """
    import torch

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        payload = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Checkpoint {ckpt_path} is corrupted or unreadable: {exc}") from exc

    if isinstance(payload, dict) and "state_dict" in payload:
        metadata = dict(payload.get("metadata") or {})
        state = payload["state_dict"]
    else:
        metadata = {}
        state = payload

    if not isinstance(state, dict) or not state or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain a state dict of tensors "
            f"(got {type(state).__name__})."
        )

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"State dict mismatch for {what} checkpoint {ckpt_path} — the "
            f"architecture or hyperparameters do not match ({exc})."
        ) from exc
    logger.info("Loaded %s state %s (strict)", what, ckpt_path)
    return metadata


def _resolve_normalization_meta(
    ckpt_meta: dict, ckpt_path: str | Path, stats_candidates: list[Path] | None = None
) -> tuple[str, str | None]:
    """Resolve the normalisation provenance (stats hash, stats path) for ONNX.

    Order: (1) hash already stamped into the checkpoint metadata at training
    time; (2) compute from the ``normalization_stats.json`` persisted beside the
    checkpoint (override with ``stats_candidates`` for testing); (3) fall back
    to "unknown" when no stats artifact exists (weights are still strictly
    loaded — provenance is recorded separately from model weights).

    Returns:
        Tuple ``(stats_hash, stats_path_or_None)``.
    """
    ckpt_meta = ckpt_meta or {}
    existing = ckpt_meta.get("normalization_stats_hash")
    if existing:
        return str(existing), ckpt_meta.get("normalization_stats_path")

    from ..preprocessing.normalise import load_normalization_stats, normalization_stats_hash

    candidates = stats_candidates if stats_candidates is not None else [
        Path(ckpt_path).parent / NORM_STATS_FILENAME,
        Path("configs") / NORM_STATS_FILENAME,
        Path("checkpoints/deinterleaver") / NORM_STATS_FILENAME,
    ]
    for cand in candidates:
        if cand.exists():
            try:
                stats = load_normalization_stats(cand)
            except Exception as exc:
                logger.warning("Could not read normalization stats %s: %s", cand, exc)
                continue
            return normalization_stats_hash(stats), str(cand)
    return "unknown", None


def _attach_onnx_metadata(onnx_path: Path, props: dict[str, str]) -> bool:
    """Attach metadata_props to an exported ONNX model in place.

    Uses the ``onnx`` package when available; falls back to writing a sidecar
    ``*.metadata.json`` (same key/value pairs) so provenance survives on hosts
    without the ``onnx`` package.
    """
    try:
        import onnx  # type: ignore

        model = onnx.load(str(onnx_path))
        for key, value in props.items():
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = value
        onnx.save(model, str(onnx_path))
        logger.info("Attached ONNX metadata_props to %s", onnx_path)
        return True
    except Exception as exc:
        sidecar = onnx_path.with_suffix(onnx_path.suffix + ".metadata.json")
        try:
            import json

            sidecar.write_text(json.dumps(props, indent=2), encoding="utf-8")
        except Exception:
            pass
        logger.warning("onnx package unavailable or metadata attach failed (%s); wrote %s", exc, sidecar)
        return False


def export_deinterleaver(model_cfg_path: str, ckpt_path: str, output_path: str, opset: int = 17) -> Path:
    """Export PDWTransformerEncoder to ONNX.

    Args:
        model_cfg_path: Path to model_config.yaml.
        ckpt_path: Checkpoint .pt path.
        output_path: Output .onnx path.
        opset: ONNX opset version.

    Returns:
        Path to exported ONNX file.

    Raises:
        FileNotFoundError: If checkpoint missing.
        RuntimeError: If the checkpoint is corrupted or the state dict does not
            match the configured architecture (NO random-init export).
    """
    import torch

    with open(model_cfg_path) as f:
        cfg = yaml.safe_load(f)["deinterleaver"]

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from ..models.deinterleaver import PDWTransformerEncoder

    device = torch.device("cpu")  # Export on CPU for portability
    model = PDWTransformerEncoder(
        pdw_dim=cfg.get("pdw_dim", 6),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        embed_dim=cfg.get("embed_dim", 64),
    ).to(device)
    ckpt_metadata = _load_checkpoint_state(ckpt_path, model, "deinterleaver")
    model.eval()

    # Dummy input: (1, N, 6) with N dynamic
    dummy = torch.randn(1, 32, cfg.get("pdw_dim", 6), device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["pdws"],
            output_names=["embeddings"],
            dynamic_axes={"pdws": {1: "seq_len"}, "embeddings": {1: "seq_len"}},
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("Exported deinterleaver ONNX → %s (opset %d)", output_path, opset)
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed for deinterleaver: {exc}") from exc

    # Phase 14: stamp normalisation provenance (train-stats hash) into metadata.
    norm_hash, norm_path = _resolve_normalization_meta(ckpt_metadata, ckpt_path)
    _attach_onnx_metadata(
        output_path,
        {
            "normalization_stats_hash": norm_hash,
            "normalization_stats_path": norm_path or "",
            "preproc_version": str(ckpt_metadata.get("preproc_version", "v1")),
            "git_revision": str(ckpt_metadata.get("git_revision", "unknown")),
        },
    )

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"pdws": dummy.cpu().numpy()})
        logger.info("ONNX verification OK, output shape %s", out[0].shape)
    except Exception as exc:
        logger.warning("ONNX verification skipped/failed: %s", exc)

    return output_path


def export_scheduler(model_cfg_path: str, ckpt_path: str, output_path: str, opset: int = 17) -> Path:
    """Export DRQNScheduler to ONNX (LSTM unwrapped for edge).

    Args:
        model_cfg_path: Path to model_config.yaml.
        ckpt_path: Checkpoint .pt path.
        output_path: Output .onnx path.
        opset: ONNX opset.

    Returns:
        Path to ONNX file.

    Raises:
        FileNotFoundError: If checkpoint missing.
        RuntimeError: If the checkpoint is corrupted or the state dict does not
            match the configured architecture (NO random-init export).
    """
    import torch

    with open(model_cfg_path) as f:
        cfg = yaml.safe_load(f)["drqn_scheduler"]

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from ..models.drqn_scheduler import DRQNScheduler

    device = torch.device("cpu")
    n_bands = int(cfg.get("n_bands", 36))
    n_modes = int(cfg.get("n_modes", 5))
    n_actions = int(cfg.get("n_actions", n_bands * n_modes))
    model = DRQNScheduler(
        obs_dim=cfg.get("obs_dim", 360),
        n_bands=n_bands,
        n_actions=n_actions,
        n_modes=n_modes,
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
    ).to(device)
    _ckpt_metadata = _load_checkpoint_state(ckpt_path, model, "scheduler")
    model.eval()

    # Dummy: (1, seq_len, obs_dim) with dynamic seq_len
    dummy_obs = torch.randn(1, 4, cfg.get("obs_dim", 360), device=device)

    # Wrapper to export without hidden state (init to zeros inside).
    # Outputs Q-values, per-action interception probability, and per-step
    # intercept time (the canonical 180-space time-frequency contract).
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, drqn: DRQNScheduler) -> None:
            super().__init__()
            self.drqn = drqn

        def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, aux, _ = self.drqn(obs, None)
            return q, aux["intercept_prob"], aux["intercept_time_us"]

    wrapper = _ExportWrapper(model).to(device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapper,
            dummy_obs,
            str(output_path),
            input_names=["obs"],
            output_names=["q_values", "intercept_prob", "intercept_time_us"],
            dynamic_axes={"obs": {1: "seq_len"}},
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("Exported scheduler ONNX → %s", output_path)
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed for scheduler: {exc}") from exc

    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"obs": dummy_obs.cpu().numpy()})
        logger.info("Scheduler ONNX verification OK, shapes %s (q/prob/time)", [o.shape for o in out])
    except Exception as exc:
        logger.warning("Scheduler ONNX verification skipped: %s", exc)

    return output_path


def main() -> None:
    """CLI for ONNX export."""
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--deinterleaver-ckpt", type=str, default="checkpoints/deinterleaver/best.pt")
    parser.add_argument("--scheduler-ckpt", type=str, default="checkpoints/scheduler/best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 15 fail-fast: missing/mismatched checkpoints abort the CLI with a
    # non-zero exit — never a silent skip or a random-init export.
    if not Path(args.deinterleaver_ckpt).exists():
        raise FileNotFoundError(f"Deinterleaver checkpoint missing: {args.deinterleaver_ckpt}")
    export_deinterleaver(
        args.model_config, args.deinterleaver_ckpt, str(out_dir / "deinterleaver.onnx"), opset=args.opset
    )

    if not Path(args.scheduler_ckpt).exists():
        raise FileNotFoundError(f"Scheduler checkpoint missing: {args.scheduler_ckpt}")
    export_scheduler(
        args.model_config, args.scheduler_ckpt, str(out_dir / "scheduler.onnx"), opset=args.opset
    )

    # Phase 17: onnx/ directory summary metadata (canonical contract artifact).
    from ..utils.checkpoint_meta import current_git_revision, write_checkpoint_metadata

    write_checkpoint_metadata(
        out_dir / "metadata.json",
        {
            "git_revision": current_git_revision(),
            "preproc_version": "v1",
            "opset": args.opset,
            "deinterleaver_ckpt": str(args.deinterleaver_ckpt),
            "scheduler_ckpt": str(args.scheduler_ckpt),
        },
        artifacts=["deinterleaver.onnx", "scheduler.onnx"],
    )
    logger.info("Exported ONNX artifacts to %s", out_dir)


if __name__ == "__main__":
    main()

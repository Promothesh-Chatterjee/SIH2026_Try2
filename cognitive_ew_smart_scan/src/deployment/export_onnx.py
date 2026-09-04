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
        RuntimeError: If export fails.
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
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded deinterleaver state %s", ckpt_path)
    except Exception as exc:
        logger.warning("Failed to load state, exporting random init: %s", exc)
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
    n_modes = int(cfg.get("n_modes", 1))
    n_actions = int(cfg.get("n_actions", n_bands * n_modes))
    model = DRQNScheduler(
        obs_dim=cfg.get("obs_dim", 360),
        n_bands=n_bands,
        n_actions=n_actions,
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
    ).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded scheduler state %s", ckpt_path)
    except Exception as exc:
        logger.warning("Failed to load scheduler state: %s", exc)
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

    if Path(args.deinterleaver_ckpt).exists():
        try:
            export_deinterleaver(args.model_config, args.deinterleaver_ckpt, str(out_dir / "deinterleaver.onnx"), opset=args.opset)
        except Exception as exc:
            logger.error("Deinterleaver export failed: %s", exc)
    else:
        logger.warning("Deinterleaver ckpt missing: %s", args.deinterleaver_ckpt)

    if Path(args.scheduler_ckpt).exists():
        try:
            export_scheduler(args.model_config, args.scheduler_ckpt, str(out_dir / "scheduler.onnx"), opset=args.opset)
        except Exception as exc:
            logger.error("Scheduler export failed: %s", exc)
    else:
        logger.warning("Scheduler ckpt missing: %s", args.scheduler_ckpt)


if __name__ == "__main__":
    main()

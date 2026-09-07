"""GNU RF deinterleaver loading utility — Phase 3V.

Loads the ACTUAL production trained deinterleaver checkpoint and hands it to
``GnuRfSchedulerTranslation`` through the existing injection boundary:

    GnuRfSchedulerTranslation(
        deinterleaver_model=<model instance>,
        deinterleaver_config=<config dict>,
    )

This utility ONLY orchestrates loading/validation.  It does NOT define any model
architecture, normalization transform, or clustering algorithm.  The
authoritative model class is the master ``PDWTransformerEncoder``; the
authoritative statistics must match the production ``normalise_pdws`` format.

Loader behaviour
----------------
- Loads the wrapped or raw checkpoint via ``torch.load``, unwrapping a
  ``{"state_dict": ..., "metadata": ...}`` envelope exactly like the existing
  production loaders (``src/deployment/api.py``, ``export_onnx.py``,
  ``evaluate_full.py``, ``train_scheduler.py``).
- Instantiates the real ``PDWTransformerEncoder`` with the production config.
- Loads normalization statistics via the existing
  ``src.preprocessing.normalise.load_normalization_stats`` (no re-implementation).
- Validates compatibility and FAILS LOUDLY on:
    - missing checkpoint            -> FileNotFoundError
    - missing normalization stats   -> FileNotFoundError
    - incompatible checkpoint       -> ValueError
    - invalid model configuration   -> ValueError
  It NEVER silently falls back to a mock or to simplified features.

Example
-------
    from deinterleaver_loader import load_deinterleaver
    payload = load_deinterleaver()
    tr = GnuRfSchedulerTranslation(
        deinterleaver_model=payload["model"],
        deinterleaver_config=payload["config"],
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_GNU_RF_SCRIPTS = str(Path(__file__).resolve().parent)
_MASTER_PKG = str(_REPO / "cognitive_ew_smart_scan")
if _MASTER_PKG not in sys.path:
    sys.path.insert(0, _MASTER_PKG)
if _GNU_RF_SCRIPTS not in sys.path:
    sys.path.insert(0, _GNU_RF_SCRIPTS)

__all__ = ["load_deinterleaver", "DEFAULT_DEINTERLEAVER_CONFIG"]


# Production architecture config (configs/model_config.yaml -> deinterleaver:)
DEFAULT_ARCH = {
    "pdw_dim": 6,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "embed_dim": 64,
}

# Translation-layer config for a production-run parity.  These match the env's
# defaults (cognitive_rf_scan_env: min_pulses=50, interval_steps=10) and the
# deinterleaver defaults (min_cluster_size=10, min_samples=5).
#
# window_size/stride are intentionally ABSENT: scheduler_translation._run_perception
# defaults them exactly like the production env
# (window_size=min(2048, len(pdws_norm)), stride=window_size//2), so presence of
# a None value would suppress those defaults.
DEFAULT_DEINTERLEAVER_CONFIG = {
    "min_pulses": 50,
    "interval_steps": 10,
    "device": "cpu",
    "min_cluster_size": 10,
    "min_samples": 5,
    # "fit_stats" is populated by load_deinterleaver
}

# State-dict prefixes that identify a PDWTransformerEncoder checkpoint.
_DEINTERLEAVER_KEY_PREFIXES = ("input_proj.", "pos_encoding.", "transformer.", "output_proj.")


def _default_checkpoint_path() -> Path:
    return _REPO / "cognitive_ew_smart_scan" / "checkpoints" / "best.pt"


def _default_stats_path() -> Path:
    return _REPO / "cognitive_ew_smart_scan" / "checkpoints" / "normalization_stats.json"


def _validate_metadata(metadata: Any, arch_tag: str) -> None:
    """Log/complain when metadata contradicts the PDWTransformerEncoder arch."""
    if not isinstance(metadata, dict):
        return
    arch = metadata.get("arch")
    mode = metadata.get("mode")
    if arch is not None and arch != arch_tag:
        raise ValueError(
            f"Checkpoint metadata arch '{arch}' != expected '{arch_tag}'"
        )
    if mode is not None and mode != "deinterleaver":
        raise ValueError(
            f"Checkpoint metadata mode '{mode}' is not 'deinterleaver'"
        )


def _validate_state_dict(state: dict, model: Any) -> None:
    """Reject checkpoints that are not a compatible deinterleaver state dict."""
    keys = list(state.keys())
    if not any(k.startswith(_DEINTERLEAVER_KEY_PREFIXES) for k in keys):
        raise ValueError(
            "Checkpoint is not a PDWTransformerEncoder state dict "
            f"(no 'input_proj'/'transformer'/'output_proj' keys; got {sorted(set(k.split('.')[0] for k in keys))[:8]})"
        )
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint incompatible with PDWTransformerEncoder arch: {exc}"
        ) from exc
    if missing:
        raise ValueError(
            f"Checkpoint incompatible with PDWTransformerEncoder arch: missing keys {missing[:8]}"
        )


def load_deinterleaver(
    checkpoint_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    *,
    arch: dict | None = None,
    device: str = "cpu",
    env_perception_config: dict | None = None,
) -> dict[str, Any]:
    """Load the actual production deinterleaver model + normalization stats.

    Parameters
    ----------
    checkpoint_path : optional
        Path to the production checkpoint.  Defaults to
        ``cognitive_ew_smart_scan/checkpoints/best.pt``.
    stats_path : optional
        Path to the production normalization statistics JSON.  Defaults to
        ``cognitive_ew_smart_scan/checkpoints/normalization_stats.json``.
    arch : optional dict
        Override architecture params (defaults to production model_config).
    device : str
        Device for model inference ("cpu").
    env_perception_config : optional dict
        Overrides for the translation-layer perception gating (min_pulses,
        interval_steps, window_size, stride, min_cluster_size, min_samples).

    Returns
    -------
    dict with
        "model" : PDWTransformerEncoder (eval mode, CPU)
        "config" : deinterleaver_config dict consumable by
                   GnuRfSchedulerTranslation (includes fit_stats)
        "checkpoint" : resolved checkpoint path
        "stats_path" : resolved stats path
        "metadata" : checkpoint metadata (or {})

    Raises
    ------
    FileNotFoundError : checkpoint or stats file missing.
    ValueError : checkpoint incompatible (missing keys / wrong arch) or
                 invalid model configuration.
    """
    from src.models.deinterleaver import PDWTransformerEncoder

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path()
    if not ckpt.exists():
        raise FileNotFoundError(f"Deinterleaver checkpoint not found: {ckpt}")

    stats = Path(stats_path) if stats_path is not None else _default_stats_path()
    if not stats.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats}")

    import json

    from src.preprocessing.normalise import load_normalization_stats

    fit_stats = load_normalization_stats(stats)

    arch_cfg = dict(DEFAULT_ARCH)
    if arch is not None:
        arch_cfg.update(arch)
    required_arch_keys = ("pdw_dim", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "embed_dim")
    for k in required_arch_keys:
        if arch_cfg.get(k) is None:
            raise ValueError(f"Invalid model configuration: missing '{k}'")

    model = PDWTransformerEncoder(
        pdw_dim=arch_cfg["pdw_dim"],
        d_model=arch_cfg["d_model"],
        nhead=arch_cfg["nhead"],
        num_layers=arch_cfg["num_layers"],
        dim_feedforward=arch_cfg["dim_feedforward"],
        dropout=arch_cfg["dropout"],
        embed_dim=arch_cfg["embed_dim"],
    )

    payload = torch_load_any(str(ckpt))
    metadata: dict = {}
    if isinstance(payload, dict) and "state_dict" in payload:
        metadata = payload.get("metadata") or {}
        payload = payload["state_dict"]

    _validate_metadata(metadata, arch_tag="PDWTransformerEncoder")
    _validate_state_dict(payload, model)

    model.to(torch_device(device))
    model.eval()

    percep_cfg = dict(DEFAULT_DEINTERLEAVER_CONFIG)
    if env_perception_config:
        percep_cfg.update(env_perception_config)
    percep_cfg["fit_stats"] = fit_stats
    percep_cfg["device"] = device

    logger.info(
        "Loaded real deinterleaver %s (stats %s, git=%s) -> %s",
        ckpt.name, stats.name, metadata.get("git_revision", "n/a"), payload.size() if hasattr(payload, "size") else "state"
    )
    return {
        "model": model,
        "config": percep_cfg,
        "checkpoint": ckpt,
        "stats_path": stats,
        "metadata": metadata,
    }


def torch_load_any(path: str):
    """Wrap torch.load to raise on unreadable checkpoints."""
    import torch
    return torch.load(str(path), map_location="cpu")


def torch_device(device: str):
    """Resolve a device name with the production fallback semantics."""
    import torch
    try:
        if device and (torch.cuda.is_available() or device == "cpu"):
            return torch.device(device)
        return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
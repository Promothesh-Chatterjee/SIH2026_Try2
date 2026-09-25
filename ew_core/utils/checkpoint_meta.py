"""Checkpoint provenance metadata and hardened checkpoint contract.

Adds git revision, dataset split, preproc version, feature order, model arch,
seed, training metrics and timestamp to saved model checkpoints so results are
reproducible and non-fabricated (P0-9).

Provides two explicit saving/loading modes:
1. WEIGHTS_ONLY: Copies model/target state dicts, arch config, norm stats;
   omits optimizer, replay, step/epoch counters, epsilon schedule, RNG.
   Used when retraining from 25k baseline to guarantee clean restart.
2. EXACT_CONTINUATION: Persists all states including RNG (CPU, CUDA, NumPy, Python),
   replay manifest, optimizer, and learning rate scheduler.
   Used for in-flight exact continuation.
"""

from __future__ import annotations

import datetime
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from ..contracts import FEATURE_ORDER as _FEATURE_ORDER

logger = logging.getLogger(__name__)

# Backward-compatible list export for checkpoint metadata callers.
FEATURE_ORDER = list(_FEATURE_ORDER)
PREPROC_VERSION = "v1"  # bump whenever normalise.py feature layout changes


class CheckpointMode(str, Enum):
    """Explicit checkpoint operation mode contract."""
    WEIGHTS_ONLY = "weights_only"
    EXACT_CONTINUATION = "exact_continuation"


def current_git_revision() -> str:
    """Return the short HEAD git revision, or "unknown" if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception as exc:
        logger.debug("Failed getting git revision: %s", exc)
    return "unknown"


def build_train_metadata(
    *,
    split: str,
    n_bands: int,
    feature_order: list[str] | None = None,
    arch: str,
    seed: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a canonical checkpoint metadata blob."""
    order = list(feature_order) if feature_order else list(FEATURE_ORDER)
    meta: dict[str, Any] = {
        "git_revision": current_git_revision(),
        "split": split,
        "n_bands": int(n_bands),
        "preproc_version": PREPROC_VERSION,
        "feature_order_per_band": order,
        "arch": arch,
        "seed": int(seed),
        "metrics": dict(metrics),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_state(model: torch.nn.Module, path: os.PathLike | str, metadata: dict[str, Any]) -> None:
    """Save a model state dict together with canonical metadata (legacy compat)."""
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def write_checkpoint_metadata(
    path: os.PathLike | str, metadata: dict[str, Any], artifacts: list[str] | None = None
) -> dict[str, Any]:
    """Write a human-readable metadata.json sidecar for a checkpoint set."""
    payload = dict(metadata)
    payload["artifacts"] = [str(a) for a in (artifacts or [])]
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def save_hardened_checkpoint(
    path: os.PathLike | str,
    mode: CheckpointMode | str,
    model: torch.nn.Module,
    target_model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    arch_config: dict[str, Any] | None = None,
    norm_stats: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    global_step: int | None = None,
    episode: int | None = None,
    epsilon: float | None = None,
    replay_manifest: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
    include_rng: bool = True,
) -> Path:
    """Save a checkpoint under an explicit contract: WEIGHTS_ONLY or EXACT_CONTINUATION.

    Args:
        path: Target file path.
        mode: CheckpointMode.WEIGHTS_ONLY or CheckpointMode.EXACT_CONTINUATION.
        model: Online model.
        target_model: Target model (optional).
        optimizer: Optimizer (persisted only if mode == EXACT_CONTINUATION).
        lr_scheduler: LR scheduler (persisted only if mode == EXACT_CONTINUATION).
        arch_config: Architecture config dict.
        norm_stats: Normalization stats dict.
        metadata: Provenance metadata dict.
        global_step: Global training step.
        episode: Episode count.
        epsilon: Exploration parameter.
        replay_manifest: Replay buffer summary/manifest.
        extra_state: Arbitrary extra state dictionary.
        include_rng: Whether to capture RNG state in EXACT_CONTINUATION mode.

    Returns:
        Path to the saved checkpoint.
    """
    mode = CheckpointMode(mode)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "checkpoint_mode": mode.value,
        "schema_version": "2.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_revision": current_git_revision(),
        "state_dict": model.state_dict(),
        "target_state_dict": target_model.state_dict() if target_model is not None else None,
        "arch_config": dict(arch_config or {}),
        "norm_stats": dict(norm_stats or {}),
        "metadata": dict(metadata or {}),
    }

    if mode == CheckpointMode.WEIGHTS_ONLY:
        # Guarantee omission of optimizer, RNG, replay, and counters for clean restart
        payload["global_step"] = 0
        payload["episode"] = 0
        payload["epsilon"] = None
        payload["optimizer_state_dict"] = None
        payload["lr_scheduler_state_dict"] = None
        payload["rng_state"] = None
        payload["cuda_rng_state"] = None
        payload["np_rng_state"] = None
        payload["py_rng_state"] = None
        payload["replay_manifest"] = None
    elif mode == CheckpointMode.EXACT_CONTINUATION:
        payload["global_step"] = int(global_step) if global_step is not None else 0
        payload["episode"] = int(episode) if episode is not None else 0
        payload["epsilon"] = float(epsilon) if epsilon is not None else None
        payload["optimizer_state_dict"] = optimizer.state_dict() if optimizer is not None else None
        payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict() if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict") else None
        payload["replay_manifest"] = dict(replay_manifest or {})
        if extra_state:
            payload["extra_state"] = dict(extra_state)

        if include_rng:
            payload["rng_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                try:
                    payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
                except Exception:
                    payload["cuda_rng_state"] = None
            else:
                payload["cuda_rng_state"] = None
            payload["np_rng_state"] = np.random.get_state()
            payload["py_rng_state"] = random.getstate()

    torch.save(payload, out_path)
    logger.info("Saved %s checkpoint to %s", mode.value, out_path)
    return out_path


def load_hardened_checkpoint(
    path: os.PathLike | str,
    expected_mode: CheckpointMode | str | None = None,
    device: torch.device | str = "cpu",
    model: torch.nn.Module | None = None,
    target_model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    strict_weights: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint enforcing the specified mode contract.

    Args:
        path: Path to checkpoint .pt.
        expected_mode: Optional CheckpointMode enforcement.
        device: Device to map tensors to.
        model: Optional model into which state_dict will be loaded.
        target_model: Optional target model into which target_state_dict will be loaded.
        optimizer: Optional optimizer (only loaded if expected_mode != WEIGHTS_ONLY).
        lr_scheduler: Optional lr_scheduler (only loaded if expected_mode != WEIGHTS_ONLY).
        strict_weights: Whether to strictly enforce state_dict layer match.
        restore_rng: Whether to restore RNG states if in EXACT_CONTINUATION mode.

    Returns:
        Structured dictionary containing loaded state, metadata, and mode.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Safe loading handling PyTorch 2.6 defaults
    try:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        try:
            raw = torch.load(ckpt_path, map_location=device)
        except Exception as inner_exc:
            raise RuntimeError(f"Failed loading checkpoint at {ckpt_path}: {inner_exc}") from exc

    # Parse structure (hardened 2.0 vs legacy dict vs raw state dict)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        target_state_dict = raw.get("target_state_dict")
        metadata = raw.get("metadata", {})
        arch_config = raw.get("arch_config", {})
        norm_stats = raw.get("norm_stats", {})
        actual_mode = raw.get("checkpoint_mode")
        global_step = raw.get("global_step", 0)
        episode = raw.get("episode", 0)
        epsilon = raw.get("epsilon")
        opt_state = raw.get("optimizer_state_dict")
        lr_state = raw.get("lr_scheduler_state_dict")
        replay_manifest = raw.get("replay_manifest")
        rng_state = raw.get("rng_state")
        cuda_rng_state = raw.get("cuda_rng_state")
        np_rng_state = raw.get("np_rng_state")
        py_rng_state = raw.get("py_rng_state")
    elif isinstance(raw, dict):
        # Likely raw state dict
        state_dict = raw
        target_state_dict = None
        metadata = {}
        arch_config = {}
        norm_stats = {}
        actual_mode = CheckpointMode.WEIGHTS_ONLY.value
        global_step = 0
        episode = 0
        epsilon = None
        opt_state = None
        lr_state = None
        replay_manifest = None
        rng_state = None
        np_rng_state = None
        py_rng_state = None
    else:
        raise ValueError(f"Unrecognized checkpoint object type: {type(raw)}")

    target_mode = CheckpointMode(expected_mode) if expected_mode is not None else None

    # Apply Weights to model(s)
    if model is not None:
        model.load_state_dict(state_dict, strict=strict_weights)
    if target_model is not None:
        t_state = target_state_dict if target_state_dict is not None else state_dict
        target_model.load_state_dict(t_state, strict=strict_weights)

    # In WEIGHTS_ONLY mode: Never restore optimizer, counters, or RNG
    if target_mode == CheckpointMode.WEIGHTS_ONLY:
        logger.info("Enforcing WEIGHTS_ONLY contract: optimizer, counters, and RNG are reset/ignored.")
        return {
            "mode": CheckpointMode.WEIGHTS_ONLY.value,
            "state_dict": state_dict,
            "target_state_dict": target_state_dict,
            "arch_config": arch_config,
            "norm_stats": norm_stats,
            "metadata": metadata,
            "global_step": 0,
            "episode": 0,
            "epsilon": None,
            "optimizer_restored": False,
            "rng_restored": False,
            "raw_checkpoint": raw,
        }

    # In EXACT_CONTINUATION mode (or unconstrained): Restore optimizer and RNG if present
    opt_restored = False
    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
            opt_restored = True
        except Exception as exc:
            logger.warning("Could not restore optimizer state: %s", exc)

    lr_restored = False
    if lr_scheduler is not None and lr_state is not None:
        try:
            lr_scheduler.load_state_dict(lr_state)
            lr_restored = True
        except Exception as exc:
            logger.warning("Could not restore lr_scheduler state: %s", exc)

    rng_restored = False
    if restore_rng:
        if rng_state is not None:
            try:
                torch.set_rng_state(rng_state)
                rng_restored = True
            except Exception:
                pass
        if cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            except Exception as exc:
                logger.warning("Could not restore CUDA RNG state: %s", exc)
        if np_rng_state is not None:
            try:
                np.random.set_state(np_rng_state)
            except Exception:
                pass
        if py_rng_state is not None:
            try:
                random.setstate(py_rng_state)
            except Exception:
                pass

    return {
        "mode": actual_mode or CheckpointMode.EXACT_CONTINUATION.value,
        "state_dict": state_dict,
        "target_state_dict": target_state_dict,
        "arch_config": arch_config,
        "norm_stats": norm_stats,
        "metadata": metadata,
        "global_step": int(global_step),
        "episode": int(episode),
        "epsilon": float(epsilon) if epsilon is not None else None,
        "optimizer_restored": opt_restored,
        "lr_scheduler_restored": lr_restored,
        "rng_restored": rng_restored,
        "replay_manifest": replay_manifest,
        "raw_checkpoint": raw,
    }


def validate_checkpoint(
    path: os.PathLike | str,
    mode: CheckpointMode | str | None = None,
    model: torch.nn.Module | None = None,
) -> tuple[bool, list[str]]:
    """Validate a checkpoint file's integrity, schema, and tensor validity.

    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors: list[str] = []
    p = Path(path)
    if not p.exists():
        return False, [f"File does not exist: {p}"]
    if p.stat().st_size == 0:
        return False, [f"File is empty (0 bytes): {p}"]

    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, [f"Failed to load checkpoint with torch.load: {exc}"]

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        errors.append(f"state_dict is not a dictionary (got {type(state_dict)})")
        return False, errors

    if len(state_dict) == 0:
        errors.append("state_dict contains zero parameters")

    # Check for NaN / Inf in parameter tensors
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                errors.append(f"Tensor '{k}' contains NaN values")
            if torch.isinf(v).any():
                errors.append(f"Tensor '{k}' contains Inf values")

    # Model shape matching if model provided
    if model is not None:
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and isinstance(v, torch.Tensor):
                if model_state[k].shape != v.shape:
                    errors.append(
                        f"Shape mismatch for '{k}': expected {model_state[k].shape}, got {v.shape}"
                    )

    # Mode-specific validation
    if mode is not None:
        target_mode = CheckpointMode(mode)
        if target_mode == CheckpointMode.EXACT_CONTINUATION:
            if isinstance(ckpt, dict):
                if "optimizer_state_dict" not in ckpt or ckpt["optimizer_state_dict"] is None:
                    errors.append("EXACT_CONTINUATION requires 'optimizer_state_dict'")
                if "global_step" not in ckpt:
                    errors.append("EXACT_CONTINUATION requires 'global_step'")

    return len(errors) == 0, errors


def quarantine_checkpoint(
    path: os.PathLike | str,
    quarantine_dir: os.PathLike | str,
    reason: str,
) -> Path:
    """Safely isolate a corrupted, collapsed, or stale checkpoint.

    Moves the checkpoint file into quarantine_dir and writes a sidecar JSON
    recording the original path, SHA-256 hash, quarantine reason, and UTC timestamp.

    Args:
        path: Path to corrupted/stale checkpoint.
        quarantine_dir: Destination quarantine folder.
        reason: Justification for quarantine (e.g., 'Q-value divergence', 'corrupt weights').

    Returns:
        Path to the quarantined file.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint to quarantine not found: {src}")

    dst_dir = Path(quarantine_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Compute SHA-256 before moving
    sha256 = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    file_sha = sha256.hexdigest()

    dst_file = dst_dir / src.name
    if dst_file.exists():
        timestamp_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        dst_file = dst_dir / f"{src.stem}_{timestamp_slug}{src.suffix}"

    shutil.move(str(src), str(dst_file))

    sidecar = {
        "quarantined_file": dst_file.name,
        "original_path": str(src.resolve()),
        "sha256": file_sha,
        "reason": reason,
        "quarantined_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    sidecar_path = dst_dir / f"{dst_file.name}.quarantine_meta.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    logger.warning("Checkpoint quarantined: %s -> %s (Reason: %s)", src, dst_file, reason)
    return dst_file


def migrate_checkpoint_to_hardened(
    src_path: os.PathLike | str,
    dst_path: os.PathLike | str,
    mode: CheckpointMode | str = CheckpointMode.WEIGHTS_ONLY,
    arch_config: dict[str, Any] | None = None,
    norm_stats: dict[str, Any] | None = None,
) -> Path:
    """Migrate an existing or legacy checkpoint into the hardened 2.0 contract format."""
    mode = CheckpointMode(mode)
    src = Path(src_path)
    dst = Path(dst_path)

    raw = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        target_state_dict = raw.get("target_state_dict")
        metadata = raw.get("metadata", {})
        arch_config = arch_config or raw.get("arch_config", {})
        norm_stats = norm_stats or raw.get("norm_stats", {})
        global_step = raw.get("global_step", 0)
        episode = raw.get("episode", 0)
        epsilon = raw.get("epsilon")
        opt_state = raw.get("optimizer_state_dict")
    elif isinstance(raw, dict):
        state_dict = raw
        target_state_dict = None
        metadata = {}
        arch_config = arch_config or {}
        norm_stats = norm_stats or {}
        global_step = 0
        episode = 0
        epsilon = None
        opt_state = None
    else:
        raise ValueError(f"Cannot migrate checkpoint of type {type(raw)}")

    payload: dict[str, Any] = {
        "checkpoint_mode": mode.value,
        "schema_version": "2.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_revision": current_git_revision(),
        "state_dict": state_dict,
        "target_state_dict": target_state_dict,
        "arch_config": dict(arch_config or {}),
        "norm_stats": dict(norm_stats or {}),
        "metadata": dict(metadata or {}),
    }

    if mode == CheckpointMode.WEIGHTS_ONLY:
        payload["global_step"] = 0
        payload["episode"] = 0
        payload["epsilon"] = None
        payload["optimizer_state_dict"] = None
    else:
        payload["global_step"] = int(global_step)
        payload["episode"] = int(episode)
        payload["epsilon"] = float(epsilon) if epsilon is not None else None
        payload["optimizer_state_dict"] = opt_state

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    logger.info("Migrated checkpoint %s -> %s under mode %s", src, dst, mode.value)
    return dst
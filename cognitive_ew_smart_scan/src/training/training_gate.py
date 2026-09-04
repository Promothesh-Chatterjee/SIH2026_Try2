"""Preflight checks that prevent invalid strict TSRD training runs."""

from __future__ import annotations

from pathlib import Path

from ..data.tsrd_manifest import resolve_split_dirs


def validate_training_gate(
    *,
    data_root: str | Path,
    deinterleaver_checkpoint: str | Path,
    normalization_stats: str | Path,
    environment_config: dict,
) -> list[str]:
    """Return blocking reasons for a strict real-TSRD scheduler run."""
    root = Path(data_root)
    errors: list[str] = []
    if not root.exists():
        errors.append(f"TSRD root does not exist: {root}")

    for mode in ("stare", "scan"):
        splits = resolve_split_dirs(root, mode)
        for split in ("train", "val"):
            directory = splits[split]
            if not directory.is_dir() or not any(directory.glob("*.h5")):
                errors.append(f"Missing {mode}/{split} .h5 files: {directory}")

    checkpoint = Path(deinterleaver_checkpoint)
    if not checkpoint.is_file():
        errors.append(f"Deinterleaver checkpoint missing: {checkpoint}")
    stats = Path(normalization_stats)
    if not stats.is_file():
        errors.append(f"Normalization statistics missing: {stats}")

    n_bands = int(environment_config.get("n_bands", 36))
    band_features = int(environment_config.get("band_features", 10))
    obs_dim = int(environment_config.get("obs_dim", n_bands * band_features))
    if (n_bands, band_features, obs_dim) != (36, 10, 360):
        errors.append(
            f"Observation contract invalid: n_bands={n_bands}, "
            f"band_features={band_features}, obs_dim={obs_dim}"
        )

    receiver_values = {
        "ibw_mhz": float(environment_config.get("ibw_mhz", 0.0)),
        "frequency_step_mhz": float(environment_config.get("frequency_step_mhz", 0.0)),
        "freq_min_mhz": float(environment_config.get("freq_min_mhz", 0.0)),
        "freq_max_mhz": float(environment_config.get("freq_max_mhz", 0.0)),
    }
    if receiver_values != {
        "ibw_mhz": 500.0,
        "frequency_step_mhz": 500.0,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
    }:
        errors.append(f"Receiver configuration invalid: {receiver_values}")
    return errors


def require_training_gate(**kwargs: object) -> None:
    """Raise before training if any strict TSRD prerequisite is missing."""
    errors = validate_training_gate(**kwargs)  # type: ignore[arg-type]
    if errors:
        raise RuntimeError("NOT_READY: strict TSRD training gate failed:\n- " + "\n- ".join(errors))
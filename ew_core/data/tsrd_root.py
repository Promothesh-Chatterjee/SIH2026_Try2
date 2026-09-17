"""Canonical TSRD dataset root resolution and split-layout contract.

Single source of truth for the location and layout of the official Turing
Synthetic Radar Dataset. Two responsibilities live here:

1. **Root resolution** — ``resolve_tsrd_root`` / ``resolve_config_data_dir``,
   with precedence:

        CLI override  >  env ``TSRD_DATA_ROOT``  >  training YAML ``data_dir``
        >  safe repository default ``data``

2. **Split layout contract** — the canonical dataset is ``<root>/<mode>/<
   split>/`` but the Kaggle download uses ``train_scan`` / ``val_scan`` /
   ``test_scan``-(style) names. ``split_candidate_dirs`` / ``resolve_split_dir``
   recognise every supported alias transparently:

    - official/Kaggle:      ``<root>/<mode>/<split>_<mode>`` (e.g. ``train_scan``)
    - conventional:         ``<root>/<mode>/{train,val,validation,test}``
    - archive layout:       ``<root>/archive/{train,validation,test}``
    - flat / nested roots:  ``<root>/*`` and ``<root>/<mode>/*`` fallbacks

Paths are handled via ``pathlib.Path`` so Windows/Linux separators normalize
identically. No developer-specific path is hard-coded: the default is the
relative, repo-local ``data``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Environment variable consulted when neither CLI nor config provides a root.
TSRD_ROOT_ENV = "TSRD_DATA_ROOT"
# Safe relative default (non-Windows, repo-local, used only as a last resort).
DEFAULT_DATA_DIR = "data"
# Canonical split keys returned by the split resolvers.
SPLIT_KEYS = ("train", "val", "test")


def resolve_tsrd_root(
    cli_value: str | os.PathLike | None = None,
    config: dict | None = None,
) -> Path:
    """Resolve the canonical dataset root from CLI, env, config, then default."""
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get(TSRD_ROOT_ENV)
    if env_value:
        return Path(env_value)
    if config:
        data_dir = config.get("data_dir")
        if data_dir:
            return Path(str(data_dir))
    return Path(DEFAULT_DATA_DIR)


def resolve_config_data_dir(
    training_config_path: str | os.PathLike,
    cli_value: str | os.PathLike | None = None,
) -> Path:
    """Load a training config's ``data_dir`` and resolve the canonical root.

    Args:
        training_config_path: Path to ``training_config.yaml``.
        cli_value: Optional CLI override (highest precedence).

    Returns:
        Resolved dataset root.
    """
    with open(training_config_path) as f:
        config = yaml.safe_load(f)
    return resolve_tsrd_root(cli_value=cli_value, config=config)


def split_candidate_dirs(
    data_root: str | os.PathLike,
    mode: str = "scan",
    mode_root: Path | None = None,
) -> dict[str, list[Path]]:
    """Return, per canonical split key, the directory candidates tried in order.

    Candidates cover the Kaggle aliases (``train_scan``), the conventional
    names (``train`` / ``val`` / ``validation`` / ``test``), the archive
    layout and flat/nested fallbacks. Only the *first* existing candidate is
    used by callers.

    Args:
        data_root: Dataset root.
        mode: World/observation mode (``stare`` or ``scan``).
        mode_root: Optional override of ``<root>/<mode>`` (kept for legacy
            callers that resolve the mode prefix differently, e.g. ``None``).

    Returns:
        Dict mapping ``train`` / ``val`` / ``test`` to ordered candidate paths.
    """
    root = Path(data_root)
    mode_root = mode_root if mode_root is not None else root / mode
    return {
        "train": [
            mode_root / f"train_{mode}",
            mode_root / "train",
            root / f"train_{mode}",
            root / "train",
            root / mode / f"train_{mode}",
            root / mode / "train",
            root / "archive" / "train",
        ],
        "val": [
            mode_root / f"val_{mode}",
            mode_root / f"validation_{mode}",
            mode_root / "val",
            mode_root / "validation",
            root / f"val_{mode}",
            root / "val",
            root / "validation",
            root / mode / f"val_{mode}",
            root / mode / "val",
            root / "archive" / "validation",
        ],
        "test": [
            mode_root / f"test_{mode}",
            mode_root / "test",
            root / f"test_{mode}",
            root / "test",
            root / mode / f"test_{mode}",
            root / mode / "test",
            root / "archive" / "test",
        ],
    }


def resolve_split_dir(
    data_root: str | os.PathLike, mode: str, split: str
) -> Path:
    """Resolve one canonical split directory for the given mode.

    Returns the first existing candidate per ``split_candidate_dirs``;
    otherwise falls back to ``<root>/<mode>/<split>_<mode>`` (Kaggle name) or
    ``<root>/<mode>/<split>``. Never creates directories.

    Args:
        data_root: Dataset root.
        mode: ``stare`` or ``scan``.
        split: Canonical key ``train`` | ``val`` | ``test``.

    Returns:
        Resolved directory path.
    """
    root = Path(data_root)
    if split not in SPLIT_KEYS:
        raise ValueError(f"split must be one of {SPLIT_KEYS}, got {split!r}")
    mode_root = root / mode
    for candidate in split_candidate_dirs(data_root, mode)[split]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    kaggle = mode_root / f"{split}_{mode}"
    return kaggle if kaggle.exists() and kaggle.is_dir() else mode_root / split
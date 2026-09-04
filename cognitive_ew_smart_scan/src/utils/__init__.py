"""Shared utilities (checkpoint metadata, provenance)."""

from .checkpoint_meta import build_train_metadata, save_state

__all__ = ["build_train_metadata", "save_state"]
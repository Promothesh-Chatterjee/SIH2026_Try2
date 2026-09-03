"""Telemetry package: run management and live metric broadcasting.

Provides ``RunManager`` for reproducible experiment directories and
``TelemetryPublisher`` for streaming / persisting real (never fabricated)
scheduler and detector metrics during training and evaluation.
"""

from __future__ import annotations

from .discovery import find_latest_run, latest_telemetry_history, latest_telemetry_snapshot
from .publisher import TelemetryPublisher
from .run_manager import RunManager

__all__ = ["RunManager", "TelemetryPublisher", "find_latest_run", "latest_telemetry_history", "latest_telemetry_snapshot"]
"""Discovery helpers: locate and read the most recent persisted run telemetry.

Although the FastAPI service usually shares an in-process
:class:`TelemetryPublisher`, training can also run as a separate process that
persists telemetry to ``runs/<run_id>/telemetry.jsonl``. These helpers let the
API serve that real (recorded) data even when no live in-process run is attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_latest_run(root: str | Path = "runs") -> Path | None:
    """Return the path to the most recently modified run directory, or None.

    Args:
        root: Parent directory holding ``runs/<run_id>`` folders.

    Returns:
        Path of the newest run dir, or None if no run exists.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    run_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs[0]


def latest_telemetry_snapshot(root: str | Path = "runs") -> dict[str, Any]:
    """Return the last telemetry record from the newest run, with a live flag.

    Never fabricates metrics: if no run or no telemetry exists, returns
    ``{"live": False, "live_message": ...}``.

    Args:
        root: Parent directory holding ``runs/<run_id>`` folders.

    Returns:
        Snapshot dict (``live`` + raw last record fields) or a not-live marker.
    """
    run_dir = find_latest_run(root)
    if run_dir is None:
        return {"live": False, "live_message": "no runs found yet", "run_id": None}
    telemetry = run_dir / "telemetry.jsonl"
    if not telemetry.exists():
        return {"live": False, "live_message": "run exists but no telemetry recorded", "run_id": run_dir.name}
    last_record: dict[str, Any] | None = None
    with telemetry.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last_record = json.loads(line)
            except json.JSONDecodeError:
                continue
    if last_record is None:
        return {"live": False, "live_message": "no telemetry records in file", "run_id": run_dir.name}
    return {"live": True, "run_id": run_dir.name, **last_record}


def latest_telemetry_history(root: str | Path = "runs", limit: int = 200) -> list[dict[str, Any]]:
    """Return the most recent telemetry records (newest first) from the newest run.

    Args:
        root: Parent directory holding ``runs/<run_id>`` folders.
        limit: Maximum number of records to return.

    Returns:
        List of parsed JSON records, newest first ([] if none).
    """
    run_dir = find_latest_run(root)
    if run_dir is None:
        return []
    telemetry = run_dir / "telemetry.jsonl"
    if not telemetry.exists():
        return []
    records: list[dict[str, Any]] = []
    with telemetry.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.reverse()
    return records[: int(limit)]
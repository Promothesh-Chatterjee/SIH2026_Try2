"""Telemetry publisher: accumulates and streams real (never fabricated) metrics.

The publisher is the single source of truth for what the live dashboard may
display. It only ever stores values that were explicitly recorded by the
training/evaluation loop via :meth:`update`. When no measurement has been
recorded, ``latest()`` returns ``{"live": False}`` so UIs render an explicit
"no live data" state instead of invented numbers (Rule: metrics must never be
fabricated).

Persistence is delegated to a :class:`src.telemetry.run_manager.RunManager`
when supplied; otherwise the publisher is a transient live-only broker used by
the FastAPI websocket/dashboard path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from .run_manager import RunManager


class TelemetryPublisher:
    """Accumulates and broadcasts real scheduler/detector telemetry.

    Args:
        run: Optional :class:`RunManager` for disk persistence.
        max_history: Max number of snapshots kept in memory for charting.
    """

    def __init__(self, run: RunManager | None = None, max_history: int = 500) -> None:
        self.run = run
        self.max_history = int(max_history)
        self._lock = threading.Lock()
        # Semantic counter: strictly incremented only by update(); 0 == never updated.
        self._n_updates: int = 0
        self._last: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._loop_guard = threading.Lock()

    @property
    def live(self) -> bool:
        """True once at least one real telemetry update has been recorded."""
        with self._lock:
            return self._n_updates > 0

    def update(self, step: int = 0, **fields: Any) -> None:
        """Record a telemetry snapshot.

        Args:
            step: Global training/eval step (for x-axis plotting).
            **fields: Named real metrics (scalars, band-priority vector, pdws…).
        """
        record = {"step": int(step), "update": time.time()}
        record.update(fields)
        with self._lock:
            self._n_updates += 1
            record["n_updates"] = self._n_updates
            self._last = dict(record)
            self._history.append(dict(record))
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]
        if self.run is not None:
            self.run.emit(step=record["step"], **fields)

    def latest(self) -> dict[str, Any]:
        """Return the latest snapshot with a ``live`` flag (never fabricated)."""
        with self._lock:
            if self._n_updates == 0:
                return {"live": False, "live_message": "no live telemetry yet"}
            return {"live": True, **dict(self._last)}

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the in-memory snapshot history (most recent first)."""
        with self._lock:
            data = list(reversed(self._history))
        if limit is not None:
            data = data[: int(limit)]
        return data

    # ── Async broadcasting (for FastAPI websockets) ──────────────────────

    async def subscribe(self) -> asyncio.Queue:
        """Register a subscriber queue (websocket) for live snapshots."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Deregister a subscriber queue."""
        with self._lock:
            self._subscribers.discard(queue)

    async def broadcast(self) -> None:
        """Push the latest snapshot to all subscriber queues (non-blocking).

        Subscribers whose queue is full (slow consumer) are dropped to avoid
        back-pressure stalling the telemetry loop.
        """
        try:
            payload = self.latest()
        except Exception:
            return
        subs: list[asyncio.Queue] = []
        with self._lock:
            subs = list(self._subscribers)
        for queue in subs:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def drain(self, interval_seconds: float = 0.25) -> None:
        """Background task: periodically broadcast latest snapshots for 4 Hz.

        Args:
            interval_seconds: Broadcast period.
        """
        while True:
            await self.broadcast()
            await asyncio.sleep(interval_seconds)
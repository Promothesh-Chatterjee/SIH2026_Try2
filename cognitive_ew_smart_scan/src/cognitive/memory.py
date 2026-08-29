"""
Semantic + Episodic Memory for Cognitive EW.

SemanticMemory: SQLite with spec schema (emitter_id PK, mean_pri, freq bounds, priority, periodic).
EpisodicMemory: Thin wrapper around LSTM hidden state.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EmitterProfile:
    """Emitter profile for SemanticMemory.

    Attributes:
        emitter_id: Unique emitter identifier (TEXT PK).
        mean_pri_us: Mean PRI (µs).
        freq_min_mhz: Lower frequency bound.
        freq_max_mhz: Upper frequency bound.
        mean_pw_us: Mean pulse width.
        aoa_mean: Mean AoA (degrees).
        amplitude_mean: Mean amplitude (dB).
        priority_score: Priority in [0,1].
        is_periodic: Whether periodic scan detected.
        scan_period_us: Estimated scan period (µs) if periodic.
        intercept_count: Number of intercepts.
        last_seen_us: Last ToA seen (µs).
    """

    emitter_id: str
    mean_pri_us: float = 0.0
    freq_min_mhz: float = 0.0
    freq_max_mhz: float = 0.0
    mean_pw_us: float = 0.0
    aoa_mean: float = 0.0
    amplitude_mean: float = 0.0
    priority_score: float = 0.5
    is_periodic: int = 0
    scan_period_us: float | None = None
    intercept_count: int = 0
    last_seen_us: float = 0.0


class SemanticMemory:
    """SQLite-backed emitter knowledge base.

    Schema: emitter_id TEXT PK, mean_pri_us REAL, freq_min_mhz REAL,
            freq_max_mhz REAL, mean_pw_us REAL, aoa_mean REAL,
            amplitude_mean REAL, priority_score REAL, is_periodic INTEGER,
            scan_period_us REAL, intercept_count INTEGER, last_seen_us REAL
    """

    def __init__(self, db_path: str | Path = "data/semantic_memory.db") -> None:
        """Initialise DB, creating tables if needed.

        Args:
            db_path: SQLite file path.
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("SemanticMemory at %s", db_path)

    def _create_tables(self) -> None:
        """Create emitter table per spec."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS emitters (
                emitter_id TEXT PRIMARY KEY,
                mean_pri_us REAL,
                freq_min_mhz REAL,
                freq_max_mhz REAL,
                mean_pw_us REAL,
                aoa_mean REAL,
                amplitude_mean REAL,
                priority_score REAL,
                is_periodic INTEGER,
                scan_period_us REAL,
                intercept_count INTEGER,
                last_seen_us REAL
            )
        """)
        self.conn.commit()

    def write_emitter(self, profile: EmitterProfile) -> None:
        """Insert or replace emitter profile.

        Args:
            profile: EmitterProfile to upsert.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO emitters
            (emitter_id, mean_pri_us, freq_min_mhz, freq_max_mhz, mean_pw_us,
             aoa_mean, amplitude_mean, priority_score, is_periodic, scan_period_us,
             intercept_count, last_seen_us)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.emitter_id, profile.mean_pri_us, profile.freq_min_mhz, profile.freq_max_mhz,
            profile.mean_pw_us, profile.aoa_mean, profile.amplitude_mean, profile.priority_score,
            int(profile.is_periodic), profile.scan_period_us, profile.intercept_count, profile.last_seen_us,
        ))
        self.conn.commit()
        logger.debug("Upsert emitter %s", profile.emitter_id)

    # Backward compat alias for older code
    def upsert_emitter(self, *args, **kwargs) -> None:  # type: ignore
        """Legacy upsert_emitter wrapper (translates old session_id/cluster_id)."""
        # If called with old signature, try to map
        if args and isinstance(args[0], str) and len(args) >= 2:
            # Heuristic: create an EmitterProfile from positional args
            try:
                session_id = args[0]
                cluster_id = args[1] if len(args) > 1 else 0
                emitter_id = f"{session_id}_{cluster_id}"
                cf_mean = float(kwargs.get("cf_mean", args[2] if len(args) > 2 else 0))
                pw_mean = float(kwargs.get("pw_mean", args[3] if len(args) > 3 else 0))
                pri_mean = float(kwargs.get("pri_mean", args[4] if len(args) > 4 else 0))
                prof = EmitterProfile(
                    emitter_id=emitter_id, mean_pri_us=pri_mean,
                    freq_min_mhz=cf_mean, freq_max_mhz=cf_mean,
                    mean_pw_us=pw_mean, intercept_count=int(kwargs.get("n_pulses", 0)),
                    last_seen_us=time.time() * 1e6,
                )
                self.write_emitter(prof)
                return
            except Exception:
                pass
        # fallback: if EmitterProfile passed
        if args and isinstance(args[0], EmitterProfile):
            self.write_emitter(args[0])

    def get_emitter(self, emitter_id: str) -> EmitterProfile | None:
        """Fetch one emitter by ID.

        Args:
            emitter_id: Primary key.

        Returns:
            EmitterProfile or None.
        """
        row = self.conn.execute("SELECT * FROM emitters WHERE emitter_id=?", (emitter_id,)).fetchone()
        if row is None:
            return None
        return EmitterProfile(
            emitter_id=row["emitter_id"], mean_pri_us=row["mean_pri_us"] or 0,
            freq_min_mhz=row["freq_min_mhz"] or 0, freq_max_mhz=row["freq_max_mhz"] or 0,
            mean_pw_us=row["mean_pw_us"] or 0, aoa_mean=row["aoa_mean"] or 0,
            amplitude_mean=row["amplitude_mean"] or 0, priority_score=row["priority_score"] or 0.5,
            is_periodic=int(row["is_periodic"] or 0), scan_period_us=row["scan_period_us"],
            intercept_count=int(row["intercept_count"] or 0), last_seen_us=row["last_seen_us"] or 0,
        )

    def list_emitters(self) -> list[EmitterProfile]:
        """List all emitters.

        Returns:
            List of EmitterProfiles.
        """
        rows = self.conn.execute("SELECT * FROM emitters").fetchall()
        out: list[EmitterProfile] = []
        for row in rows:
            out.append(EmitterProfile(
                emitter_id=row["emitter_id"], mean_pri_us=row["mean_pri_us"] or 0,
                freq_min_mhz=row["freq_min_mhz"] or 0, freq_max_mhz=row["freq_max_mhz"] or 0,
                mean_pw_us=row["mean_pw_us"] or 0, aoa_mean=row["aoa_mean"] or 0,
                amplitude_mean=row["amplitude_mean"] or 0, priority_score=row["priority_score"] or 0.5,
                is_periodic=int(row["is_periodic"] or 0), scan_period_us=row["scan_period_us"],
                intercept_count=int(row["intercept_count"] or 0), last_seen_us=row["last_seen_us"] or 0,
            ))
        return out

    def query_by_session(self, session_id: str) -> list[dict]:
        """Legacy query by session prefix (for notebook compat).

        Args:
            session_id: Prefix to filter emitter_id.

        Returns:
            List of dicts with keys matching old fingerprint schema.
        """
        rows = self.conn.execute("SELECT * FROM emitters WHERE emitter_id LIKE ?", (f"{session_id}%",)).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["centroid"] = np.zeros(64, dtype=np.float32)  # placeholder
            out.append(d)
        return out

    def get_band_priority_boost(self, n_bands: int, freq_min: float, freq_max: float) -> np.ndarray:
        """Return n_bands priority boost vector from known emitter frequencies.

        Boost = priority_score if emitter's freq range overlaps band, else 0.
        Feeds into MoE as third source. Max over overlapping emitters per band.

        Args:
            n_bands: Number of bands.
            freq_min: Receiver min freq (MHz).
            freq_max: Receiver max freq (MHz).

        Returns:
            Array (n_bands,) float32 in [0,1].
        """
        boost = np.zeros(n_bands, dtype=np.float32)
        band_width = (freq_max - freq_min) / n_bands
        band_edges = np.linspace(freq_min, freq_max, n_bands + 1)
        for prof in self.list_emitters():
            # Find overlapping bands
            lo = max(freq_min, prof.freq_min_mhz)
            hi = min(freq_max, prof.freq_max_mhz if prof.freq_max_mhz > prof.freq_min_mhz else prof.freq_min_mhz + 1.0)
            if hi <= lo:
                continue
            b_lo = int(np.clip(np.floor((lo - freq_min) / band_width), 0, n_bands - 1))
            b_hi = int(np.clip(np.ceil((hi - freq_min) / band_width), 0, n_bands))
            for b in range(b_lo, b_hi):
                boost[b] = max(boost[b], float(prof.priority_score))
        return boost

    def update_priority(self, emitter_id: str, intercept_occurred: bool) -> None:
        """Decay priority if repeatedly missed, boost if intercepted.

        Args:
            emitter_id: PK.
            intercept_occurred: Whether this emitter was intercepted this dwell.
        """
        prof = self.get_emitter(emitter_id)
        if prof is None:
            logger.warning("update_priority unknown emitter %s", emitter_id)
            return
        if intercept_occurred:
            prof.priority_score = min(1.0, prof.priority_score * 1.1 + 0.05)
            prof.intercept_count += 1
        else:
            prof.priority_score = max(0.0, prof.priority_score * 0.95 - 0.02)
        prof.last_seen_us = time.time() * 1e6
        self.write_emitter(prof)
        logger.debug("Priority %s -> %.3f", emitter_id, prof.priority_score)

    def close(self) -> None:
        """Close DB connection."""
        try:
            self.conn.close()
        except Exception:
            pass


class EpisodicMemory:
    """Thin wrapper around LSTM hidden state for episode management.

    Mirrors DRQN's hidden state as explicit episodic context.
    """

    def __init__(self) -> None:
        """Initialise with no state."""
        self._state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        """Clear hidden state at episode start."""
        self._state = None

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return current hidden state.

        Returns:
            Tuple (h, c) or None if not set.
        """
        return self._state

    def set_state(self, state: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        """Set hidden state.

        Args:
            state: Tuple (h,c) or None to clear.
        """
        self._state = state

    # Extended API for buffer-style usage
    def record(self, obs: np.ndarray, action: int, hit: bool, reward: float) -> None:
        """No-op record for compat (state is managed via set_state)."""
        pass

    def get_recent(self, n: int) -> list[dict]:
        """Return empty (state-based, not buffer)."""
        return []

    def hit_rate(self, n: int = 64) -> float:
        """Return 0 (override if buffer needed)."""
        return 0.0

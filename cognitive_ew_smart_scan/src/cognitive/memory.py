"""
Cognitive Memory Module: Episodic + Semantic Memory.

Provides two tiers of memory:
  - SemanticMemory: SQLite-backed persistent store of learned emitter fingerprints.
  - EpisodicMemory: In-memory ring buffer of recent (obs, action, outcome) tuples
    mirroring the LSTM hidden state's temporal context.
"""

import sqlite3
import json
import logging
import time
from pathlib import Path
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    SQLite-backed store for learned emitter fingerprints.

    Each entry captures the characterised properties of a distinct emitter 
    cluster identified during deinterleaving, including its average CF, PW, PRI,
    and the embedding centroid for fast similarity queries.
    """

    def __init__(self, db_path: str = "data/semantic_memory.db") -> None:
        """
        Initializes the SemanticMemory database.

        Args:
            db_path: File path for the SQLite database.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        logger.info(f"SemanticMemory initialized at {db_path}")

    def _create_tables(self) -> None:
        """Creates the emitter fingerprint table if it does not exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS emitter_fingerprints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                cluster_id  INTEGER NOT NULL,
                cf_mean_mhz REAL,
                pw_mean_us  REAL,
                pri_mean_us REAL,
                pri_std_us  REAL,
                n_pulses    INTEGER,
                centroid    TEXT,   -- JSON-encoded float list
                last_seen   REAL,   -- Unix timestamp
                UNIQUE(session_id, cluster_id)
            )
        """)
        self.conn.commit()

    def upsert_emitter(
        self,
        session_id: str,
        cluster_id: int,
        cf_mean: float,
        pw_mean: float,
        pri_mean: float,
        pri_std: float,
        n_pulses: int,
        centroid: np.ndarray,
    ) -> None:
        """
        Inserts or updates a known emitter fingerprint.

        Args:
            session_id: Unique identifier for the current mission/session.
            cluster_id: File-local cluster label from HDBSCAN.
            cf_mean: Mean centre frequency (MHz).
            pw_mean: Mean pulse width (µs).
            pri_mean: Estimated mean PRI (µs).
            pri_std: Standard deviation of estimated PRI (µs).
            n_pulses: Number of pulses in the cluster.
            centroid: Embedding centroid as a float numpy array.
        """
        centroid_json = json.dumps(centroid.tolist())
        self.conn.execute("""
            INSERT INTO emitter_fingerprints
                (session_id, cluster_id, cf_mean_mhz, pw_mean_us, pri_mean_us,
                 pri_std_us, n_pulses, centroid, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, cluster_id) DO UPDATE SET
                cf_mean_mhz = excluded.cf_mean_mhz,
                pw_mean_us  = excluded.pw_mean_us,
                pri_mean_us = excluded.pri_mean_us,
                pri_std_us  = excluded.pri_std_us,
                n_pulses    = excluded.n_pulses,
                centroid    = excluded.centroid,
                last_seen   = excluded.last_seen
        """, (session_id, cluster_id, cf_mean, pw_mean, pri_mean, pri_std,
              n_pulses, centroid_json, time.time()))
        self.conn.commit()

    def query_by_session(self, session_id: str) -> list[dict]:
        """
        Returns all stored emitter fingerprints for a given session.

        Args:
            session_id: Mission/session identifier.

        Returns:
            List of emitter fingerprint dictionaries.
        """
        cursor = self.conn.execute(
            "SELECT * FROM emitter_fingerprints WHERE session_id = ?", (session_id,)
        )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        for row in rows:
            row["centroid"] = np.array(json.loads(row["centroid"]), dtype=np.float32)
        return rows

    def close(self) -> None:
        """Closes the database connection."""
        self.conn.close()


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    In-memory ring buffer storing recent (obs, action, hit, reward) tuples
    that complement the LSTM hidden state as explicit episodic context.
    """

    def __init__(self, max_len: int = 512) -> None:
        """
        Initializes the episodic memory.

        Args:
            max_len: Maximum number of recent steps to retain.
        """
        self._buffer: deque[dict] = deque(maxlen=max_len)

    def record(self, obs: np.ndarray, action: int, hit: bool, reward: float) -> None:
        """
        Appends a single step to episodic memory.

        Args:
            obs: Observation vector at the recorded step.
            action: Band index chosen by the agent.
            hit: Whether the action intercepted any pulse.
            reward: Scalar reward received.
        """
        self._buffer.append({
            "obs":    obs.copy(),
            "action": action,
            "hit":    hit,
            "reward": reward,
        })

    def get_recent(self, n: int) -> list[dict]:
        """
        Returns the n most recent episodic records.

        Args:
            n: Number of most-recent steps to return.

        Returns:
            List of step dictionaries (up to n entries).
        """
        entries = list(self._buffer)
        return entries[-n:] if len(entries) >= n else entries

    def hit_rate(self, n: int = 64) -> float:
        """
        Computes the hit rate over the last n steps.

        Args:
            n: Window size.

        Returns:
            Float hit rate in [0.0, 1.0].
        """
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return sum(1 for e in recent if e["hit"]) / len(recent)

    def reset(self) -> None:
        """Clears the episodic buffer at the start of a new episode."""
        self._buffer.clear()

"""Scenario generator: build PulseRecord lists for CognitiveRFScanEnv.

Reads TSRD-style .h5 pulse trains (datasets "data" (n,5) = [toa, cf, pw, aoa,
amp] and "labels" (n,) emitter ids) into PulseRecord lists, or falls back to a
synthetic scenario when no real files are present. This keeps the receiver-driven
CognitiveRFScanEnv data source agnostic (TSRD or synthetic) and file-local
emitter-id scoped per file (labels must never be mixed across files).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .radio_environment import PulseRecord

logger = logging.getLogger(__name__)

SYN_COLUMNS = ["toa_us", "frequency_mhz", "pulse_width_us", "amplitude_db", "aoa_deg"]


def records_from_array(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    source_id: str = "tsrd",
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: Optional[float] = None,
) -> list[PulseRecord]:
    """Convert a pulse train array into PulseRecord objects.

    Args:
        data: Array shape (n, 5) with columns [toa, cf, pw, aoa, amp].
        labels: Array shape (n,) of per-pulse emitter ids (file-local).
        source_id: Source identifier stored on each record.
        freq_min_mhz: Lower spectral edge; pulses outside are clipped.
        freq_max_mhz: Upper spectral edge; pulses outside are clipped.
        time_horizon_us: If set, pulses with toa beyond this are dropped.

    Returns:
        List of PulseRecord, dropped pulses (out of band / beyond horizon) are
        excluded so the radio world only contains physically valid pulses.
    """
    records: list[PulseRecord] = []
    if data is None or data.size == 0:
        return records

    toa = np.asarray(data[:, 0], dtype=np.float64).flatten()
    cf = np.asarray(data[:, 1], dtype=np.float64).flatten()
    pw = np.asarray(data[:, 2], dtype=np.float64).flatten()
    aoa = np.asarray(data[:, 3], dtype=np.float64).flatten()
    amp = np.asarray(data[:, 4], dtype=np.float64).flatten()
    lbl = np.asarray(labels, dtype=np.int64).flatten() if labels is not None and labels.size else np.zeros(len(toa), dtype=np.int64)
    if len(lbl) < len(toa):
        lbl = np.concatenate([lbl, np.zeros(len(toa) - len(lbl), dtype=np.int64)])

    valid = np.isfinite(toa) & np.isfinite(cf)
    valid &= (cf >= freq_min_mhz) & (cf <= freq_max_mhz)
    if time_horizon_us is not None:
        valid &= toa <= float(time_horizon_us)

    for i in np.where(valid)[0]:
        records.append(
            PulseRecord(
                toa_us=float(toa[i]),
                frequency_mhz=float(cf[i]),
                pulse_width_us=float(pw[i]),
                amplitude_db=float(amp[i]),
                aoa_deg=float(aoa[i]),
                emitter_id=int(lbl[i]),
                source_id=source_id,
            )
        )
    return records


def load_h5_records(
    path: str | Path,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: Optional[float] = None,
    max_pulses: int = 50000,
) -> list[PulseRecord]:
    """Load a single TSRD-style .h5 file into PulseRecords (file-local labels).

    Args:
        path: Path to .h5 file containing "data" (n,5) and "labels" (n,).
        freq_min_mhz / freq_max_mhz: Spectral clipping range.
        time_horizon_us: Optional toa upper bound.
        max_pulses: Cap on pulses loaded (keeps episodes bounded).

    Returns:
        List of PulseRecord.
    """
    import h5py

    path = Path(path)
    with h5py.File(str(path), "r") as handle:
        data = handle["data"][:max_pulses]
        labels = handle["labels"][:max_pulses] if "labels" in handle else None
    records = records_from_array(data, labels, source_id=f"tsrd:{path.stem}")
    if not records:
        return records
    # Normalise ToA so the scenario starts at t=0. TSRD ToA are absolute relative
    # timestamps with an arbitrary per-file offset; only the relative spacing
    # matters for the receiver simulation. Keep the earliest pulse at 0.
    t0 = min(r.toa_us for r in records)
    for r in records:
        r.toa_us -= t0
    records.sort(key=lambda r: r.toa_us)
    return records


# Map our {mode}/{split} names onto the official TSRD repo directory naming.
# Official layout: <mode>/<mode>_<split>/config_*.h5  (e.g. scan/train_scan/).
# split aliases: train|train_scan|train_stare, val|validation|val_scan|val_stare,
#                test|test_scan|test_stare.
_SPLIT_DIR_ALIASES = {
    "train": {"train", "train_scan", "train_stare"},
    "val": {"val", "val_scan", "val_stare", "validation"},
    "test": {"test", "test_scan", "test_stare"},
    "validation": {"validation", "val", "val_scan", "val_stare"},
}


def _search_subdirs(data_root: Path, mode: str, split: str) -> list[Path]:
    """Search candidate subdirectories for the mode/split combo."""
    desired = _SPLIT_DIR_ALIASES.get(split, {split})
    base_dir = data_root / mode
    candidates: list[Path] = []
    if base_dir.exists():
        for sub in base_dir.iterdir():
            if sub.is_dir() and sub.name in desired:
                candidates.append(sub)
    # Also try data_root directly (some dumps flatten): data_root/<mode>/<split>
    plain = data_root / mode / split
    if plain.is_dir() and plain not in candidates:
        candidates.append(plain)
    files: list[Path] = []
    for cand in candidates:
        files.extend(sorted(cand.glob("*.h5")))
    return files


def discover_h5_files(data_root: str | Path, mode: str = "scan", subset: str = "train") -> list[Path]:
    """Discover .h5 files under a TSRD-compatible layout.

    Handles both the official ``<mode>/<mode>_<split>`` layout (e.g.
    ``scan/train_scan``) and the simpler ``<mode>/<split>`` layout.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    found = _search_subdirs(data_root, mode, subset)
    return sorted(set(found))


def synthetic_records(
    n_pulses: int = 800,
    n_emitters: int = 6,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: float = 50000.0,
    seed: int = 42,
) -> list[PulseRecord]:
    """Generate a synthetic, multi-emitter pulse scenario.

    Each emitter emits a contiguous train (PriT-staggered bursts) with a distinct
    frequency so the scheduler can learn band selectivity. Visible spans are spread
    across the time horizon so a scanning receiver must revisit over time.
    """
    rng = np.random.default_rng(seed)
    records: list[PulseRecord] = []
    total = 0
    for emitter_id in range(n_emitters):
        bursts = int(np.clip(3 + emitter_id, 2, 8))
        for b in range(bursts):
            cf = freq_min_mhz + (freq_max_mhz - freq_min_mhz) * (
                (emitter_id + 0.5) / n_emitters + rng.uniform(-0.03, 0.03)
            )
            cf = float(np.clip(cf, freq_min_mhz, freq_max_mhz))
            burst_start = time_horizon_us * ((b + rng.uniform(0.1, 0.9)) / bursts)
            pri = rng.uniform(300.0, 1500.0)
            n_pulses_here = int(np.clip(n_pulses // (n_emitters * bursts), 1, 40))
            pw = rng.uniform(0.5, 8.0)
            for k in range(n_pulses_here):
                toa = burst_start + k * pri
                if toa > time_horizon_us:
                    break
                records.append(
                    PulseRecord(
                        toa_us=float(toa),
                        frequency_mhz=cf,
                        pulse_width_us=pw,
                        amplitude_db=float(rng.uniform(-90, -50)),
                        aoa_deg=float(rng.uniform(-60, 60)),
                        emitter_id=emitter_id,
                        source_id="synthetic",
                    )
                )
                total += 1
    records.sort(key=lambda r: r.toa_us)
    return records


def build_scenario(
    data_root: str | Path | None = None,
    mode: str = "scan",
    subset: str = "train",
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: Optional[float] = None,
    max_pulses: int = 50000,
    seed: int = 42,
    allow_synthetic_fallback: bool = True,
) -> tuple[list[PulseRecord], str, list[Path]]:
    """Build a PulseRecord scenario from TSRD .h5 files.

    Args:
        data_root: Root data dir (contains mode/subset). If None or no .h5 files
            found, falls back to synthetic ONLY if allow_synthetic_fallback=True.
        mode / subset: Dataset split layout.
        freq_min_mhz / freq_max_mhz: Spectral clip range.
        time_horizon_us: Optional toa cap.
        max_pulses: Cap per file.
        seed: RNG seed for synthetic fallback.
        allow_synthetic_fallback: If False, raise FileNotFoundError when no TSRD data found.

    Returns:
        Tuple (records, source_label, file_paths_used).
        source_label is "tsrd" or "synthetic".

    Raises:
        FileNotFoundError: If no TSRD files found and allow_synthetic_fallback=False.
    """
    files: list[Path] = []
    if data_root is not None:
        files = discover_h5_files(data_root, mode=mode, subset=subset)

    if files:
        records: list[PulseRecord] = []
        for f in files:
            records.extend(
                load_h5_records(
                    f,
                    freq_min_mhz=freq_min_mhz,
                    freq_max_mhz=freq_max_mhz,
                    time_horizon_us=time_horizon_us,
                    max_pulses=max_pulses,
                )
            )
        logger.info("Scenario[tsrd]: %d pulses from %d file(s) in %s/%s", len(records), len(files), mode, subset)
        return records, "tsrd", files

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"No TSRD .h5 files found in {data_root}/{mode}/{subset}. "
            f"Set allow_synthetic_fallback=True to use synthetic data, or provide valid TSRD data."
        )

    records = synthetic_records(freq_min_mhz=freq_min_mhz, freq_max_mhz=freq_max_mhz, seed=seed)
    logger.warning("Scenario[synthetic]: %d pulses (no %s/%s .h5 found; using synthetic fallback)", len(records), mode, subset)
    return records, "synthetic", []


def build_world_scenario(
    data_root: str | Path,
    subset: str = "train",
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: Optional[float] = None,
    max_pulses: int = 50000,
    seed: int = 42,
) -> tuple[list[PulseRecord], str, list[Path]]:
    """Build RF world scenario from TSRD STARE data (latent truth).

    This is the ground-truth RF world that the scheduler's receiver observes through
    its limited IBW. Uses STARE mode exclusively - no synthetic fallback.

    Args:
        data_root: Root data dir containing stare/ subdirectories.
        subset: Dataset split (train/val/test).
        freq_min_mhz / freq_max_mhz: Spectral clip range.
        time_horizon_us: Optional toa cap.
        max_pulses: Cap per file.
        seed: RNG seed (unused for STARE, kept for interface consistency).

    Returns:
        Tuple (records, source_label, file_paths_used).
        source_label is always "tsrd_stare".

    Raises:
        FileNotFoundError: If no STARE .h5 files found.
    """
    return build_scenario(
        data_root=data_root,
        mode="stare",
        subset=subset,
        freq_min_mhz=freq_min_mhz,
        freq_max_mhz=freq_max_mhz,
        time_horizon_us=time_horizon_us,
        max_pulses=max_pulses,
        seed=seed,
        allow_synthetic_fallback=False,
    )


def build_observation_scenario(
    data_root: str | Path,
    subset: str = "train",
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    time_horizon_us: Optional[float] = None,
    max_pulses: int = 50000,
    seed: int = 42,
) -> tuple[list[PulseRecord], str, list[Path]]:
    """Build observation scenario from TSRD SCAN data (realistic observed data).

    This represents what a real narrowband receiver would observe. Used for:
    - Deinterleaver validation
    - Realistic observed-data comparison
    - Benchmark evaluation

    Args:
        data_root: Root data dir containing scan/ subdirectories.
        subset: Dataset split (train/val/test).
        freq_min_mhz / freq_max_mhz: Spectral clip range.
        time_horizon_us: Optional toa cap.
        max_pulses: Cap per file.
        seed: RNG seed (unused for SCAN, kept for interface consistency).

    Returns:
        Tuple (records, source_label, file_paths_used).
        source_label is always "tsrd_scan".

    Raises:
        FileNotFoundError: If no SCAN .h5 files found.
    """
    return build_scenario(
        data_root=data_root,
        mode="scan",
        subset=subset,
        freq_min_mhz=freq_min_mhz,
        freq_max_mhz=freq_max_mhz,
        time_horizon_us=time_horizon_us,
        max_pulses=max_pulses,
        seed=seed,
        allow_synthetic_fallback=False,
    )


class ScenarioSource:
    """Providers of per-episode PulseRecords from a local TSRD split.

    Each call to :meth:`sample` loads ONE randomly-chosen .h5 file's records
    (capped to ``max_pulses``, ToA-normalised) so that an RL episode gets a
    single, diverse, memory-bounded scenario — unlike concatenating every file.
    Falls back to a fresh synthetic scenario if no .h5 files are present.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        mode: str = "scan",
        subset: str = "train",
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 18000.0,
        time_horizon_us: Optional[float] = None,
        max_pulses: int = 50000,
        seed: int = 42,
        synthetic: bool = False,
        source_type: str = "observation",  # "world" (STARE) or "observation" (SCAN)
    ) -> None:
        self.freq_min_mhz = freq_min_mhz
        self.freq_max_mhz = freq_max_mhz
        self.time_horizon_us = time_horizon_us
        self.max_pulses = max_pulses
        self._rng = np.random.default_rng(seed)
        self.files: list[Path] = []
        self.source_type = source_type
        self.source_mode = "stare" if source_type == "world" else "scan"

        if data_root is not None and not synthetic:
            self.files = discover_h5_files(data_root, mode=self.source_mode, subset=subset)
        self.source_label = f"tsrd_{self.source_mode}" if self.files else "synthetic"
        if self.files:
            logger.info("ScenarioSource[%s]: %d files in %s/%s", self.source_label, self.source_mode, subset)
        else:
            if synthetic:
                logger.info("ScenarioSource[synthetic]: explicit synthetic mode")
            else:
                logger.warning("ScenarioSource[synthetic]: no %s/%s .h5 found — using synthetic fallback", self.source_mode, subset)

    def __len__(self) -> int:
        return len(self.files)

    def sample(self) -> list[PulseRecord]:
        """Return records for one episode (a single random file, or synthetic)."""
        if not self.files:
            return synthetic_records(
                freq_min_mhz=self.freq_min_mhz,
                freq_max_mhz=self.freq_max_mhz,
                seed=int(self._rng.integers(0, 2**31)),
            )
        fpath = Path(self._rng.choice(self.files))
        return load_h5_records(
            fpath,
            freq_min_mhz=self.freq_min_mhz,
            freq_max_mhz=self.freq_max_mhz,
            time_horizon_us=self.time_horizon_us,
            max_pulses=self.max_pulses,
        )
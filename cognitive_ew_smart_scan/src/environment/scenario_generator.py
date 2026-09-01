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
    return records_from_array(data, labels, source_id=f"tsrd:{path.stem}")


def discover_h5_files(data_root: str | Path, mode: str = "scan", subset: str = "train") -> list[Path]:
    """Discover .h5 files under data_root/mode/subset (the fixed path layout)."""
    base = Path(data_root) / mode / subset
    if not base.exists():
        return []
    return sorted(base.glob("*.h5"))


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
) -> tuple[list[PulseRecord], str, list[Path]]:
    """Build a PulseRecord scenario from TSRD .h5 files or a synthetic fallback.

    Args:
        data_root: Root data dir (contains mode/subset). If None or no .h5 files
            found, falls back to synthetic.
        mode / subset: Dataset split layout.
        freq_min_mhz / freq_max_mhz: Spectral clip range.
        time_horizon_us: Optional toa cap.
        max_pulses: Cap per file.
        seed: RNG seed for synthetic fallback.

    Returns:
        Tuple (records, source_label, file_paths_used).
        source_label is "tsrd" or "synthetic".
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
        logger.info("Scenario[tsrd]: %d pulses from %d file(s) in %s", len(records), len(files), Path(data_root) / mode / subset)
        return records, "tsrd", files

    records = synthetic_records(freq_min_mhz=freq_min_mhz, freq_max_mhz=freq_max_mhz, seed=seed)
    logger.info("Scenario[synthetic]: %d pulses (no %s/%s .h5 found)", len(records), mode, subset)
    return records, "synthetic", []
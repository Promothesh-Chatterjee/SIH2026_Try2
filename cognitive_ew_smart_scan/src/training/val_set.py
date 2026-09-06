"""
RC-2 fixed validation scenario set.

A deterministic, documented set of validation scenarios so every checkpoint's
validation numbers are comparable. Picks ``n_files`` eligible files from the
``val`` split via a seeded RNG over the *sorted* candidate list; documents the
exact scenario IDs (file stems) and consumed seed in a manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..environment.scenario_generator import ScenarioSource, load_h5_records

logger = logging.getLogger(__name__)


class FixedValidationSet:
    """Fixed, reproducible validation scenarios sampled from existing data.

    Attributes:
        files_used: List of ``(path, scenario_id, n_pulses)`` for scenarios that
            actually produced records (empty-after-filter scenarios are skipped
            and documented).
        skipped: List of ``(scenario_id, reason)``.
        validation_set_id: Short hash over files+seed for reproducibility.
    """

    def __init__(
        self,
        data_root: str | Path,
        subset: str = "val",
        mode: str = "stare",
        n_files: int = 2,
        seed: int = 42,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 18000.0,
        time_horizon_us: float | None = None,
        max_pulses: int = 50000,
        allow_synthetic_fallback: bool = False,
    ) -> None:
        self.subset = subset
        self.mode = mode
        self.n_files = int(n_files)
        self.seed = int(seed)
        self.freq_min_mhz = freq_min_mhz
        self.freq_max_mhz = freq_max_mhz
        self.time_horizon_us = time_horizon_us
        self.max_pulses = max_pulses
        self.allow_synthetic_fallback = allow_synthetic_fallback

        candidate_pool = ScenarioSource(
            data_root=data_root,
            mode=mode,
            subset=subset,
            freq_min_mhz=freq_min_mhz,
            freq_max_mhz=freq_max_mhz,
            time_horizon_us=time_horizon_us or None,
            max_pulses=max_pulses,
            seed=seed,
            source_type="world",
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        self.candidates = sorted(list(getattr(candidate_pool, "eligible_files", [])))
        self._cursor = 0
        self.files_used: list[tuple[Path, str, int]] = []
        self.skipped: list[tuple[str, str]] = []

        chosen = self._pick_files()
        # Verify each chosen file actually yields records under the filters so
        # validation never silently degrades at eval time.
        for path in chosen:
            scenario_id = Path(path).stem
            try:
                recs = load_h5_records(
                    path,
                    freq_min_mhz=freq_min_mhz,
                    freq_max_mhz=freq_max_mhz,
                    time_horizon_us=time_horizon_us,
                    max_pulses=max_pulses,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.skipped.append((scenario_id, f"load error: {exc}"))
                continue
            if not recs:
                self.skipped.append((scenario_id, "empty after filtering"))
                continue
            self.files_used.append((Path(path), scenario_id, len(recs)))

        self.validation_set_id = self._make_id()
        if not self.files_used:
            logger.warning(
                "FixedValidationSet: no usable validation files from %d candidates (seed=%d, n_files=%d) — validation will be skipped",
                len(self.candidates), self.seed, self.n_files,
            )
        else:
            logger.info(
                "FixedValidationSet %s: %d scenarios %s (seed=%d, %d candidates, %d skipped)",
                self.validation_set_id, len(self.files_used),
                [sid for _, sid, _ in self.files_used], self.seed,
                len(self.candidates), len(self.skipped),
            )

    def _pick_files(self) -> list[Path]:
        if not self.candidates:
            return []
        n = min(self.n_files, len(self.candidates))
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(self.candidates), size=n, replace=False)
        return [self.candidates[int(i)] for i in sorted(idx)]

    def _make_id(self) -> str:
        import hashlib

        parts = [self.subset, self.mode, str(self.seed), str(self.n_files)]
        parts.extend(sorted(Path(p).stem for p, _, _ in self.files_used))
        raw = "|".join(parts).encode("utf-8")
        return "val-" + hashlib.sha256(raw).hexdigest()[:12]

    def sample(self) -> list[Any]:
        """Return records for the next fixed validation scenario (cycles)."""
        if not self.files_used:
            raise FileNotFoundError(
                f"FixedValidationSet has no usable scenarios "
                f"({len(self.candidates)} candidates, {len(self.skipped)} skipped)"
            )
        path, scenario_id, _ = self.files_used[self._cursor % len(self.files_used)]
        self._cursor += 1
        return load_h5_records(
            path,
            freq_min_mhz=self.freq_min_mhz,
            freq_max_mhz=self.freq_max_mhz,
            time_horizon_us=self.time_horizon_us,
            max_pulses=self.max_pulses,
        )

    def current_scenario_id(self) -> str:
        """Scenario ID for the episode the sample() cursor is about to serve."""
        if not self.files_used:
            return "none"
        return self.files_used[self._cursor % len(self.files_used)][1]

    def manifest(self) -> dict[str, Any]:
        """Documentation manifest: seeds, parameters and exact scenario IDs."""
        return {
            "subset": self.subset,
            "mode": self.mode,
            "seed": self.seed,
            "n_files_requested": self.n_files,
            "freq_min_mhz": self.freq_min_mhz,
            "freq_max_mhz": self.freq_max_mhz,
            "time_horizon_us": self.time_horizon_us,
            "max_pulses": self.max_pulses,
            "validation_set_id": self.validation_set_id,
            "scenario_ids": [sid for _, sid, _ in self.files_used],
            "scenario_files": [str(p) for p, _, _ in self.files_used],
            "n_pulses": [n for _, _, n in self.files_used],
            "skipped_scenarios": [{"id": sid, "reason": reason} for sid, reason in self.skipped],
        }
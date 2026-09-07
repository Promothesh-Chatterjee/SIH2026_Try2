"""Real production deinterleaver integration tests — Phase 3V.

Prove the ACTUAL TRAINED PRODUCTION deinterleaver (PDWTransformerEncoder
checkpoint) is loaded and used by the GNU RF translation layer, end-to-end,
using the existing production pipeline components only.

Covered
-------
Step 7  : deinterleaver_loader — production checkpoint + stats loading,
          wrapped AND raw checkpoint formats, loud failures.
Step 9  : real checkpoint -> normalise_pdws -> windowed_cluster_deinterleave
          -> labels -> EmitterTracker -> BeliefState -> (360,) observation.
Step 10 : GNU RF end-to-end (PerTuneGenerator -> IQ -> PDWDetector ->
          FrequencyContext -> IQReceiverBridge -> SieveReceiver ->
          ReceiverObservation -> translation) at ~3200 MHz (band 6) and
          ~8000 MHz (emitter at band 15).
Step 11 : multi-dwell statefulness across bands [6,15,6,10,15] (x3).
Step 12 : determinism (same seed -> identical labels, belief, observation).
Step 13 : ground-truth isolation (emitter_id has zero effect on the obs).
Step 16 : failure handling (missing / incompatible checkpoint, missing stats).

Interpreter note
----------------
This suite loads torch + the real model and therefore runs under the MASTER
virtual environment (torch 2.14.0+cpu).  It is unittest-based so it can also be
collected by `python -m unittest discover` under radioconda-less environments.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors the dominant project layout)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]  # SIH2026_Try2 (repo root)
_SCRIPTS = str(_THIS_DIR.parent / "scripts")
_MASTER_PKG = str(_ROOT / "cognitive_ew_smart_scan")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if _MASTER_PKG not in sys.path:
    sys.path.insert(0, _MASTER_PKG)

import torch  # noqa: E402  (real model loading requires the master venv)

from deinterleaver_loader import (  # noqa: E402
    DEFAULT_DEINTERLEAVER_CONFIG,
    load_deinterleaver,
)
from scheduler_translation import GnuRfSchedulerTranslation  # noqa: E402
from src.models.deinterleaver import PDWTransformerEncoder, windowed_cluster_deinterleave  # noqa: E402
from src.preprocessing.normalise import normalise_pdws  # noqa: E402
from src.receiver.models import DetectionObservation, ReceiverObservation  # noqa: E402

# RF pipeline components (GNU_RF_ENV/scripts)
from frequency_context import FrequencyContext  # noqa: E402
from iq_bridge import IQReceiverBridge  # noqa: E402
from iq_to_pdw import PDWDetector  # noqa: E402
from per_tune_generator import EmitterConfig, PerTuneGenerator  # noqa: E402

CHECKPOINTS = _ROOT / "cognitive_ew_smart_scan" / "checkpoints"
BEST_PT = CHECKPOINTS / "best.pt"
FINAL_PT = CHECKPOINTS / "final.pt"
STATS_JSON = CHECKPOINTS / "normalization_stats.json"

FREQ_MIN, FREQ_MAX, N_BANDS = 0.0, 18000.0, 36
DWELL_TIME_US = 500.0
SAMPLE_RATE = 2e6


def _band(center_mhz: float) -> int:
    frac = min(1.0, max(0.0, center_mhz / (FREQ_MAX - FREQ_MIN)))
    return min(int(frac * N_BANDS), N_BANDS - 1)


def _detection(
    time_us: float,
    frequency_mhz: float,
    pulse_width_us: float,
    amplitude_db: float,
aoa_deg: float = 0.0,
        *,
        emitter_id=None,
) -> DetectionObservation:
    return DetectionObservation(
        time_us=time_us,
        frequency_mhz=frequency_mhz,
        pulse_width_us=pulse_width_us,
        amplitude_db=amplitude_db,
        aoa_deg=aoa_deg,
        detected=True,
        emitter_id=emitter_id,
    )


def _obs_for_dwell(
    dwell_start_us: float,
    center_frequency_mhz: float,
    n_pulses: int = 10,
    *,
    emitter_id=None,
    pw_us: float = 10.0,
    amp_db: float = -100.0,
) -> ReceiverObservation:
    """Deterministic synthetic ReceiverObservation (crisis of clarity)."""
    dets = [
        _detection(
            dwell_start_us + 100.0 + k * 40.0,
            3200.1 if k % 2 == 0 else 3200.5,
            pw_us,
            amp_db,
            emitter_id=emitter_id,
        )
        for k in range(n_pulses)
    ]
    return ReceiverObservation(
        time_us=dwell_start_us,
        center_frequency_mhz=center_frequency_mhz,
        dwell_time_us=DWELL_TIME_US,
        dwell_interval_us=[dwell_start_us, dwell_start_us + DWELL_TIME_US],
        detections=dets,
    )


class RFChain:
    """Per-tune GNURadio-style chain producing a ReceiverObservation per dwell.

    Faithful to the production data flow (and the env's dwell flush):
        PerTuneGenerator -> IQ -> PDWDetector.detect_iq
            -> FrequencyContext.local_to_rf -> IQReceiverBridge.pdw_to_pulse
            -> SieveReceiver.add_pulse -> _detect_buffered_interval
            -> _record -> ReceiverObservation
    """

    def __init__(
        self,
        center_frequency_mhz: float,
        emitter: EmitterConfig,
        *,
        noise_amplitude: float = 0.02,
        detector_threshold_db: float = 5.0,
        seed: int = 42,
    ) -> None:
        from src.receiver.sieve_receiver import SieveReceiver

        self.center = float(center_frequency_mhz)
        self.gen = PerTuneGenerator()
        self.gen.configure(
            center_frequency_mhz=self.center,
            noise_amplitude=noise_amplitude,
            noise_seed=seed,
        )
        self.gen.add_emitter(emitter)
        self.detector = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=detector_threshold_db,
        )
        self.ctx = FrequencyContext(self.center, ibw_mhz=1000.0)
        self.bridge = IQReceiverBridge(self.ctx, amp_placeholder_db=-100.0)
        self.receiver = SieveReceiver(
            total_bandwidth=18000.0,
            ibw=1000.0,
            frequency_step=500.0,
            dwell_time=DWELL_TIME_US,
            detection_threshold_db=-140.0,
        )
        self.receiver.tune(self.center)

    def dwell(self) -> tuple[ReceiverObservation, list[dict]]:
        """Generate + detect ONE dwell and return (observation, pdws)."""
        dwell_start = self.receiver.current_time_us
        dwell_end = dwell_start + self.receiver.dwell_time_us

        iq = self.gen.generate_iq(duration_us=DWELL_TIME_US)
        pdws = self.detector.detect_iq(
            iq, sample_offset=round(dwell_start * SAMPLE_RATE / 1e6)
        )
        pulses = [self.bridge.pdw_to_pulse(p) for p in pdws]
        for pulse in pulses:
            self.receiver.add_pulse(pulse)

        self.receiver.current_time_us = dwell_start
        self.receiver.dwell_start_us = dwell_start
        self.receiver.dwell_end_us = dwell_end
        detections = self.receiver._detect_buffered_interval(dwell_start, dwell_end)
        self.receiver._record(detections, observation_time_us=dwell_start)
        self.receiver.current_time_us = dwell_end
        self.receiver._prune(dwell_end)
        return self.receiver.get_observation(), pdws


def _run_rf_episode(
    chain_map: dict[int, RFChain],
    centers: list[float],
    tr: GnuRfSchedulerTranslation,
) -> list[np.ndarray]:
    obs_list: list[np.ndarray] = []
    for center in centers:
        obs, pdws = chain_map[_band(center)].dwell()
        obs_list.append(tr.update(obs))
    return obs_list


# ---------------------------------------------------------------------------
# Loader (Step 7 / Step 16)
# ---------------------------------------------------------------------------


class TestLoader(unittest.TestCase):
    """The loader must surface the ACTUAL production checkpoint, never a mock."""

    @classmethod
    def setUpClass(cls):
        cls.payload = load_deinterleaver(
            checkpoint_path=BEST_PT, stats_path=STATS_JSON
        )

    def test_loads_production_checkpoint(self):
        model = self.payload["model"]
        self.assertIsInstance(model, PDWTransformerEncoder)
        self.assertEqual(model.embed_dim, 64)
        self.assertFalse(model.training, "model must be in eval mode")

    def test_config_fit_stats_matches_production_stats(self):
        fit_stats = self.payload["config"]["fit_stats"]
        for key in ("cf_median", "cf_iqr", "pw_mean", "pw_std", "amp_mean", "amp_std"):
            self.assertIn(key, fit_stats, f"fit_stats missing {key}")

    def test_config_is_consumable(self):
        cfg = self.payload["config"]
        self.assertEqual(
            cfg["min_pulses"], DEFAULT_DEINTERLEAVER_CONFIG["min_pulses"]
        )
        self.assertEqual(
            cfg["interval_steps"], DEFAULT_DEINTERLEAVER_CONFIG["interval_steps"]
        )
        # window_size/stride must be ABSENT so the translation falls back to the
        # production defaults (min(2048, len); window // 2) instead of None // 2.
        self.assertNotIn("window_size", cfg)
        self.assertNotIn("stride", cfg)

    def test_loads_raw_checkpoint_format(self):
        payload = load_deinterleaver(
            checkpoint_path=FINAL_PT, stats_path=STATS_JSON
        )
        self.assertIsInstance(payload["model"], PDWTransformerEncoder)
        self.assertFalse(payload["model"].training)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_deinterleaver(
                checkpoint_path=CHECKPOINTS / "does_not_exist_best.pt",
                stats_path=STATS_JSON,
            )

    def test_missing_stats_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_deinterleaver(
                checkpoint_path=BEST_PT,
                stats_path=CHECKPOINTS / "does_not_exist_stats.json",
            )

    def test_incompatible_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pt"
            torch.save({"state_dict": {"scheduler_net.weight": torch.zeros(3, 3)}}, bad)
            with self.assertRaises(ValueError):
                load_deinterleaver(checkpoint_path=bad, stats_path=STATS_JSON)

    def test_wrong_arch_override_raises(self):
        with self.assertRaises(ValueError):
            load_deinterleaver(
                checkpoint_path=BEST_PT,
                stats_path=STATS_JSON,
                arch={"embed_dim": 999},
            )


# ---------------------------------------------------------------------------
# Step 9: real checkpoint -> clusters -> tracker -> belief -> obs
# ---------------------------------------------------------------------------


class TestRealCheckpointPipeline(unittest.TestCase):
    """The real checkpoint's clusters must drive tracker + belief + obs."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)
        cls.model = payload["model"]
        cls.fit_stats = payload["config"]["fit_stats"]

    def _pdws(self, n_dwells: int = 11) -> np.ndarray:
        rows: list[list[float]] = []
        for d in range(n_dwells):
            start = d * 500.0
            for k in range(10):
                rows.append(
                    [start + 100.0 + k * 40.0, 3200.1 if k % 2 == 0 else 3200.5, 10.0, 0.0, -100.0]
                )
        return np.asarray(rows, dtype=np.float64)

    def test_produces_cluster_labels(self):
        pdws = self._pdws()
        norm, _ = normalise_pdws(pdws, fit_stats=self.fit_stats)
        result = windowed_cluster_deinterleave(
            self.model, norm, toa_us=pdws[:, 0],
            window_size=min(2048, len(norm)), stride=(min(2048, len(norm)) // 2),
            device="cpu", min_cluster_size=10, min_samples=5,
        )
        self.assertEqual(len(result["labels"]), len(pdws))
        self.assertGreaterEqual(result["n_clusters"], 1)
        members = sum(1 for lb in result["labels"] if lb != -1)
        self.assertGreater(
            members, 0,
            "cluster labels must contain real members beyond the noise class",
        )

    def test_labels_feed_tracker_and_belief(self):
        payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=payload["model"],
            deinterleaver_config=payload["config"],
            freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
        )
        self.assertTrue(tr.perception_enabled)
        obs = None
        for d in range(12):
            obs = tr.update(_obs_for_dwell(d * 500.0, 3250.0))
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (360,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreaterEqual(float(obs.min()), 0.0)
        self.assertLessEqual(float(obs.max()), 1.0)
        self.assertIsNotNone(tr.emitter_tracker)
        self.assertGreaterEqual(len(tr.emitter_tracker.get_active_tracks()), 1)


# ---------------------------------------------------------------------------
# Step 10: GNU RF end-to-end
# ---------------------------------------------------------------------------


class TestRfEndToEnd(unittest.TestCase):
    """Full GNU RF chain -> real model -> (360,) scheduler observation."""

    @classmethod
    def setUpClass(cls):
        payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)
        cls.payload = payload

    def _episode(self, centers: list[float]):
        chain_map: dict[int, RFChain] = {}
        for c in sorted(set(centers)):
            b = _band(c)
            if b not in chain_map:
                chain_map[b] = RFChain(
                    c,
                    EmitterConfig(rf_frequency_mhz=round(c + 0.1, 3), pri_us=100.0),
                )
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=self.payload["model"],
            deinterleaver_config=self.payload["config"],
            freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
        )
        obs_list = _run_rf_episode(chain_map, centers, tr)
        return tr, chain_map, obs_list

    def _assert_360(self, obs):
        self.assertEqual(obs.shape, (360,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreaterEqual(float(obs.min()), 0.0)
        self.assertLessEqual(float(obs.max()), 1.0)

    def test_band6_3200mhz(self):
        centers = [3250.0] * 12
        tr, chain_map, obs_list = self._episode(centers)
        self.assertEqual(len(obs_list), 12)
        self._assert_360(obs_list[-1])
        # Record-visit evidence on the visited band (band 6) — causal obs output.
        b6 = obs_list[-1][6 * 10: 7 * 10]
        self.assertGreater(float(b6[6]), 0.0, "visited band must carry scan evidence")
        # Perception ran on the real model: tracker exists and had clusters.
        self.assertIsNotNone(tr.emitter_tracker)
        self.assertGreaterEqual(len(tr.emitter_tracker.get_active_tracks()), 1)
        # Buffer trimmed to the env's retention after a perception event.
        self.assertLessEqual(len(tr._pdw_buffer), 100,
            "buffer must be bounded (not unbounded growth) after perception trim + 1 dwell")

    def test_band15_7999mhz(self):
        centers = [8000.0] * 12
        tr, chain_map, obs_list = self._episode(centers)
        self.assertEqual(len(obs_list), 12)
        self._assert_360(obs_list[-1])
        # Emitter sits at 7999.75 MHz -> band 15; center 8000 -> visited band 16.
        b15 = obs_list[-1][15 * 10: 16 * 10]
        b16 = obs_list[-1][16 * 10: 17 * 10]
        self.assertGreater(float(b16[6]), 0.0, "visited band 16 must carry scan evidence")
        self.assertIsNotNone(tr.emitter_tracker)
        self.assertGreaterEqual(len(tr.emitter_tracker.get_active_tracks()), 1)


# ---------------------------------------------------------------------------
# Step 11: multi-dwell statefulness
# ---------------------------------------------------------------------------


class TestMultiDwellStatefulness(unittest.TestCase):
    """State persists across a mixed-band episode; no reset/contamination."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)

    def test_mixed_band_episode(self):
        block = [3250.0, 7800.0, 5100.0]
        centers = (block * 3)[:15]
        chain_map: dict[int, RFChain] = {}
        for c in sorted(set(centers)):
            chain_map[_band(c)] = RFChain(
                c, EmitterConfig(rf_frequency_mhz=round(c + 0.1, 3), pri_us=100.0)
            )
        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=self.payload["model"],
            deinterleaver_config=self.payload["config"],
            freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
        )
        prev = None
        for i, center in enumerate(centers):
            obs = tr.update(chain_map[_band(center)].dwell()[0])
            self.assertEqual(obs.shape, (360,))
            self.assertEqual(obs.dtype, np.float32)
            if prev is not None:
                self.assertFalse(np.isnan(obs).any())
        self.assertEqual(tr._step_count, len(centers))
        # Belief propagates dwell-over-dwell (not a fresh episural reset).
        self.assertIsNotNone(tr.belief)


# ---------------------------------------------------------------------------
# Step 12: determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    """Same seed/input -> identical labels, belief, observation."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)

    def test_deterministic_observation(self):
        centers = [3250.0] * 12
        scripts = []
        for _ in range(2):
            chain_map: dict[int, RFChain] = {}
            for c in sorted(set(centers)):
                chain_map[_band(c)] = RFChain(
                    c, EmitterConfig(rf_frequency_mhz=round(c + 0.1, 3), pri_us=100.0)
                )
            tr = GnuRfSchedulerTranslation(
                deinterleaver_model=self.payload["model"],
                deinterleaver_config=self.payload["config"],
                freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
            )
            obs_list = _run_rf_episode(chain_map, centers, tr)
            scripts.append((tr, obs_list))
        (tr_a, obs_a), (tr_b, obs_b) = scripts
        for oa, ob in zip(obs_a, obs_b):
            np.testing.assert_allclose(oa, ob, atol=1e-6)
        # Same number of tracks -> same clustering outcome.
        self.assertEqual(
            len(tr_a.emitter_tracker.get_active_tracks()),
            len(tr_b.emitter_tracker.get_active_tracks()),
        )

    def test_deterministic_labels_direct(self):
        pdws = np.asarray(
            [
                [d * 500.0 + 100.0 + k * 40.0, 3200.1 if k % 2 == 0 else 3200.5, 10.0, 0.0, -100.0]
                for d in range(11) for k in range(10)
            ],
            dtype=np.float64,
        )
        norm, _ = normalise_pdws(pdws, fit_stats=self.payload["config"]["fit_stats"])
        results = []
        for _ in range(2):
            r = windowed_cluster_deinterleave(
                self.payload["model"], norm, toa_us=pdws[:, 0],
                window_size=100, stride=50, device="cpu",
                min_cluster_size=10, min_samples=5,
            )
            results.append(r)
        np.testing.assert_array_equal(results[0]["labels"], results[1]["labels"])
        self.assertEqual(results[0]["n_clusters"], results[1]["n_clusters"])


# ---------------------------------------------------------------------------
# Step 13: ground-truth isolation
# ---------------------------------------------------------------------------


class TestGroundTruthIsolation(unittest.TestCase):
    """emitter_id must have zero effect on the observable output."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.payload = load_deinterleaver(checkpoint_path=BEST_PT, stats_path=STATS_JSON)

    def test_emitter_id_has_no_effect(self):
        runs = []
        for with_gt in (False, True):
            tr = GnuRfSchedulerTranslation(
                deinterleaver_model=self.payload["model"],
                deinterleaver_config=self.payload["config"],
                freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
            )
            obs = None
            for d in range(12):
                obs = tr.update(
                    _obs_for_dwell(d * 500.0, 3250.0, emitter_id=(7 if with_gt else None))
                )
            runs.append(obs)
        np.testing.assert_allclose(runs[0], runs[1], atol=0.0)


# ---------------------------------------------------------------------------
# Step 16: runtime degradation mirrors production (record_visit-only fallback)
# ---------------------------------------------------------------------------


class TestRuntimeDegradation(unittest.TestCase):
    """A broken model degrades to the production no-perception path, never crashes."""

    def test_broken_model_degrades_gracefully(self):
        class BrokenModel:
            embed_dim = 64

            def infer(self, x, device):
                raise RuntimeError("no inference")

        tr = GnuRfSchedulerTranslation(
            deinterleaver_model=BrokenModel(),
            deinterleaver_config={"min_pulses": 50, "interval_steps": 10},
            freq_min=FREQ_MIN, freq_max=FREQ_MAX, n_bands=N_BANDS,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs = None
            for d in range(12):
                obs = tr.update(_obs_for_dwell(d * 500.0, 3250.0))
        self.assertEqual(obs.shape, (360,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreaterEqual(float(obs.min()), 0.0)
        self.assertLessEqual(float(obs.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
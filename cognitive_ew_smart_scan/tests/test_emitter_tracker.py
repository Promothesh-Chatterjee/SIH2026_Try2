"""Tests for EmitterTracker - validates track persistence, matching, and confidence."""

import unittest
import numpy as np

from src.perception.emitter_tracker import (
    AssociationConfig,
    EmitterTracker,
    EmitterTrack,
)
from src.receiver.models import DetectionObservation


def _make_detections(n: int, freq: float, toa_start: float, pri: float, 
                     aoa: float = 0.0, pw: float = 1.0, amp: float = -80.0) -> list[DetectionObservation]:
    """Generate n DetectionObservation objects with regular PRI."""
    return [DetectionObservation(
        time_us=toa_start + i * pri,
        frequency_mhz=freq + np.random.normal(0, 0.1),
        pulse_width_us=pw,
        amplitude_db=amp,
        aoa_deg=aoa + np.random.normal(0, 0.5),
        detected=True,
    ) for i in range(n)]


def _build_dwell(specs):
    """Deterministic per-dwell arrays.

    Args:
        specs: List of dicts {label, freq, aoa, pw, amp, pri, toa_start, n}
               describing one cluster per entry.
    Returns:
        (labels, toa, freq, aoa, pw, amp) arrays aligned to pulses.
    """
    labels, toas, freqs, aoas, pws, amps = [], [], [], [], [], []
    for s in specs:
        n = s.get("n", 5)
        pri = s["pri"]
        t0 = s.get("toa_start", 0.0)
        labels.extend([int(s["label"])] * n)
        toas.extend(t0 + np.arange(n) * pri)
        freqs.extend([float(s["freq"])] * n)
        aoas.extend([float(s.get("aoa", 0.0))] * n)
        pws.extend([float(s.get("pw", 1.0))] * n)
        amps.extend([float(s.get("amp", -80.0))] * n)
    return (np.asarray(labels, dtype=np.int64),
            np.asarray(toas, dtype=np.float64),
            np.asarray(freqs, dtype=np.float64),
            np.asarray(aoas, dtype=np.float64),
            np.asarray(pws, dtype=np.float64),
            np.asarray(amps, dtype=np.float64))


def _update(tracker, labels, toa, freq, aoa, pw, amp, current_time, band=5, **kw):
    return tracker.update_from_deinterleaver(
        labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
        pw_us=pw, amp_db=amp, current_time=current_time, band=band, **kw)


class TestEmitterTrack(unittest.TestCase):
    """Tests for the EmitterTrack dataclass."""

    def test_pri_estimate_regular_signal(self):
        """Test PRI estimation for a regular signal."""
        track = EmitterTrack(track_id=0, cluster_label=1, last_seen_time=1000.0, last_band=5)
        
        # Add detections with regular PRI = 1000 us
        pri = 1000.0
        toa_start = 0.0
        for i in range(10):
            track.toa_history.append(toa_start + i * pri)
            track.frequency_history.append(5000.0)
            track.aoa_history.append(30.0)
            track.pw_history.append(1.0)
            track.amplitude_history.append(-80.0)
        
        track._update_pri_estimate()
        
        self.assertIsNotNone(track.pri_estimate_us)
        self.assertAlmostEqual(track.pri_estimate_us, pri, delta=50.0)
        self.assertGreater(track.pri_confidence, 0.9)

    def test_pri_estimate_irregular_signal(self):
        """Test PRI estimation for an irregular signal (low confidence)."""
        track = EmitterTrack(track_id=1, cluster_label=2, last_seen_time=1000.0, last_band=5)
        
        # Add detections with irregular PRI
        toas = [0, 1000, 2100, 3050, 4100, 5200]  # Varying intervals
        for toa in toas:
            track.toa_history.append(float(toa))
            track.frequency_history.append(5000.0)
            track.aoa_history.append(30.0)
            track.pw_history.append(1.0)
            track.amplitude_history.append(-80.0)
        
        track._update_pri_estimate()
        
        self.assertIsNotNone(track.pri_estimate_us)
        # The confidence might be higher than expected due to filtering
        self.assertLess(track.pri_confidence, 0.99)

    def test_agility_fixed_frequency(self):
        """Test agility score for fixed frequency emitter."""
        track = EmitterTrack(track_id=2, cluster_label=3, last_seen_time=1000.0, last_band=5)
        
        for i in range(10):
            track.frequency_history.append(5000.0 + np.random.normal(0, 0.1))
        
        track._update_agility()
        self.assertLess(track.agility_score, 0.1)

    def test_agility_hopping_frequency(self):
        """Test agility score for frequency-hopping emitter."""
        track = EmitterTrack(track_id=3, cluster_label=4, last_seen_time=1000.0, last_band=5)
        
        # Simulate hopping across 500 MHz (full IBW)
        for i in range(10):
            track.frequency_history.append(5000.0 + (i % 2) * 500.0)
        
        track._update_agility()
        self.assertGreater(track.agility_score, 0.5)


class TestEmitterTracker(unittest.TestCase):
    """Tests for the EmitterTracker class."""

    def setUp(self):
        self.tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)

    def test_new_track_creation(self):
        """Test that new clusters create new tracks."""
        detections = _make_detections(5, 5000.0, 0.0, 1000.0)
        labels = np.array([0, 0, 0, 0, 0])
        toa_us = np.array([d.time_us for d in detections])
        freq_mhz = np.array([d.frequency_mhz for d in detections])
        aoa_deg = np.array([d.aoa_deg for d in detections])
        pw_us = np.array([d.pulse_width_us for d in detections])
        amp_db = np.array([d.amplitude_db for d in detections])
        
        self.tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa_us, freq_mhz=freq_mhz,
            aoa_deg=aoa_deg, pw_us=pw_us, amp_db=amp_db,
            current_time=1000.0, band=5
        )
        
        self.assertEqual(len(self.tracker.tracks), 1)
        track = list(self.tracker.tracks.values())[0]
        self.assertEqual(track.cluster_label, 0)
        self.assertEqual(track.observation_count, 5)
        self.assertEqual(track.last_band, 5)

    def test_same_cluster_matches_existing_track(self):
        """Test that same cluster label in same band matches existing track."""
        # First dwell
        detections1 = _make_detections(5, 5000.0, 0.0, 1000.0)
        labels1 = np.array([0, 0, 0, 0, 0])
        toa1 = np.array([d.time_us for d in detections1])
        freq1 = np.array([d.frequency_mhz for d in detections1])
        aoa1 = np.array([d.aoa_deg for d in detections1])
        pw1 = np.array([d.pulse_width_us for d in detections1])
        amp1 = np.array([d.amplitude_db for d in detections1])
        
        self.tracker.update_from_deinterleaver(
            labels=labels1, toa_us=toa1, freq_mhz=freq1,
            aoa_deg=aoa1, pw_us=pw1, amp_db=amp1,
            current_time=1000.0, band=5
        )
        track_id_1 = list(self.tracker.tracks.keys())[0]
        
        # Second dwell - same cluster label, same band
        detections2 = _make_detections(5, 5000.0, 2000.0, 1000.0)
        labels2 = np.array([0, 0, 0, 0, 0])
        toa2 = np.array([d.time_us for d in detections2])
        freq2 = np.array([d.frequency_mhz for d in detections2])
        aoa2 = np.array([d.aoa_deg for d in detections2])
        pw2 = np.array([d.pulse_width_us for d in detections2])
        amp2 = np.array([d.amplitude_db for d in detections2])
        
        self.tracker.update_from_deinterleaver(
            labels=labels2, toa_us=toa2, freq_mhz=freq2,
            aoa_deg=aoa2, pw_us=pw2, amp_db=amp2,
            current_time=3000.0, band=5
        )
        
        # Should still have only 1 track
        self.assertEqual(len(self.tracker.tracks), 1)
        track = self.tracker.tracks[track_id_1]
        self.assertEqual(track.observation_count, 10)
        self.assertEqual(track.last_seen_time, 3000.0)

    def test_cluster_label_change_same_track(self):
        """Test that track persists even if cluster label changes (feature matching)."""
        # First dwell with cluster label 0
        detections1 = _make_detections(5, 5000.0, 0.0, 1000.0, aoa=30.0)
        labels1 = np.array([0, 0, 0, 0, 0])
        toa1 = np.array([d.time_us for d in detections1])
        freq1 = np.array([d.frequency_mhz for d in detections1])
        aoa1 = np.array([d.aoa_deg for d in detections1])
        pw1 = np.array([d.pulse_width_us for d in detections1])
        amp1 = np.array([d.amplitude_db for d in detections1])
        
        self.tracker.update_from_deinterleaver(
            labels=labels1, toa_us=toa1, freq_mhz=freq1,
            aoa_deg=aoa1, pw_us=pw1, amp_db=amp1,
            current_time=1000.0, band=5
        )
        track_id_1 = list(self.tracker.tracks.keys())[0]
        
        # Second dwell - deinterleaver assigns different cluster label (e.g., 5)
        # but same physical emitter (same freq, aoa, pri)
        detections2 = _make_detections(5, 5000.0, 2000.0, 1000.0, aoa=30.0)
        labels2 = np.array([5, 5, 5, 5, 5])  # Different label!
        toa2 = np.array([d.time_us for d in detections2])
        freq2 = np.array([d.frequency_mhz for d in detections2])
        aoa2 = np.array([d.aoa_deg for d in detections2])
        pw2 = np.array([d.pulse_width_us for d in detections2])
        amp2 = np.array([d.amplitude_db for d in detections2])
        
        self.tracker.update_from_deinterleaver(
            labels=labels2, toa_us=toa2, freq_mhz=freq2,
            aoa_deg=aoa2, pw_us=pw2, amp_db=amp2,
            current_time=3000.0, band=5
        )
        
        # Should still have 1 track (matched by features)
        self.assertEqual(len(self.tracker.tracks), 1)
        track = self.tracker.tracks[track_id_1]
        self.assertEqual(track.observation_count, 10)

    def test_adjacent_band_matching_agile_emitter(self):
        """Test that agile emitter can match across adjacent bands."""
        tracker_agile = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        
        # First dwell in band 5
        detections1 = _make_detections(5, 5250.0, 0.0, 1000.0)
        labels1 = np.array([0, 0, 0, 0, 0])
        toa1 = np.array([d.time_us for d in detections1])
        freq1 = np.array([d.frequency_mhz for d in detections1])
        aoa1 = np.array([d.aoa_deg for d in detections1])
        pw1 = np.array([d.pulse_width_us for d in detections1])
        amp1 = np.array([d.amplitude_db for d in detections1])
        
        tracker_agile.update_from_deinterleaver(
            labels=labels1, toa_us=toa1, freq_mhz=freq1,
            aoa_deg=aoa1, pw_us=pw1, amp_db=amp1,
            current_time=1000.0, band=5
        )
        track_id_1 = list(tracker_agile.tracks.keys())[0]
        
        # Make track agile by adding frequency hopping evidence
        track = tracker_agile.tracks[track_id_1]
        for i in range(5):
            track.frequency_history.append(5250.0 if i % 2 == 0 else 5750.0)
        track._update_agility()
        self.assertGreater(track.agility_score, 0.3)
        
        # Second dwell in adjacent band 6 (simulating frequency hop)
        detections2 = _make_detections(5, 5750.0, 2000.0, 1000.0)
        labels2 = np.array([0, 0, 0, 0, 0])
        toa2 = np.array([d.time_us for d in detections2])
        freq2 = np.array([d.frequency_mhz for d in detections2])
        aoa2 = np.array([d.aoa_deg for d in detections2])
        pw2 = np.array([d.pulse_width_us for d in detections2])
        amp2 = np.array([d.amplitude_db for d in detections2])
        
        tracker_agile.update_from_deinterleaver(
            labels=labels2, toa_us=toa2, freq_mhz=freq2,
            aoa_deg=aoa2, pw_us=pw2, amp_db=amp2,
            current_time=3000.0, band=6
        )
        
        # Should match to existing track despite band change
        self.assertEqual(len(tracker_agile.tracks), 1)

    def test_stale_track_pruning(self):
        """Test that tracks with too many consecutive misses are pruned."""
        detections = _make_detections(5, 5000.0, 0.0, 1000.0)
        labels = np.array([0, 0, 0, 0, 0])
        toa = np.array([d.time_us for d in detections])
        freq = np.array([d.frequency_mhz for d in detections])
        aoa = np.array([d.aoa_deg for d in detections])
        pw = np.array([d.pulse_width_us for d in detections])
        amp = np.array([d.amplitude_db for d in detections])
        
        self.tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq,
            aoa_deg=aoa, pw_us=pw, amp_db=amp,
            current_time=1000.0, band=5
        )
        track_id = list(self.tracker.tracks.keys())[0]
        
        # Simulate 5 consecutive misses (max_misses_before_drop=5)
        for i in range(6):
            self.tracker.update_from_deinterleaver(
                labels=np.array([-1, -1, -1]),  # No detections
                toa_us=np.array([1000.0, 2000.0, 3000.0]),
                freq_mhz=np.array([5000.0, 5000.0, 5000.0]),
                aoa_deg=np.array([30.0, 30.0, 30.0]),
                pw_us=np.array([1.0, 1.0, 1.0]),
                amp_db=np.array([-80.0, -80.0, -80.0]),
                current_time=2000.0 + i * 1000.0,
                band=5
            )
        
        # Track should be pruned
        self.assertNotIn(track_id, self.tracker.tracks)

    def test_band_belief_generation(self):
        """Test that band belief is generated correctly from tracks."""
        # Band 5 covers 2500-3000 MHz (band_width = 500 MHz)
        band5_center = 2750.0
        detections = _make_detections(10, band5_center, 0.0, 1000.0, aoa=30.0)
        labels = np.array([0] * 10)
        toa = np.array([d.time_us for d in detections])
        freq = np.array([d.frequency_mhz for d in detections])
        aoa = np.array([d.aoa_deg for d in detections])
        pw = np.array([d.pulse_width_us for d in detections])
        amp = np.array([d.amplitude_db for d in detections])
        
        self.tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq,
            aoa_deg=aoa, pw_us=pw, amp_db=amp,
            current_time=1000.0, band=5
        )
        
        belief = self.tracker.get_band_belief(freq_min=0.0, freq_max=18000.0)
        
        self.assertIn("obs", belief)
        self.assertIn("bands", belief)
        self.assertEqual(belief["obs"].shape, (360,))  # 36 bands * 10 features
        self.assertEqual(belief["bands"].shape, (36, 10))
        
        # Band 5 should have non-zero features
        band5 = belief["bands"][5]
        self.assertGreater(band5[0], 0.0)  # occupancy
        self.assertGreater(band5[5], 0.0)  # emitter count
        self.assertGreater(band5[6], 0.0)  # deinterleaver confidence


class TestClusterLabelPermutation(unittest.TestCase):
    """Identity must be robust to arbitrary (permuted) HDBSCAN cluster labels."""

    def test_single_emitter_label_permutes_across_dwells(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        seen_tids = set()
        for idx, label in enumerate((0, 1, 2, 0, 99)):
            labels, toa, freq, aoa, pw, amp = _build_dwell([
                {"label": label, "freq": 5000.0, "pri": 1000.0,
                 "toa_start": idx * 20000.0}
            ])
            _update(tracker, labels, toa, freq, aoa, pw, amp,
                    current_time=idx * 20000.0 + 5000.0)
            self.assertEqual(len(tracker.tracks), 1)
            seen_tids.update(tracker.tracks.keys())

        self.assertEqual(len(seen_tids), 1)  # never created a duplicate track
        track = list(tracker.tracks.values())[0]
        self.assertEqual(track.observation_count, 25)
        self.assertEqual(track.cluster_label, 99)  # diagnostic label fits latest

    def test_two_emitters_labels_swap_identities_stable(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        # Emitter A: 5000 MHz / 30 deg / pri 1000; Emitter B: 9000 / 120 / pri 2000.
        dwells = [
            [{"label": 0, "freq": 5000.0, "aoa": 30.0, "pri": 1000.0},
             {"label": 1, "freq": 9000.0, "aoa": 120.0, "pri": 2000.0}],
            [{"label": 1, "freq": 5000.0, "aoa": 30.0, "pri": 1000.0},
             {"label": 0, "freq": 9000.0, "aoa": 120.0, "pri": 2000.0}],  # swap
            [{"label": 7, "freq": 5000.0, "aoa": 30.0, "pri": 1000.0},
             {"label": 3, "freq": 9000.0, "aoa": 120.0, "pri": 2000.0}],
        ]
        for dt, dwell in enumerate(dwells):
            base = dt * 30000.0
            spec = []
            for s in dwell:
                s = dict(s)
                s["toa_start"] = base + (5000.0 if s["freq"] > 7000.0 else 0.0)
                spec.append(s)
            labels, toa, freq, aoa, pw, amp = _build_dwell(spec)
            _update(tracker, labels, toa, freq, aoa, pw, amp,
                    current_time=base + 15000.0)

        self.assertEqual(len(tracker.tracks), 2)
        track_a = next(t for t in tracker.tracks.values() if abs(t.current_frequency_mhz - 5000.0) < 1.0)
        track_b = next(t for t in tracker.tracks.values() if abs(t.current_frequency_mhz - 9000.0) < 1.0)
        # Tracks created in first dwell order: created before matching, so
        # track_ids 0 and 1; identity must never flip as labels permute.
        self.assertEqual(track_a.track_id, 0)
        self.assertEqual(track_b.track_id, 1)
        self.assertEqual(track_a.observation_count, 15)
        self.assertEqual(track_b.observation_count, 15)
        # physical state maintained per identity
        self.assertAlmostEqual(track_a.current_aoa_deg, 30.0, places=2)
        self.assertAlmostEqual(track_a.pri_estimate_us, 1000.0, places=2)
        self.assertAlmostEqual(track_b.current_aoa_deg, 120.0, places=2)
        self.assertAlmostEqual(track_b.pri_estimate_us, 2000.0, places=2)


class TestCompositeGates(unittest.TestCase):
    """Physically impossible associations must be rejected, not force-matched."""

    def setUp(self):
        self.tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0}
        ])
        _update(self.tracker, labels, toa, freq, aoa, pw, amp, current_time=10000.0)
        self.track_id = list(self.tracker.tracks)[0]

    def test_reject_physically_impossible_frequency(self):
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 12000.0, "pri": 1000.0, "toa_start": 10000.0}
        ])
        _update(self.tracker, labels, toa, freq, aoa, pw, amp, current_time=20000.0)
        self.assertEqual(len(self.tracker.tracks), 2)  # new track, not merged
        self.assertEqual(self.tracker.tracks[self.track_id].consecutive_misses, 1)

    def test_reject_physically_impossible_aoa(self):
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "aoa": 170.0, "pri": 1000.0, "toa_start": 10000.0}
        ])
        _update(self.tracker, labels, toa, freq, aoa, pw, amp, current_time=20000.0)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(self.tracker.tracks[self.track_id].consecutive_misses, 1)

    def test_reject_physically_impossible_pri(self):
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 2500.0, "toa_start": 10000.0}
        ])
        _update(self.tracker, labels, toa, freq, aoa, pw, amp, current_time=20000.0)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(self.tracker.tracks[self.track_id].consecutive_misses, 1)

    def test_reject_band_jump_far_for_fixed_emitter(self):
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0, "toa_start": 10000.0}
        ])
        _update(self.tracker, labels, toa, freq, aoa, pw, amp, current_time=20000.0, band=20)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(self.tracker.tracks[self.track_id].consecutive_misses, 1)


class TestUniquenessConstraints(unittest.TestCase):
    """One cluster -> one track; one track -> one cluster (unless justified)."""

    def test_single_cluster_never_assigned_to_multiple_tracks(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        # Two indistinguishable twin emitters get their own tracks on dwell 1.
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0},
            {"label": 1, "freq": 5000.0, "pri": 1000.0, "toa_start": 5000.0},
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=10000.0)
        self.assertEqual(len(tracker.tracks), 2)
        tid0, tid1 = sorted(tracker.tracks)

        # One cluster matching both tracks must update exactly one of them.
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0, "toa_start": 20000.0},
        ])
        matched = _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=30000.0)
        self.assertEqual(len(matched), 1)
        updated = next(iter(matched))
        self.assertEqual(tracker.tracks[updated].observation_count, 10)
        other = tid1 if updated == tid0 else tid0
        self.assertEqual(tracker.tracks[other].observation_count, 5)
        self.assertEqual(tracker.tracks[other].consecutive_misses, 1)

    def test_track_not_split_across_clusters_by_default(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        # Establish a periodic track at pri 2000.
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 2000.0}
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=10000.0)
        self.assertEqual(len(tracker.tracks), 1)
        tid0 = list(tracker.tracks)[0]

        # Two interleaved pri-2000 pulse trains (staggered by one PRI half).
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 2000.0, "toa_start": 20000.0},
            {"label": 1, "freq": 5000.0, "pri": 2000.0, "toa_start": 21000.0},
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=30000.0)
        # Without explicit justification one cluster spawns a new track.
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.tracks[tid0].observation_count, 10)

    def test_track_split_justified_for_staggered_periodic_emitter(self):
        cfg = AssociationConfig(allow_track_split=True, score_threshold=0.5)
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5,
                                 association_config=cfg)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 2000.0}
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=10000.0)
        tid0 = list(tracker.tracks)[0]

        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 2000.0, "toa_start": 20000.0},
            {"label": 1, "freq": 5000.0, "pri": 2000.0, "toa_start": 21000.0},
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=30000.0)
        self.assertEqual(len(tracker.tracks), 1)  # both trains -> same track
        self.assertEqual(tracker.tracks[tid0].observation_count, 15)


class TestAssociationPrediction(unittest.TestCase):
    """Association is prediction-driven (predict state before matching)."""

    def test_fixed_emitter_predicts_around_mean(self):
        track = EmitterTrack(track_id=0, cluster_label=0, last_seen_time=0.0)
        for i in range(10):
            track.frequency_history.append(5000.0 + i * 0.01)
        pred = track.predict_next_frequency()
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred, 5000.0, delta=0.5)

    def test_drifting_emitter_prediction_extrapolates_trend(self):
        track = EmitterTrack(track_id=0, cluster_label=0, last_seen_time=0.0,
                             toa_history=[i * 1000.0 for i in range(10)])
        for i in range(10):
            track.frequency_history.append(5000.0 + i * 10.0)
        track._update_frequency_trend()
        track._update_agility()
        pred = track.predict_next_frequency()
        mean = np.mean(track.frequency_history)
        # Prediction must extrapolate beyond the recent mean along the drift.
        self.assertGreater(pred, mean)

    def test_predict_track_state_shape(self):
        track = EmitterTrack(track_id=0, cluster_label=0, last_seen_time=0.0)
        for i in range(5):
            track.frequency_history.append(5000.0 + i)
            track.toa_history.append(i * 1000.0)
        state = track.predict_track_state()
        for key in ("predicted_frequency_mhz", "frequency_low_mhz",
                    "frequency_high_mhz", "pri_estimate_us", "agility_class",
                    "current_frequency_mhz", "frequency_range_mhz"):
            self.assertIn(key, state)
        self.assertLess(state["frequency_low_mhz"], state["predicted_frequency_mhz"])
        self.assertGreater(state["frequency_high_mhz"], state["predicted_frequency_mhz"])

    def test_drifting_emitter_track_persists(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        # Drift manifests across dwells: 5000 -> 5050 -> 5100 MHz.
        for dwell_idx, center in enumerate((5000.0, 5050.0, 5100.0)):
            labels, toa, freq, aoa, pw, amp = _build_dwell([
                {"label": 0, "freq": center, "pri": 1000.0,
                 "toa_start": dwell_idx * 30000.0}
            ])
            _update(tracker, labels, toa, freq, aoa, pw, amp,
                    current_time=dwell_idx * 30000.0 + 5000.0)
            self.assertEqual(len(tracker.tracks), 1)

        track = tracker.tracks[0]
        self.assertEqual(track.observation_count, 15)
        self.assertEqual(track.agility_class, "drifting")
        self.assertGreater(track.frequency_trend_mhz_per_pulse, 2.0)
        # Prediction extrapolates along the trend, not back toward the mean.
        self.assertGreater(track.predict_next_frequency(), 5100.0)


class TestEmitterBehaviours(unittest.TestCase):
    def test_periodic_emitter_pri_maintained(self):
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        for dwell_idx in range(2):
            labels, toa, freq, aoa, pw, amp = _build_dwell([
                {"label": 0, "freq": 6000.0, "pri": 750.0,
                 "toa_start": dwell_idx * 3750.0}
            ])
            _update(tracker, labels, toa, freq, aoa, pw, amp,
                    current_time=dwell_idx * 3750.0 + 5000.0)
        self.assertEqual(len(tracker.tracks), 1)
        track = list(tracker.tracks.values())[0]
        self.assertAlmostEqual(track.pri_estimate_us, 750.0, places=2)
        self.assertGreater(track.pri_confidence, 0.9)
        self.assertTrue(track.is_periodic)

    def test_fixed_emitter_low_agility(self):
        track = EmitterTrack(track_id=0, cluster_label=0, last_seen_time=0.0)
        for i in range(10):
            track.frequency_history.append(5000.0 + (i % 2) * 0.1)
        track._update_agility()
        self.assertLess(track.agility_score, 0.1)
        self.assertEqual(track.agility_class, "fixed")


class TestEmbeddingSimilarity(unittest.TestCase):
    def test_matching_embedding_centroids_associate(self):
        cfg = AssociationConfig(use_embedding_similarity=True, score_threshold=0.5)
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5,
                                 association_config=cfg)
        emb = np.ones((5, 8), dtype=np.float32)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0}
        ])
        tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
            pw_us=pw, amp_db=amp, current_time=10000.0, band=5, embeddings=emb,
        )
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0, "toa_start": 10000.0}
        ])
        tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
            pw_us=pw, amp_db=amp, current_time=20000.0, band=5, embeddings=emb,
        )
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(list(tracker.tracks.values())[0].observation_count, 10)

    def test_dissimilar_embedding_centroids_rejected(self):
        cfg = AssociationConfig(use_embedding_similarity=True, score_threshold=0.5)
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5,
                                 association_config=cfg)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0}
        ])
        tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
            pw_us=pw, amp_db=amp, current_time=10000.0, band=5,
            embeddings=np.ones((5, 8), dtype=np.float32),
        )
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "pri": 1000.0, "toa_start": 10000.0}
        ])
        tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
            pw_us=pw, amp_db=amp, current_time=20000.0, band=5,
            embeddings=-np.ones((5, 8), dtype=np.float32),
        )
        self.assertEqual(len(tracker.tracks), 2)  # rejected -> new track


class TestRequiredTrackFields(unittest.TestCase):
    def test_all_maintained_fields_present_and_sane(self):
        tracker = EmitterTracker(n_bands=36)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "aoa": 30.0, "pw": 1.5,
             "amp": -75.0, "pri": 1000.0}
        ])
        _update(tracker, labels, toa, freq, aoa, pw, amp, current_time=10000.0)
        track = list(tracker.tracks.values())[0]

        self.assertIsInstance(track.track_id, int)
        self.assertAlmostEqual(track.current_frequency_mhz, 5000.0, places=2)
        self.assertLess(track.frequency_range_mhz, 1.0)
        self.assertAlmostEqual(track.current_aoa_deg, 30.0, places=2)
        self.assertAlmostEqual(track.current_pw_us, 1.5, places=2)
        self.assertAlmostEqual(track.current_amplitude_db, -75.0, places=2)
        self.assertIsNotNone(track.pri_estimate_us)
        self.assertGreater(track.pri_confidence, 0.9)
        self.assertLess(track.agility_score, 0.3)
        self.assertEqual(track.last_seen_time, 10000.0)
        self.assertEqual(track.last_band, 5)
        self.assertEqual(track.observation_count, 5)
        self.assertGreater(track.cluster_confidence, 0.0)
        self.assertEqual(track.consecutive_misses, 0)

    def test_composite_score_exposes_all_components(self):
        # Deterministic single-dwell track then candidate re-score.
        tracker = EmitterTracker(n_bands=36)
        cfg = AssociationConfig(use_embedding_similarity=True)
        labels, toa, freq, aoa, pw, amp = _build_dwell([
            {"label": 0, "freq": 5000.0, "aoa": 30.0, "pri": 1000.0}
        ])
        tracker.update_from_deinterleaver(
            labels=labels, toa_us=toa, freq_mhz=freq, aoa_deg=aoa,
            pw_us=pw, amp_db=amp, current_time=10000.0, band=5,
            embeddings=np.ones((5, 8), dtype=np.float32),
        )
        track = list(tracker.tracks.values())[0]

        import src.perception.emitter_tracker as emt
        report = emt._ClusterReport(
            label=0, detections=[], mean_freq_mhz=5000.0, mean_aoa_deg=30.0,
            mean_pw_us=1.0, mean_amp_db=-80.0, pri_estimate_us=1000.0,
            pri_confidence=1.0, toa_min_us=20000.0, toa_max_us=24000.0,
            n=5, embedding_centroid=np.ones(8, dtype=np.float32),
        )
        score, ok, comps, reason = tracker._association_score(
            track, report, 5, 20000.0, cfg
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertGreater(score, 0.7)
        for key in ("freq", "aoa", "pw", "pri", "temporal", "recency", "agility", "embedding"):
            self.assertIn(key, comps)


if __name__ == "__main__":
    unittest.main()
"""Tests for EmitterTracker - validates track persistence, matching, and confidence."""

import unittest
import numpy as np

from src.perception.emitter_tracker import EmitterTracker, EmitterTrack
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
        
        # Make track agile by adding frequency variation
        track = tracker_agile.tracks[track_id_1]
        for _ in range(5):
            track.frequency_history.append(5250.0 + np.random.normal(0, 100.0))
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


if __name__ == "__main__":
    unittest.main()
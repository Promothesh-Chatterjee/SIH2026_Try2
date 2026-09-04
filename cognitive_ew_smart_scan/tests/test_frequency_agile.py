"""Tests for frequency-agile emitter scenarios - validates agile emitter detection and tracking."""

import unittest
import numpy as np

from src.perception.emitter_tracker import EmitterTracker, EmitterTrack
from src.receiver.models import DetectionObservation
from src.environment.scenario_generator import synthetic_records


class TestFrequencyAgileEmitters(unittest.TestCase):
    """Tests for frequency-agile emitter scenarios."""

    def test_fixed_vs_agile_emitter_scenario(self):
        """Test that agile emitters are correctly distinguished from fixed ones."""
        # Fixed frequency emitter
        fixed_track = EmitterTrack(
            track_id=0, cluster_label=0, last_seen_time=0.0, last_band=5
        )
        for i in range(10):
            fixed_track.toa_history.append(i * 1000.0)
            fixed_track.frequency_history.append(5000.0 + np.random.normal(0, 0.1))
            fixed_track.aoa_history.append(30.0)
            fixed_track.pw_history.append(1.0)
            fixed_track.amplitude_history.append(-80.0)
        fixed_track._update_agility()
        
        # Agile frequency-hopping emitter
        agile_track = EmitterTrack(
            track_id=1, cluster_label=1, last_seen_time=0.0, last_band=5
        )
        for i in range(10):
            agile_track.toa_history.append(i * 1000.0)
            agile_track.frequency_history.append(5000.0 + (i % 3) * 500.0)  # Hops across 3 bands
            agile_track.aoa_history.append(30.0)
            agile_track.pw_history.append(1.0)
            agile_track.amplitude_history.append(-80.0)
        agile_track._update_agility()
        
        # Fixed emitter should have low agility
        self.assertLess(fixed_track.agility_score, 0.2)
        
        # Agile emitter should have high agility
        self.assertGreater(agile_track.agility_score, 0.5)

    def test_agile_emitter_track_persistence(self):
        """Test that agile emitter tracks persist across band hops."""
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        
        # Simulate frequency-hopping emitter that hops between bands 5, 6, 7
        # First dwell in band 5
        detections1 = [
            DetectionObservation(
                time_us=1000.0, frequency_mhz=5250.0, pulse_width_us=1.0,
                amplitude_db=-80.0, aoa_deg=30.0, detected=True
            ) for _ in range(5)
        ]
        labels1 = np.array([0] * 5)
        toa1 = np.array([1000.0 + i * 1000.0 for i in range(5)])
        freq1 = np.array([5250.0] * 5)
        aoa1 = np.array([30.0] * 5)
        pw1 = np.array([1.0] * 5)
        amp1 = np.array([-80.0] * 5)
        
        tracker.update_from_deinterleaver(
            labels=labels1, toa_us=toa1, freq_mhz=freq1,
            aoa_deg=aoa1, pw_us=pw1, amp_db=amp1,
            current_time=1000.0, band=5
        )
        track_id = list(tracker.tracks.keys())[0]
        
        # Second dwell in adjacent band 6 (frequency hop)
        detections2 = [
            DetectionObservation(
                time_us=3000.0, frequency_mhz=5750.0, pulse_width_us=1.0,
                amplitude_db=-80.0, aoa_deg=30.0, detected=True
            ) for _ in range(5)
        ]
        labels2 = np.array([0] * 5)
        toa2 = np.array([3000.0 + i * 1000.0 for i in range(5)])
        freq2 = np.array([5750.0] * 5)
        aoa2 = np.array([30.0] * 5)
        pw2 = np.array([1.0] * 5)
        amp2 = np.array([-80.0] * 5)
        
        tracker.update_from_deinterleaver(
            labels=labels2, toa_us=toa2, freq_mhz=freq2,
            aoa_deg=aoa2, pw_us=pw2, amp_db=amp2,
            current_time=3000.0, band=6
        )
        
        # Track should persist and update
        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[track_id]
        self.assertEqual(track.observation_count, 10)
        self.assertEqual(track.last_band, 6)
        
        # Track should be marked as agile
        self.assertGreater(track.agility_score, 0.3)

    def test_multiple_agile_emitters(self):
        """Test tracking multiple agile emitters simultaneously."""
        tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
        
        # Emitter 1: hops between bands 5 and 6
        for hop in range(3):
            band = 5 + (hop % 2)
            detections = [
                DetectionObservation(
                    time_us=1000.0 + hop * 5000.0 + i * 1000.0,
                    frequency_mhz=5250.0 if band == 5 else 5750.0,
                    pulse_width_us=1.0, amplitude_db=-80.0,
                    aoa_deg=30.0, detected=True
                ) for i in range(5)
            ]
            labels = np.array([0] * 5)
            toa_us = np.array([d.time_us for d in detections])
            freq_mhz = np.array([d.frequency_mhz for d in detections])
            aoa_deg = np.array([d.aoa_deg for d in detections])
            pw_us = np.array([d.pulse_width_us for d in detections])
            amp_db = np.array([d.amplitude_db for d in detections])
            
            tracker.update_from_deinterleaver(
                labels=labels, toa_us=toa_us, freq_mhz=freq_mhz,
                aoa_deg=aoa_deg, pw_us=pw_us, amp_db=amp_db,
                current_time=1000.0 + hop * 5000.0, band=band
            )
        
        # Emitter 2: hops between bands 8 and 9
        for hop in range(3):
            band = 8 + (hop % 2)
            detections = [
                DetectionObservation(
                    time_us=2000.0 + hop * 5000.0 + i * 1000.0,
                    frequency_mhz=8250.0 if band == 8 else 8750.0,
                    pulse_width_us=1.0, amplitude_db=-85.0,
                    aoa_deg=45.0, detected=True
                ) for i in range(5)
            ]
            labels = np.array([1] * 5)
            toa_us = np.array([d.time_us for d in detections])
            freq_mhz = np.array([d.frequency_mhz for d in detections])
            aoa_deg = np.array([d.aoa_deg for d in detections])
            pw_us = np.array([d.pulse_width_us for d in detections])
            amp_db = np.array([d.amplitude_db for d in detections])
            
            tracker.update_from_deinterleaver(
                labels=labels, toa_us=toa_us, freq_mhz=freq_mhz,
                aoa_deg=aoa_deg, pw_us=pw_us, amp_db=amp_db,
                current_time=2000.0 + hop * 5000.0, band=band
            )
        
        # Should have 2 tracks
        self.assertEqual(len(tracker.tracks), 2)
        
        # Both should have high agility
        for track in tracker.tracks.values():
            self.assertGreater(track.agility_score, 0.3)

    def test_agile_emitter_confidence(self):
        """Test that agile emitter confidence reflects tracking quality."""
        tracker = EmitterTracker(n_bands=36)
        
        # Create agile emitter with good tracking
        for hop in range(5):
            band = 5 + (hop % 2)
            detections = [
                DetectionObservation(
                    time_us=1000.0 + hop * 5000.0 + i * 1000.0,
                    frequency_mhz=5250.0 if band == 5 else 5750.0,
                    pulse_width_us=1.0, amplitude_db=-80.0,
                    aoa_deg=30.0, detected=True
                ) for i in range(5)
            ]
            labels = np.array([0] * 5)
            toa_us = np.array([d.time_us for d in detections])
            freq_mhz = np.array([d.frequency_mhz for d in detections])
            aoa_deg = np.array([d.aoa_deg for d in detections])
            pw_us = np.array([d.pulse_width_us for d in detections])
            amp_db = np.array([d.amplitude_db for d in detections])
            
            tracker.update_from_deinterleaver(
                labels=labels, toa_us=toa_us, freq_mhz=freq_mhz,
                aoa_deg=aoa_deg, pw_us=pw_us, amp_db=amp_db,
                current_time=1000.0 + hop * 5000.0, band=band
            )
        
        track = list(tracker.tracks.values())[0]
        confidence = track.get_cluster_confidence()
        
        # Should have reasonable confidence despite agility
        self.assertGreater(confidence, 0.3)
        self.assertLess(confidence, 1.0)
        
        # Test confidence decreases with missed dwells
        for _ in range(3):
            tracker.update_from_deinterleaver(
                labels=np.array([-1, -1, -1]),
                toa_us=np.array([0.0, 1000.0, 2000.0]),
                freq_mhz=np.array([5000.0, 5000.0, 5000.0]),
                aoa_deg=np.array([30.0, 30.0, 30.0]),
                pw_us=np.array([1.0, 1.0, 1.0]),
                amp_db=np.array([-80.0, -80.0, -80.0]),
                current_time=30000.0, band=5
            )
        
        confidence_after_misses = track.get_cluster_confidence()
        self.assertLess(confidence_after_misses, confidence)

    def test_agile_vs_fixed_scheduler_implication(self):
        """Test that agile emitters produce different belief features than fixed ones."""
        from src.perception.adapters import build_band_belief_from_tracks
        
        # Fixed emitter - use more pulses to build up occupancy
        fixed_labels = np.array([0] * 50)
        fixed_toa = np.array([i * 1000.0 for i in range(50)])
        fixed_freq = np.array([5000.0 + np.random.normal(0, 0.1) for _ in range(50)])
        
        # Agile emitter - frequency hops WITHIN bands (not just across bands)
        agile_labels = np.array([1] * 50)
        agile_toa = np.array([i * 1000.0 for i in range(50)])
        # Frequency varies within band 10 (5000-5500 MHz range)
        agile_freq = np.array([5000.0 + np.random.uniform(0, 500) for _ in range(50)])
        
        # Build beliefs
        fixed_belief = build_band_belief_from_tracks(fixed_labels, fixed_toa, fixed_freq, 36, 0.0, 18000.0)
        agile_belief = build_band_belief_from_tracks(agile_labels, agile_toa, agile_freq, 36, 0.0, 18000.0)
        
        # Fixed emitter: single band should have high occupancy, low agility
        fixed_bands = fixed_belief["bands"]
        # Band 10 (5000 MHz) should be active (occupancy = 0.3 from EMA with alpha=0.3)
        self.assertGreaterEqual(fixed_bands[10, 0], 0.3)  # occupancy (EMA with alpha=0.3)
        self.assertLess(fixed_bands[10, 8], 0.2)  # agility
        
        # Agile emitter: multiple bands active, higher agility
        agile_bands = agile_belief["bands"]
        active_bands = np.sum(agile_bands[:, 0] >= 0.3)
        self.assertGreaterEqual(active_bands, 1)  # At least 1 band active
        
        # Agile emitter should have higher agility in at least one band
        max_agile_agility = np.max(agile_bands[:, 8])
        self.assertGreater(max_agile_agility, 0.0)


if __name__ == "__main__":
    unittest.main()
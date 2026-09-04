"""Tests for SemanticMemory - validates empty initialization, learning, and priority updates."""

import unittest
import tempfile
import os
from pathlib import Path

from src.cognitive.memory import SemanticMemory, EmitterProfile


class TestSemanticMemory(unittest.TestCase):
    """Tests for the SemanticMemory class."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.memory = SemanticMemory(self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        self.memory.close()
        if self.db_path.exists():
            self.db_path.unlink()
        os.rmdir(self.temp_dir)

    def test_starts_empty(self):
        """Test that semantic memory starts empty."""
        emitters = self.memory.list_emitters()
        self.assertEqual(len(emitters), 0)

    def test_write_and_read_emitter(self):
        """Test writing and reading an emitter profile."""
        profile = EmitterProfile(
            emitter_id="test_emitter_1",
            mean_pri_us=1000.0,
            freq_min_mhz=5000.0,
            freq_max_mhz=5500.0,
            mean_pw_us=1.0,
            aoa_mean=30.0,
            amplitude_mean=-80.0,
            priority_score=0.7,
            is_periodic=1,
            scan_period_us=10000.0,
            intercept_count=5,
            last_seen_us=1000000.0,
        )
        
        self.memory.write_emitter(profile)
        
        retrieved = self.memory.get_emitter("test_emitter_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.emitter_id, "test_emitter_1")
        self.assertEqual(retrieved.mean_pri_us, 1000.0)
        self.assertEqual(retrieved.freq_min_mhz, 5000.0)
        self.assertEqual(retrieved.freq_max_mhz, 5500.0)
        self.assertEqual(retrieved.priority_score, 0.7)
        self.assertEqual(retrieved.is_periodic, 1)

    def test_update_priority_on_intercept(self):
        """Test priority boost on intercept."""
        profile = EmitterProfile(
            emitter_id="emitter_1",
            priority_score=0.5,
        )
        self.memory.write_emitter(profile)
        
        # Simulate intercept
        self.memory.update_priority("emitter_1", intercept_occurred=True)
        
        updated = self.memory.get_emitter("emitter_1")
        self.assertGreater(updated.priority_score, 0.5)
        self.assertEqual(updated.intercept_count, 1)

    def test_update_priority_on_miss(self):
        """Test priority decay on miss."""
        profile = EmitterProfile(
            emitter_id="emitter_2",
            priority_score=0.8,
        )
        self.memory.write_emitter(profile)
        
        # Simulate miss
        self.memory.update_priority("emitter_2", intercept_occurred=False)
        
        updated = self.memory.get_emitter("emitter_2")
        self.assertLess(updated.priority_score, 0.8)

    def test_priority_bounds(self):
        """Test that priority stays within [0, 1]."""
        profile = EmitterProfile(
            emitter_id="emitter_3",
            priority_score=0.99,
        )
        self.memory.write_emitter(profile)
        
        # Multiple intercepts should not exceed 1.0
        for _ in range(10):
            self.memory.update_priority("emitter_3", intercept_occurred=True)
        
        updated = self.memory.get_emitter("emitter_3")
        self.assertLessEqual(updated.priority_score, 1.0)
        
        # Multiple misses should not go below 0.0
        for _ in range(20):
            self.memory.update_priority("emitter_3", intercept_occurred=False)
        
        updated = self.memory.get_emitter("emitter_3")
        self.assertGreaterEqual(updated.priority_score, 0.0)

    def test_list_emitters(self):
        """Test listing all emitters."""
        for i in range(5):
            profile = EmitterProfile(
                emitter_id=f"emitter_{i}",
                priority_score=0.5,
            )
            self.memory.write_emitter(profile)
        
        emitters = self.memory.list_emitters()
        self.assertEqual(len(emitters), 5)

    def test_get_band_priority_boost(self):
        """Test band priority boost from known emitter frequencies."""
        # Add emitters in different frequency ranges
        profile1 = EmitterProfile(
            emitter_id="emitter_low",
            freq_min_mhz=1000.0,
            freq_max_mhz=2000.0,
            priority_score=0.8,
        )
        profile2 = EmitterProfile(
            emitter_id="emitter_high",
            freq_min_mhz=8000.0,
            freq_max_mhz=9000.0,
            priority_score=0.6,
        )
        self.memory.write_emitter(profile1)
        self.memory.write_emitter(profile2)
        
        # Get boost for 36 bands covering 0-18000 MHz
        boost = self.memory.get_band_priority_boost(n_bands=36, freq_min=0.0, freq_max=18000.0)
        
        # Band covering 1000-2000 MHz should have boost
        # Band width = 18000/36 = 500 MHz
        # Band 2: 1000-1500, Band 3: 1500-2000
        self.assertGreater(boost[2], 0.0)
        self.assertGreater(boost[3], 0.0)
        
        # Band covering 8000-9000 MHz should have boost
        # Band 16: 8000-8500, Band 17: 8500-9000
        self.assertGreater(boost[16], 0.0)
        self.assertGreater(boost[17], 0.0)
        
        # Other bands should have zero boost
        self.assertEqual(boost[0], 0.0)
        self.assertEqual(boost[10], 0.0)

    def test_upsert_emitter_replaces(self):
        """Test that upserting same emitter_id replaces the profile."""
        profile1 = EmitterProfile(
            emitter_id="emitter_x",
            priority_score=0.5,
        )
        self.memory.write_emitter(profile1)
        
        profile2 = EmitterProfile(
            emitter_id="emitter_x",
            priority_score=0.9,
        )
        self.memory.write_emitter(profile2)
        
        retrieved = self.memory.get_emitter("emitter_x")
        self.assertEqual(retrieved.priority_score, 0.9)

    def test_close_closes_connection(self):
        """Test that close() closes the database connection."""
        self.memory.close()
        # Should not raise
        self.memory.close()


if __name__ == "__main__":
    unittest.main()
"""
Gymnasium Environment for RF Scanning Scheduler

Simulates a narrow-band Electronic Support (ES) receiver sweeping across 
the electromagnetic spectrum to intercept radar pulses.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import random
from pathlib import Path

# Try importing the Challenge dataset. 
try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    from typing import Any
    PulseTrain = Any

from .state_matrix import build_transmission_matrix, get_pdws_in_band
from ..evaluation.metrics import FiguresOfMerit

logger = logging.getLogger(__name__)

class RFScanEnv(gym.Env):
    """
    Cognitive EW Environment simulating a frequency-scanning ES receiver.
    
    The agent chooses a frequency band to monitor for a given dwell time.
    It receives a reward based on intercepting novel emitters, prioritizing 
    threats, and minimizing missed intercepts.
    """
    
    def __init__(self, config: dict, data_dir: str | Path, subset: str = 'train'):
        super().__init__()
        
        self.n_bands = config.get("n_bands", 180)
        self.freq_min = config.get("freq_min_mhz", 0.0)
        self.freq_max = config.get("freq_max_mhz", 18000.0)
        self.time_resolution = config.get("time_resolution_us", 100.0)
        
        self.dwell_slots = config.get("dwell_slots", 5)
        self.dwell_time_us = self.dwell_slots * self.time_resolution
        
        # Reward function weights
        self.w1 = config.get("w1", 5.0)  # Novel intercepts
        self.w2 = config.get("w2", 8.0)  # Priority hits (stubbed for future integration)
        self.w3 = config.get("w3", 0.1)  # Time error penalty
        self.w4 = config.get("w4", 4.0)  # Miss penalty
        
        # Action space: which frequency band to tune to
        self.action_space = spaces.Discrete(self.n_bands)
        
        # Observation space: 2 * n_bands
        # [0:n_bands] -> binary occupancy estimate from last visit
        # [n_bands:2*n_bands] -> normalized time since last visit
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.n_bands,), dtype=np.float32
        )
        
        # Load file list lazily to prevent OOM
        # Assuming HDF5 files are in data_dir / stare / subset
        self.data_dir = Path(data_dir) / 'stare' / subset
        self.file_list = list(self.data_dir.glob("*.h5")) if self.data_dir.exists() else []
        if not self.file_list:
            logger.warning(f"No .h5 files found in {self.data_dir}. Ensure data is downloaded.")
            
        # State variables
        self.current_pt = None
        self.current_pt_min_toa = 0.0
        self.transmission_matrix = None
        self.current_slot = 0
        self.total_slots = 0
        
        # Belief state tracking
        self.occupancy_estimate = np.zeros(self.n_bands, dtype=np.float32)
        self.last_visit_time = np.zeros(self.n_bands, dtype=np.float32)
        
        # Tracking intercepts and performance
        self.intercepted_emitters = set()
        self.fom = FiguresOfMerit()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly sample a pulse train file
        if not self.file_list:
            # Fallback for testing without dataset
            # Provide an empty state 
            self.transmission_matrix = np.zeros((100, self.n_bands), dtype=np.int8)
            self.total_slots = 100
        else:
            file_path = self.np_random.choice(self.file_list)
            self.current_pt = PulseTrain.load(file_path)
            
            # Extract ground truth matrix (Oracle Stare Mode)
            self.transmission_matrix = build_transmission_matrix(
                self.current_pt, self.n_bands, self.time_resolution, self.freq_min, self.freq_max
            )
            self.total_slots = self.transmission_matrix.shape[0]
            if len(self.current_pt) > 0:
                self.current_pt_min_toa = np.min(self.current_pt.data[:, 0])
            else:
                self.current_pt_min_toa = 0.0
                
        # Reset trackers
        self.current_slot = 0
        self.occupancy_estimate.fill(0.0)
        self.last_visit_time.fill(0.0)
        self.intercepted_emitters.clear()
        self.fom.reset()
        
        return self._get_obs(), {}

    def step(self, action: int):
        """
        Advances the environment by dwell_slots by tuning into the specified band.
        """
        if self.current_slot >= self.total_slots:
            return self._get_obs(), 0.0, True, False, {}
            
        t_start_idx = self.current_slot
        t_end_idx = min(self.current_slot + self.dwell_slots, self.total_slots)
        
        # 1. Oracle Check (Did ANY pulse exist in the chosen band during this dwell?)
        # For simplicity, if any slot in the dwell period has a pulse, we consider it occupied.
        ground_truth_slice = self.transmission_matrix[t_start_idx:t_end_idx, action]
        hit = np.any(ground_truth_slice)
        
        # 2. Extract specific PDWs intercepted for downstream cognitive tracking
        t_start_us = self.current_pt_min_toa + (t_start_idx * self.time_resolution)
        t_end_us = self.current_pt_min_toa + (t_end_idx * self.time_resolution)
        
        pdws, labels = (np.empty((0, 5)), np.empty((0,)))
        if self.current_pt is not None:
            pdws, labels = get_pdws_in_band(
                self.current_pt, action, t_start_us, t_end_us, 
                self.n_bands, self.freq_min, self.freq_max
            )
        
        # 3. Update Belief State
        self.occupancy_estimate[action] = 1.0 if hit else 0.0
        self.last_visit_time[action] = self.current_slot
        
        # 4. Compute Reward
        reward, new_emitters_found = self._compute_reward(action, t_start_idx, t_end_idx, hit, labels)
        self.intercepted_emitters.update(new_emitters_found)
        
        # 5. Update Metrics (FoM)
        # Note: True positive logic. 
        # For full FoM, we need to compare predicted vs actual across all bands.
        # This basic step just logs the single action context.
        true_active = np.any(self.transmission_matrix[t_start_idx:t_end_idx, :], axis=0)
        pred_active = np.zeros(self.n_bands, dtype=bool)
        pred_active[action] = True
        
        # Arbitrary intercept time error for now, usually computed by MoE
        intercept_time_error = 0.0 
        self.fom.update(action, true_active[action], True, intercept_time_error, float(reward))
        
        # Advance time
        self.current_slot += self.dwell_slots
        done = self.current_slot >= self.total_slots
        
        info = {
            "hit": bool(hit),
            "intercepted_pdws": pdws.shape[0],
            "novel_emitters": list(new_emitters_found)
        }
        
        return self._get_obs(), reward, done, False, info

    def _compute_reward(self, action: int, t_start: int, t_end: int, hit: bool, labels: np.ndarray) -> tuple[float, set]:
        """Calculates the domain-specific reward based on the scheduler's performance."""
        reward = 0.0
        new_emitters = set()
        
        if hit:
            # Check for novel intercepts (emitters we haven't seen in this episode)
            unique_labels = set(labels[labels >= 0])
            for lbl in unique_labels:
                if lbl not in self.intercepted_emitters:
                    reward += self.w1
                    new_emitters.add(lbl)
                # If priority tracking was implemented, add W2 here
        else:
            # We missed. Did something happen elsewhere?
            # Check if any OTHER band had pulses
            missed_opportunity = np.any(self.transmission_matrix[t_start:t_end, :])
            if missed_opportunity:
                reward -= self.w4
                
        # W3 Intercept time penalty (placeholder if prediction was late)
        # In this basic step, we assume time error is 0 unless guided by MoE Preemptive logic.
        
        return reward, new_emitters

    def _get_obs(self) -> np.ndarray:
        """Constructs the observation state vector."""
        # Calculate normalized time since last visit
        time_since = (self.current_slot - self.last_visit_time) / max(self.total_slots, 1)
        
        # Concatenate occupancy and time_since
        obs = np.concatenate([self.occupancy_estimate, time_since])
        return obs.astype(np.float32)
        
    def get_fom(self) -> FiguresOfMerit:
        """Returns the current figures of merit tracking object."""
        return self.fom

    def render(self):
        """Provides a simple terminal ASCII visualization of the spectrum."""
        if self.current_slot == 0:
            return
            
        print(f"\n--- Timestep: {self.current_slot}/{self.total_slots} ---")
        
        # Display the 180 bands as blocks. '█' for occupied, '.' for empty.
        viz = ""
        for b in range(self.n_bands):
            viz += "█" if self.occupancy_estimate[b] > 0.5 else "."
            
        print("Spectrum State: " + viz)
        print(f"Intercepted Emitters: {len(self.intercepted_emitters)}")
        print("-" * 50)

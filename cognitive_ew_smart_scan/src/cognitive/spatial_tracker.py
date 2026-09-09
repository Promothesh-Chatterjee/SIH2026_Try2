from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SpatialBelief:
    track_id: int
    mean_aoa_deg: float
    circular_variance: float
    confidence: float
    last_update_us: float
    sector_index: int
    aoa_samples: list[float] = field(default_factory=list)

class SpatialTracker:
    def __init__(self, n_sectors: int = 12, max_history: int = 50) -> None:
        self.n_sectors = n_sectors
        self.sector_width_deg = 360.0 / n_sectors
        self.max_history = max_history
        self.beliefs: dict[int, SpatialBelief] = {}

    @staticmethod
    def compute_circular_mean_and_r(angles_deg: list[float]) -> tuple[float, float]:
        if not angles_deg:
            return 0.0, 0.0
        rads = np.deg2rad(angles_deg)
        sin_sum = float(np.sum(np.sin(rads)))
        cos_sum = float(np.sum(np.cos(rads)))
        n = len(angles_deg)
        r = math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / max(1, n)
        r = min(1.0, max(0.0, r))
        mean_rad = math.atan2(sin_sum, cos_sum)
        mean_deg = (math.degrees(mean_rad)) % 360.0
        return mean_deg, r

    def update_from_track(self, track_id: int, new_aoa_deg: float, current_time_us: float) -> SpatialBelief:
        if track_id not in self.beliefs:
            self.beliefs[track_id] = SpatialBelief(
                track_id=track_id,
                mean_aoa_deg=new_aoa_deg % 360.0,
                circular_variance=0.0,
                confidence=1.0,
                last_update_us=current_time_us,
                sector_index=int((new_aoa_deg % 360.0) // self.sector_width_deg),
                aoa_samples=[new_aoa_deg % 360.0],
            )
            return self.beliefs[track_id]

        sb = self.beliefs[track_id]
        sb.aoa_samples.append(new_aoa_deg % 360.0)
        if len(sb.aoa_samples) > self.max_history:
            sb.aoa_samples = sb.aoa_samples[-self.max_history:]

        mean_deg, r = self.compute_circular_mean_and_r(sb.aoa_samples)
        sb.mean_aoa_deg = mean_deg
        sb.confidence = r
        sb.circular_variance = 1.0 - r
        sb.last_update_us = current_time_us
        sb.sector_index = int(mean_deg // self.sector_width_deg) % self.n_sectors
        return sb

    def get_spatial_priority(self, track_id: int, current_time_us: float) -> float:
        sb = self.beliefs.get(track_id)
        if not sb:
            return 0.0
        dt = max(0.0, current_time_us - sb.last_update_us)
        decay = math.exp(-dt / 100000.0)
        return float(min(1.0, max(0.0, sb.confidence * decay)))

    def prune_stale(self, max_age_us: float, current_time_us: float) -> None:
        stale = [tid for tid, sb in self.beliefs.items() if (current_time_us - sb.last_update_us) > max_age_us]
        for tid in stale:
            del self.beliefs[tid]

    @property
    def tracks(self) -> dict[int, SpatialBelief]:
        return self.beliefs

"""CPU Runtime Latency & Memory Profiler for Cognitive EW SmartScan.

Profiles P50, P95, P99 latency and resident memory footprint on CPU.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import psutil

# Ensure paths
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("cognitive_ew_smart_scan").resolve()))

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler

def profile_cpu_runtime():
    frozen_ckpt = Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
    ckpt = torch.load(frozen_ckpt, map_location="cpu", weights_only=False)
    
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()
    
    hidden = drqn.init_hidden(1, "cpu")
    dummy_obs = torch.randn(360, dtype=torch.float32)
    
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _, hidden = drqn.act(dummy_obs, hidden)
            
    # Measure 500 decision steps
    latencies_ms = []
    process = psutil.Process(os.getpid())
    
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            _, hidden = drqn.act(dummy_obs, hidden)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    rss_mb = process.memory_info().rss / (1024 * 1024)
    
    print("\n" + "="*70)
    print("CPU RUNTIME & LATENCY BENCHMARK PROFILE")
    print("="*70)
    print(f"Device:               CPU")
    print(f"Decisions profiled:   500 steps")
    print(f"Mean Latency:         {mean_lat:.3f} ms ({mean_lat*1000:.1f} us)")
    print(f"P50 Latency:          {p50:.3f} ms ({p50*1000:.1f} us)")
    print(f"P95 Latency:          {p95:.3f} ms ({p95*1000:.1f} us)")
    print(f"P99 Latency:          {p99:.3f} ms ({p99*1000:.1f} us)")
    print(f"Process Resident RAM: {rss_mb:.1f} MB")
    print(f"Base Dwell Budget:    500.0 us (0.5 ms) - 2000.0 us (2.0 ms)")
    print("="*70 + "\n")

if __name__ == "__main__":
    profile_cpu_runtime()

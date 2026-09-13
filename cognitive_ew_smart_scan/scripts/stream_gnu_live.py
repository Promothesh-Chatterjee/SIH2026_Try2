import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def generate_gnu_pulses(gt_file: str, time_horizon_us: float = 600_000.0) -> List[Dict[str, Any]]:
    print(f"Loading GNU Ground Truth: {gt_file}")
    with open(gt_file, 'r') as f:
        gt = json.load(f)
    
    # Emitters are mostly static per scenario. We'll extract them from the first dwell.
    emitters = gt['dwells'][0]['emitters']
    print(f"Found {len(emitters)} emitters in scenario.")
    
    records = []
    
    for em in emitters:
        rng = np.random.default_rng(em['configured_seed'])
        pri = em['pri_us']
        pw = em['pulse_width_us']
        freq = em['rf_frequency_mhz']
        amp = em['amplitude'] * -50.0 # Convert linear to rough dBm
        jitter_frac = em['jitter_fraction']
        
        t = rng.uniform(0.0, pri) # random start phase
        
        while t < time_horizon_us:
            # Jitter
            j = rng.uniform(-jitter_frac, jitter_frac) * pri
            t_pulse = t + j
            
            if t_pulse < time_horizon_us:
                records.append({
                    "toa_us": float(t_pulse),
                    "frequency_mhz": float(freq),
                    "pulse_width_us": float(pw),
                    "amplitude_db": float(amp),
                    "aoa_deg": 15.0 # hardcoded
                })
            
            t += pri
            
    # Sort by TOA
    records.sort(key=lambda x: x['toa_us'])
    return records

def stream_gnu_live(gt_file: str):
    pulses = generate_gnu_pulses(gt_file, time_horizon_us=1_000_000.0)
    print(f"Generated {len(pulses)} raw pulses from GNU Emitter definitions.")
    print("Streaming to backend... Press Ctrl+C to stop.")
    
    API_URL = "http://127.0.0.1:8000/mission/step"
    chunk_size_us = 5000.0 # 5 ms chunks
    current_time_us = 0.0
    idx = 0
    
    try:
        while idx < len(pulses):
            end_time = current_time_us + chunk_size_us
            chunk = []
            
            while idx < len(pulses) and pulses[idx]["toa_us"] < end_time:
                chunk.append(pulses[idx])
                idx += 1
                
            payload = {
                "mission_id": "GNU-LIVE-01",
                "current_time_us": float(end_time),
                "pdws": chunk
            }
            
            try:
                resp = requests.post(API_URL, json=payload, timeout=2.0)
                if resp.status_code == 200:
                    pass
                else:
                    print(f"Server returned {resp.status_code}")
            except Exception as e:
                print(f"Failed to reach backend: {e}")
                
            current_time_us = end_time
            time.sleep(0.05) # Throttle to 50ms real-time delay per 5ms simulation block (10x slow-mo for vis)
            
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")

if __name__ == '__main__':
    import argparse
    repo_root = Path(__file__).resolve().parents[2]
    default_gt = repo_root / "GNU_RF_ENV" / "p3ac_50k" / "episodes" / "EP000001.gt.json"
    if not default_gt.exists():
        default_gt = Path('C:/HACKATHONS/SIH2026_Try2/GNU_RF_ENV/p3ac_50k/episodes/EP000001.gt.json')

    parser = argparse.ArgumentParser(description="Live Stream Physical GNU Emitters to Backend")
    parser.add_argument("--gt-file", "--scenario", default=str(default_gt), help="Path to GNU .gt.json file")
    args = parser.parse_args()

    stream_gnu_live(args.gt_file)


"""
Diagnostic Dump for Phase 4.2 Hopper Audit.

Step 1: Verify PDW frequencies for multiple pulses [3400, 3500, 3600, 3700 MHz].
Step 2: Verify Track frequency history during runtime.
Step 3: Verify State Builder active bands and observation collapse.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cognitive_ew_smart_scan"))
sys.path.insert(0, str(REPO_ROOT / "cognitive_ew_smart_scan" / "src"))
sys.path.insert(0, str(REPO_ROOT / "receiver_env"))

from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig
from receiver_env.deinterleaver.deinterleaver import PDWDeinterleaver
from src.contracts import CANONICAL_N_BANDS
from src.operational.state_builder import OperationalStateBuilder
from src.perception.emitter_tracker import EmitterTracker


def run_diagnostic():
    print("=" * 80)
    print("  PHASE 4.2 DIAGNOSTIC DUMP: HOPPER VERIFICATION")
    print("=" * 80)

    # 1. Generate hopping pulse sequence (3400, 3500, 3600, 3700 MHz)
    hops = [3400.0, 3500.0, 3600.0, 3700.0]
    pri_us = 100.0
    pw_us = 10.0
    total_pulses = 60

    pdws: list[PDW] = []
    for i in range(total_pulses):
        f = hops[i % len(hops)]
        t = float(i * pri_us)
        pdws.append(
            PDW(
                pdw_id=1000 + i + 1,
                pulse_id=i + 1,
                toa_us=t,
                pulse_width_us=pw_us,
                frequency_mhz=f,
                amplitude_db=-50.0,
                confidence=0.95,
                receiver_id="RX_01",
                generation_timestamp_us=t + 1.0,
            )
        )

    print("\n[DIAGNOSTIC 1] PDW Frequencies (First 20 Pulses):")
    print(f"{'Pulse ID':>10} | {'PDW ID':>10} | {'ToA (us)':>12} | {'Frequency (MHz)':>18}")
    print("-" * 60)
    for pdw in pdws[:20]:
        print(f"{pdw.pulse_id:>10} | {pdw.pdw_id:>10} | {pdw.toa_us:>12.1f} | {pdw.frequency_mhz:>18.1f}")

    # Verify PDWs show hop frequencies
    pdw_freqs = [p.frequency_mhz for p in pdws[:8]]
    expected_hops = hops * 2
    assert pdw_freqs == expected_hops, f"PDWs do not show hop pattern: {pdw_freqs}"
    print("\n[+] CONFIRMED: PDWs accurately contain the hopping sequence [3400, 3500, 3600, 3700, ...]")

    # 2. Feed into Phase 4 PDWDeinterleaver
    print("\n" + "-" * 80)
    print("[DIAGNOSTIC 2] Processing PDWs through Phase 4 Deinterleaver...")
    deint = PDWDeinterleaver(DeinterleaverConfig())
    tracks = deint.process_batch(pdws)

    print(f"Deinterleaver produced {len(tracks)} active tracks:")
    for t in tracks:
        freq_hist = list(t.history.recent_frequency_mhz)
        print(f"  Track #{t.track_id}: Mean Freq={t.mean_frequency_mhz:.2f} MHz, Pulses={t.pulse_count}, PRI={t.estimated_pri_us:.1f} us")
        print(f"    History length: {len(freq_hist)}, Recent frequencies: {freq_hist[-8:]}")

    # 3. Feed into EmitterTracker & OperationalStateBuilder
    print("\n" + "-" * 80)
    print("[DIAGNOSTIC 3] Feeding Tracks into OperationalStateBuilder...")
    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS, recent_hop_window_us=10_000.0)
    emitter_tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS, max_misses_before_drop=100)

    # Ingest each pulse through emitter tracker as done in live receiver controller
    for p in pdws:
        d_pdw = {
            "frequency_mhz": p.frequency_mhz,
            "aoa_deg": 30.0,
            "time_us": p.toa_us,
            "pulse_width_us": p.pulse_width_us,
            "amplitude_db": p.amplitude_db,
            "band": int(p.frequency_mhz // 500.0),
        }
        labels = np.array([0], dtype=np.int64)
        toa_us = np.array([p.toa_us], dtype=np.float64)
        freq_mhz = np.array([p.frequency_mhz], dtype=np.float64)
        aoa_deg = np.array([30.0], dtype=np.float64)
        pw_us = np.array([p.pulse_width_us], dtype=np.float64)
        amp_db = np.array([p.amplitude_db], dtype=np.float64)

        emitter_tracker.update_from_deinterleaver(
            labels=labels,
            toa_us=toa_us,
            freq_mhz=freq_mhz,
            aoa_deg=aoa_deg,
            pw_us=pw_us,
            amp_db=amp_db,
            current_time=p.toa_us,
            band=int(p.frequency_mhz // 500.0),
            min_cluster_size=1,
        )

        state_builder.record_dwell_outcome(
            band=int(p.frequency_mhz // 500.0),
            hit=True,
            current_time_us=p.toa_us,
            num_pulses=1,
        )

    print(f"EmitterTracker tracks: {len(emitter_tracker.tracks)}")
    for tid, trk in emitter_tracker.tracks.items():
        print(f"  Track #{tid}: current_frequency_mhz = {trk.current_frequency_mhz:.2f} MHz, latest = {trk.latest_frequency_mhz:.2f} MHz")
        print(f"    Frequency history (last 12): {trk.frequency_history[-12:]}")
        print(f"    Frequency span = {trk.frequency_span_mhz:.2f} MHz, Hopping detected = {trk.frequency_hopping_detected}, Hop rate = {trk.hop_rate_hz:.1f} Hz")

    # Build state vector in StateBuilder
    sim_end_time = float(total_pulses * pri_us)
    state_vector = state_builder.build_state(current_time_us=sim_end_time, active_tracks=emitter_tracker.tracks)
    state_matrix = state_vector.reshape(CANONICAL_N_BANDS, 10)

    # Check which bands have emitter_count > 0 in state_matrix (feature index 5 is emitter_count)
    bands_with_emitters = [b for b in range(CANONICAL_N_BANDS) if state_matrix[b, 5] > 0.0]
    print(f"\nBands with Emitters in State Matrix (Feature 5): {bands_with_emitters}")
    for b in bands_with_emitters:
        cf = b * 500.0 + 250.0
        print(f"  Band {b}: Center Freq = {cf:.1f} MHz, Emitter Count = {state_matrix[b, 5]:.2f}, Occupancy = {state_matrix[b, 0]:.2f}, Agility = {state_matrix[b, 8]:.2f}")

    print("\nExpected Bands for 3400, 3500, 3600, 3700 MHz:")
    print("  3400 MHz -> Band 6 (3000 - 3500 MHz)")
    print("  3500 MHz -> Band 7 (3500 - 4000 MHz)")
    print("  3600 MHz -> Band 7 (3500 - 4000 MHz)")
    print("  3700 MHz -> Band 7 (3500 - 4000 MHz)")
    print(f"Actual bands populated in State Matrix: {bands_with_emitters}")

    if 6 in bands_with_emitters and 7 in bands_with_emitters:
        print("\n[+] SUCCESS: Both Band 6 and Band 7 are active in the OperationalStateBuilder!")
        print("    Frequency hopper information is properly preserved in the DRQN perception vector.")
    elif bands_with_emitters == [7]:
        print("\n[!] BUG PERSISTS: Band 6 is still missing.")


if __name__ == "__main__":
    run_diagnostic()

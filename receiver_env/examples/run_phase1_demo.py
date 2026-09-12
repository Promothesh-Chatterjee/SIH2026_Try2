"""Phase 1 EW Receiver Front End & Pulse Detector - End-to-End Live Demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure workspace root is on sys.path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.frontend import ReceiverFrontend
from receiver_env.pulse_detector.detector import PulseDetector
from receiver_env.validation.metrics import evaluate_detections
from receiver_env.validation.validation_runner import (
    SyntheticSignalGenerator,
    ValidationRunner,
)


def main() -> None:
    print("=" * 70)
    print("  COGNITIVE EW RECEIVER - PHASE 1 ARCHITECTURE DEMO")
    print("  Subsystem: Receiver Front End & Pulse Detection (Strict Phase 1)")
    print("=" * 70)

    # 1. Receiver Configuration
    config = ReceiverConfig(
        sample_rate_hz=20_000_000.0,         # 20 MSamples/sec (50 ns resolution)
        center_frequency_hz=3_000_000_000.0, # 3.0 GHz RF center
        bandwidth_hz=10_000_000.0,           # 10.0 MHz IF channel filter
        snr_margin_db=8.0,                   # CFAR threshold margin over noise floor
        hysteresis_db=3.0,                   # Release threshold drop (anti-splitting)
        min_pulse_samples=10,                # Minimum pulse duration filter (anti-glitch)
        min_gap_samples=15,                  # Minimum gap between distinct pulses (anti-merging)
    )

    print("\n[1] Initialized Receiver Configuration:")
    for k, v in config.to_dict().items():
        print(f"    - {k:22s}: {v}")

    # 2. Synthesize Validation Test Scenario
    # Generate 5 pulses with varying carrier offsets and pulse widths across continuous chunks
    gen = SyntheticSignalGenerator(
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        seed=42,
    )

    print("\n[2] Generating Synthetic EW RF Scenario with 5 Radar Pulses...")
    iq_samples, truth_pulses = gen.generate_pulse_train(
        num_pulses=5,
        pri_us=100.0,       # 100 us PRI (2000 samples)
        pw_us=8.0,          # 8 us PW (160 samples)
        snr_db=15.0,        # 15 dB SNR
        freq_offset_hz=1.5e6, # 1.5 MHz IF carrier offset
        initial_delay_us=25.0, # 25 us lead-in
    )

    chunk_size = 2048  # Streamed in chunks
    runner = ValidationRunner(config=config, chunk_size=chunk_size)

    print(f"    - Generated {len(iq_samples):,} IQ samples ({len(iq_samples) / config.sample_rate_hz * 1e3:.2f} ms)")
    print(f"    - Sliced into {int(np.ceil(len(iq_samples)/chunk_size)) if 'np' in globals() else len(list(runner.chunk_stream(iq_samples)))} streaming chunks of {chunk_size} samples each")
    print(f"    - Ground truth pulses: {len(truth_pulses)}")

    # 3. Stream through Frontend & Pulse Detector
    print("\n[3] Streaming Chunks through ReceiverFrontend & PulseDetector...")
    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)

    detected_pulses = []
    for chunk_idx, chunk in enumerate(runner.chunk_stream(iq_samples)):
        frontend_out = frontend.process_chunk(chunk)
        completed = detector.process_chunk(frontend_out)
        if completed:
            for p in completed:
                print(f"    [Chunk {chunk_idx:02d} | Offset {chunk.global_sample_offset:05d}] >> Completed Pulse #{p.pulse_id}: "
                      f"samples [{p.global_start_sample}..{p.global_end_sample}] "
                      f"(dur={p.global_end_sample - p.global_start_sample + 1} spl, peak={p.peak_magnitude:.3f}, conf={p.confidence:.2f})")
            detected_pulses.extend(completed)

    # Flush detector at end of stream
    flushed = detector.flush()
    if flushed:
        for p in flushed:
            print(f"    [FLUSH] >> Completed Pulse #{p.pulse_id}: samples [{p.global_start_sample}..{p.global_end_sample}]")
        detected_pulses.extend(flushed)

    # 4. Output Legal Phase 1 Deliverable JSON
    print("\n[4] Emitted DetectedPulses (Phase 1 JSON Specification):")
    print("=" * 70)
    for p in detected_pulses:
        print(p.to_json())
    print("=" * 70)

    # 5. Evaluate Against Ground Truth and Print Scorecard
    print("\n[5] Evaluating Performance Against Ground Truth...")
    metrics = evaluate_detections(
        truth_pulses=truth_pulses,
        detected_pulses=detected_pulses,
        tolerance_samples=25,
        min_iou=0.3,
    )
    print(metrics.summary())


if __name__ == "__main__":
    main()

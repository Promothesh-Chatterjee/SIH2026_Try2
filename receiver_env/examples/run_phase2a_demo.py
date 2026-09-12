"""Phase 2A EW Receiver Parameter Extraction - End-to-End Live Demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

# Ensure workspace root is on sys.path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.frontend import ReceiverFrontend
from receiver_env.pulse_detector.detector import PulseDetector
from receiver_env.parameter_extractor.extractor import ParameterExtractor
from receiver_env.parameter_extractor.validation import Phase2AValidation
from receiver_env.validation.validation_runner import (
    SyntheticSignalGenerator,
    ValidationRunner,
)


def main() -> None:
    print("=" * 75)
    print("  COGNITIVE EW RECEIVER - PHASE 2A ARCHITECTURE DEMO")
    print("  Subsystem: Receiver Parameter Extraction (Strict Phase 2A Contract)")
    print("=" * 75)

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
        print(f"    - {k:24s}: {v}")

    # 2. Synthesize Validation EW Scenario
    gen = SyntheticSignalGenerator(
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        seed=42,
    )

    print("\n[2] Generating Synthetic EW RF Scenario with 6 Radar Pulses...")
    iq_samples, truth_pulses = gen.generate_pulse_train(
        num_pulses=6,
        pri_us=120.0,         # 120 us PRI (2400 samples)
        pw_us=15.0,           # 15 us PW (300 samples)
        snr_db=25.0,          # 25 dB SNR
        freq_offset_hz=1.0e6, # 1.0 MHz IF offset
        initial_delay_us=30.0, # 30 us lead-in
        amplitude=0.8,
    )

    chunk_size = 2048  # Streamed in discrete chunks
    runner = ValidationRunner(config=config, chunk_size=chunk_size)
    num_chunks = int(np.ceil(len(iq_samples) / chunk_size))

    print(f"    - Total IQ samples   : {len(iq_samples):,} ({len(iq_samples) / config.sample_rate_hz * 1e3:.2f} ms)")
    print(f"    - Streaming chunks   : {num_chunks} chunks of {chunk_size} samples")
    print(f"    - Ground truth pulses: {len(truth_pulses)}")

    # 3. Stream through Frontend -> Pulse Detector -> Parameter Extractor
    print("\n[3] Streaming Chunks through Pipeline:")
    print("    [IQ Chunk] -> [ReceiverFrontend] -> [PulseDetector] -> [ParameterExtractor] -> [PulseMeasurement]")

    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)
    extractor = ParameterExtractor(config)

    all_measurements = []
    chunk_stream = runner.chunk_stream(iq_samples)

    for chunk_idx, chunk in enumerate(chunk_stream):
        frontend_out = frontend.process_chunk(chunk)
        completed_pulses = detector.process_chunk(frontend_out)
        measurements = extractor.process_chunk(frontend_out, completed_pulses)

        if measurements:
            for m in measurements:
                print(
                    f"    [Chunk {chunk_idx:02d} | Offset {chunk.global_sample_offset:05d}] "
                    f">> Extracted Pulse #{m.pulse_id}: "
                    f"ToA={m.toa_us:8.2f} us | PW={m.pulse_width_us:6.2f} us | "
                    f"Amp={m.amplitude_db:6.2f} dBFS | Conf={m.confidence:.2f}"
                )
            all_measurements.extend(measurements)

    flushed_pulses = detector.flush()
    if flushed_pulses:
        flushed_meas = extractor.process_chunk(frontend_out, flushed_pulses)
        for m in flushed_meas:
            print(
                f"    [FLUSH] >> Extracted Pulse #{m.pulse_id}: "
                f"ToA={m.toa_us:8.2f} us | PW={m.pulse_width_us:6.2f} us | "
                f"Amp={m.amplitude_db:6.2f} dBFS | Conf={m.confidence:.2f}"
            )
        all_measurements.extend(flushed_meas)

    # 4. Output Legal Phase 2A Deliverable JSON
    print("\n[4] Emitted PulseMeasurement Deliverables (Strict Phase 2A JSON Specification):")
    print("=" * 75)
    for m in all_measurements:
        print(m.to_json())
    print("=" * 75)

    # 5. Evaluate Against Ground Truth and Print Phase 2A Scorecard
    print("\n[5] Evaluating Against Ground Truth (Phase 2A Exit Criteria):")
    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth_pulses,
        measurements=all_measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    print(metrics.summary())

    print("=" * 75)
    if metrics.meets_exit_criteria:
        print("  >>> PHASE 2A VALIDATION RESULT: PASSED (ALL EXIT CRITERIA MET) <<<")
    else:
        print("  >>> PHASE 2A VALIDATION RESULT: FAILED <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()

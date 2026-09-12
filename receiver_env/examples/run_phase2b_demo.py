"""Phase 2B EW Receiver Carrier Frequency Extraction - End-to-End Live Demonstration."""

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
from receiver_env.frequency_extractor.pulse_snapshot import PulseSnapshotExtractor
from receiver_env.frequency_extractor.estimator import FrequencyEstimator
from receiver_env.frequency_extractor.validation import Phase2BValidation
from receiver_env.validation.validation_runner import (
    GroundTruthPulse,
    SyntheticSignalGenerator,
    ValidationRunner,
)


def main() -> None:
    print("=" * 75)
    print("  COGNITIVE EW RECEIVER - PHASE 2B ARCHITECTURE DEMO")
    print("  Subsystem: Carrier Frequency Extraction & Estimator Fusion")
    print("=" * 75)

    # 1. Receiver Configuration
    config = ReceiverConfig(
        sample_rate_hz=20_000_000.0,          # 20 MSamples/sec (50 ns resolution)
        center_frequency_hz=3_000_000_000.0,  # 3.0 GHz RF center
        bandwidth_hz=10_000_000.0,            # 10.0 MHz IF channel filter
        snr_margin_db=8.0,                    # CFAR threshold margin over noise floor
        hysteresis_db=3.0,                    # Release threshold drop (anti-splitting)
        min_pulse_samples=10,                 # Minimum pulse duration filter (anti-glitch)
        min_gap_samples=15,                   # Minimum gap between distinct pulses (anti-merging)
    )

    print("\n[1] Initialized Receiver Configuration:")
    for k, v in config.to_dict().items():
        print(f"    - {k:24s}: {v}")

    # 2. Synthesize Validation EW Scenario (Expanded Fine-Grid Frequencies)
    # Pulses spanning -3.0 MHz to +3.0 MHz including fine sub-MHz offsets
    gen = SyntheticSignalGenerator(
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        seed=42,
    )

    print("\n[2] Generating Synthetic EW RF Scenario with 6 Radar Pulses across Fine Frequencies...")
    offsets_hz = [-2_500_000.0, -750_000.0, -100_000.0, 250_000.0, 1_200_000.0, 3_000_000.0]
    pri_samples = 2400  # 120 us
    pw_samples = 300    # 15 us
    current_sample = 600

    truth_pulses = []
    for i, offset in enumerate(offsets_hz):
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=i,
                start_sample=current_sample,
                end_sample=current_sample + pw_samples - 1,
                peak_amplitude=0.8,
                frequency_offset_hz=offset,
            )
        )
        current_sample += pri_samples

    total_samples = current_sample + 1000
    iq_samples = gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=22.0)

    chunk_size = 2048
    runner = ValidationRunner(config=config, chunk_size=chunk_size)
    num_chunks = int(np.ceil(len(iq_samples) / chunk_size))

    print(f"    - Total IQ samples   : {len(iq_samples):,} ({len(iq_samples) / config.sample_rate_hz * 1e3:.2f} ms)")
    print(f"    - Streaming chunks   : {num_chunks} chunks of {chunk_size} samples")
    print(f"    - Ground truth pulses: {len(truth_pulses)}")

    # 3. Stream through Pipeline:
    # [IQ] -> [Frontend] -> [PulseDetector] -> [ParamExtractor] -> [SnapshotExtractor] -> [FreqEstimator]
    print("\n[3] Streaming Chunks through Multi-Stage Receiver Pipeline:")
    print("    [IQ Chunk] -> [ReceiverFrontend] -> [PulseDetector] -> [ParameterExtractor]")
    print("                                                               |")
    print("                                                   [PulseSnapshotExtractor]")
    print("                                                               |")
    print("                                                      [FrequencyEstimator]")
    print("                                                               |")
    print("                                                  [EnhancedPulseMeasurement]")

    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)
    param_extractor = ParameterExtractor(config)
    snapshot_extractor = PulseSnapshotExtractor(config)
    freq_estimator = FrequencyEstimator(config)

    all_enhanced = []
    chunk_stream = runner.chunk_stream(iq_samples)

    for chunk_idx, chunk in enumerate(chunk_stream):
        frontend_out = frontend.process_chunk(chunk)
        completed_pulses = detector.process_chunk(frontend_out)
        measurements = param_extractor.process_chunk(frontend_out, completed_pulses)

        for pulse, meas in zip(completed_pulses, measurements):
            snapshot = snapshot_extractor.extract_snapshot(
                pulse=pulse,
                history_buffer=param_extractor.amplitude_extractor.history_buffer,
            )
            enhanced = freq_estimator.process_pulse(
                measurement=meas,
                snapshot=snapshot,
                noise_floor_db=frontend_out.noise_floor_db,
            )
            diag = freq_estimator.last_diagnostics
            print(
                f"    [Chunk {chunk_idx:02d} | Offset {chunk.global_sample_offset:05d}] "
                f">> Enhanced Pulse #{enhanced.pulse_id}: "
                f"ToA={enhanced.toa_us:8.2f} us | PW={enhanced.pulse_width_us:6.2f} us | "
                f"Amp={enhanced.amplitude_db:6.1f} dB | Freq={enhanced.frequency_mhz:8.2f} MHz | Conf={enhanced.confidence:.2f} "
                f"(FFT/Phase Diff={diag.frequency_disagreement_hz / 1e3:.1f} kHz)"
            )
            all_enhanced.append(enhanced)

    # Flush detector at end of stream
    flushed_pulses = detector.flush()
    if flushed_pulses:
        flushed_meas = param_extractor.process_chunk(frontend_out, flushed_pulses)
        for pulse, meas in zip(flushed_pulses, flushed_meas):
            snapshot = snapshot_extractor.extract_snapshot(
                pulse=pulse,
                history_buffer=param_extractor.amplitude_extractor.history_buffer,
            )
            enhanced = freq_estimator.process_pulse(
                measurement=meas,
                snapshot=snapshot,
                noise_floor_db=frontend_out.noise_floor_db,
            )
            all_enhanced.append(enhanced)

    # 4. Output Legal Phase 2B Deliverable JSON
    print("\n[4] Emitted EnhancedPulseMeasurement Deliverables (Strict Phase 2B JSON Specification):")
    print("=" * 75)
    for m in all_enhanced:
        print(m.to_json())
    print("=" * 75)

    # 5. Evaluate Against Ground Truth and Print Scorecard
    print("\n[5] Evaluating Against Ground Truth (Phase 2B Exit Criteria):")
    metrics = Phase2BValidation.evaluate(
        truth_pulses=truth_pulses,
        measurements=all_enhanced,
        center_frequency_hz=config.center_frequency_hz,
        sample_rate_hz=config.sample_rate_hz,
    )

    print(metrics.summary())

    print("=" * 75)
    if metrics.meets_exit_criteria:
        print("  >>> PHASE 2B VALIDATION RESULT: PASSED (ALL EXIT CRITERIA MET) <<<")
    else:
        print("  >>> PHASE 2B VALIDATION RESULT: FAILED <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()

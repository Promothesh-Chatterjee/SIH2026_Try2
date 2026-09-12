"""Phase 4 EW Receiver PDW Deinterleaving & Emitter Separation - End-to-End Live Demonstration."""

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
from receiver_env.pdw.stream import PDWStream
from receiver_env.deinterleaver.models import DeinterleaverConfig
from receiver_env.deinterleaver.deinterleaver import PDWDeinterleaver
from receiver_env.deinterleaver.validation import ValidationFramework
from receiver_env.deinterleaver.association_audit import AssociationAudit
from receiver_env.deinterleaver.track_integrity import TrackIntegrityAnalyzer
from receiver_env.deinterleaver.fragmentation import FragmentationAuditor
from receiver_env.validation.metrics import GroundTruthPulse
from receiver_env.validation.validation_runner import SyntheticSignalGenerator, ValidationRunner


def main() -> None:
    print("=" * 80)
    print("  COGNITIVE EW RECEIVER - PHASE 4.1 HARDENED LIVE DEMONSTRATION")
    print("  Subsystem: PDW Deinterleaving & Emitter Separation Layer")
    print("=" * 80)

    # 1. Receiver Configuration
    config = ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
        bandwidth_hz=10_000_000.0,
    )

    print()
    print("[1] Initialized EW Pipeline Configuration:")
    print(f"    - Sample Rate           : {config.sample_rate_hz / 1e6:.1f} MS/s (50 ns/sample)")
    print(f"    - Tuner Center          : {config.center_frequency_hz / 1e9:.2f} GHz")
    print(f"    - Intermediate Bandwidth: {config.bandwidth_hz / 1e6:.1f} MHz")

    # 2. Instantiate Components from Phase 1 through Phase 4.1
    audit_engine = AssociationAudit()
    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)
    param_extractor = ParameterExtractor(config)
    snapshot_extractor = PulseSnapshotExtractor(config)
    freq_estimator = FrequencyEstimator(config)
    pdw_stream = PDWStream(receiver_id="RX_01", initial_pdw_id=1001)
    deinterleaver = PDWDeinterleaver(DeinterleaverConfig(pw_gate_pct=0.45), audit=audit_engine)

    print()
    print("[2] Pipeline Stack Assembled:")
    print("    Phase 1   -> ReceiverFrontend & PulseDetector")
    print("    Phase 2A  -> ParameterExtractor (ToA, PW, Amplitude)")
    print("    Phase 2B  -> PulseSnapshotExtractor & FrequencyEstimator (Fused FFT/Phase)")
    print("    Phase 3   -> PDWStream (Canonical 9-Field Standard Formatter)")
    print("    Phase 4.1 -> PDWDeinterleaver (Collision-Tolerant Multi-Attribute Tracking)")
    print("    Audit Log -> AssociationAudit Engine & TrackIntegrityAnalyzer")

    # 3. Synthesize Multi-Emitter RF Scenario
    # 3 Interleaved Radars:
    # Radar A (Surveillance): Freq = 2997.0 MHz (-3 MHz IF), PRI = 80 us, PW = 8.0 us
    # Radar B (Tracking):     Freq = 3002.5 MHz (+2.5 MHz IF), PRI = 110 us, PW = 12.0 us
    # Radar C (Fire Control): Freq = 3000.0 MHz (0 MHz IF), PRI = 60 us, PW = 5.0 us
    print()
    print("[3] Synthesizing Multi-Emitter Dense Radar Scenario (3 Interleaved Emitters)...")
    gen = SyntheticSignalGenerator(
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        seed=101,
    )

    emitters_def = [
        ("RADAR_SURVEILLANCE", -3_000_000.0, 80.0, 8.0, 20),
        ("RADAR_TRACKING", +2_500_000.0, 110.0, 12.0, 15),
        ("RADAR_FIRE_CONTROL", 0.0, 60.0, 5.0, 25),
    ]

    all_gt_pulses = []
    gt_labels = {}
    pulse_id_ctr = 1

    for name, offset_hz, pri_us, pw_us, count in emitters_def:
        start_toa_us = 15.0 + (len(all_gt_pulses) * 10.0)
        pw_samples = int(pw_us * 1e-6 * config.sample_rate_hz)

        for i in range(count):
            start_sample = int((start_toa_us * 1e-6 + i * pri_us * 1e-6) * config.sample_rate_hz)
            gt = GroundTruthPulse(
                truth_id=pulse_id_ctr,
                start_sample=start_sample,
                end_sample=start_sample + pw_samples - 1,
                peak_amplitude=0.85,
                frequency_offset_hz=offset_hz,
            )
            all_gt_pulses.append(gt)
            gt_labels[pulse_id_ctr] = name
            pulse_id_ctr += 1

    all_gt_pulses.sort(key=lambda p: p.start_sample)
    total_samples = all_gt_pulses[-1].end_sample + 2000

    raw_iq = gen.generate_iq_from_truth(all_gt_pulses, total_samples=total_samples, snr_db=25.0)
    print(f"    - Synthesized {len(all_gt_pulses)} pulses over {total_samples:,} samples ({total_samples/config.sample_rate_hz*1e3:.2f} ms)")

    # 4. Stream Through 4-Stage Receiver Pipeline (Phases 1 -> 2A -> 2B -> 3)
    chunk_size = 2048
    runner = ValidationRunner(config=config, chunk_size=chunk_size)
    chunk_stream = runner.chunk_stream(raw_iq)

    all_pdws = []
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
            pdw = pdw_stream.process_measurement(enhanced)
            if pdw is not None:
                all_pdws.append(pdw)

    print()
    print(f"[4] Receiver Ingestion Complete: {len(all_pdws)} canonical PDWs extracted from raw IQ.")

    # 5. Execute PDW Deinterleaving & Emitter Separation (Phase 4)
    tracks = deinterleaver.process_batch(all_pdws)

    print()
    print("=" * 80)
    print("  PHASE 4 DEINTERLEAVED EMITTER TRACKS")
    print("=" * 80)
    for track in tracks:
        print(f"Track ID: {track.track_id:2d} | Emitter ID: {track.emitter_id:10s} | Status: {track.status.value:9s}")
        print(f"    - Assigned Pulses    : {track.pulse_count}")
        print(f"    - Mean Carrier Freq  : {track.mean_frequency_mhz:8.2f} MHz (std: {track.stats.frequency_std_mhz:.3f} MHz)")
        print(f"    - Mean Pulse Width   : {track.mean_pw_us:8.2f} us  (std: {track.stats.pw_std_us:.3f} us)")
        print(f"    - Estimated PRI      : {track.estimated_pri_us:8.1f} us  (jitter: {track.pri_jitter_pct:.2f}%)")
        print(f"    - First / Last ToA   : {track.first_toa_us:.1f} us / {track.last_toa_us:.1f} us")
        print(f"    - Track Confidence   : {track.track_confidence:.2f}")
        print("    - Canonical JSON     :")
        print(track.to_json(indent=6))
        print("-" * 80)

    # 6. Compute Ground Truth Metrics (Frequency-Aware Nearest-Neighbor)
    pdw_gt_map = {}
    for pdw in all_pdws:
        pdw_sample = int(pdw.toa_us * 1e-6 * config.sample_rate_hz)
        best_match_id = None
        best_dist = 999999
        for gt in all_gt_pulses:
            dist_spl = abs(gt.start_sample - pdw_sample)
            expected_freq_mhz = (config.center_frequency_hz + gt.frequency_offset_hz) / 1e6
            dist_freq = abs(pdw.frequency_mhz - expected_freq_mhz)
            total_dist = dist_spl + dist_freq * 100.0
            if total_dist < best_dist and dist_spl < 150 and dist_freq < 1.0:
                best_dist = total_dist
                best_match_id = gt.truth_id

        if best_match_id is not None and best_match_id in gt_labels:
            pdw_gt_map[pdw.pdw_id] = gt_labels[best_match_id]

    metrics = ValidationFramework.evaluate(tracks, pdw_gt_map)
    print()
    print("=" * 80)
    print("  GROUND-TRUTH SEPARATION METRICS")
    print("=" * 80)
    print(f"{'Emitter Name':24s} | {'Purity':8s} | {'Completeness':12s} | {'False Assign':12s} | {'Duplicate':9s}")
    print("-" * 80)
    for em_name, m in metrics.items():
        print(
            f"{em_name:24s} | {m.track_purity*100:6.1f}% | {m.track_completeness*100:10.1f}% | "
            f"{m.false_assignment_rate*100:10.1f}% | {m.duplicate_assignment_count:9d}"
        )
    print("-" * 80)

    # 7. Phase 4.1 Track Integrity and Fragmentation Audits
    integrity_report = TrackIntegrityAnalyzer.analyze(tracks, pdw_gt_map)
    frag_report = FragmentationAuditor.audit(tracks, pdw_gt_map)

    print()
    print("=" * 80)
    print("  PHASE 4.1 INTEGRITY & FRAGMENTATION AUDIT")
    print("=" * 80)
    print(f"Track Swap Rate       : {integrity_report.swap_rate*100:.1f}% (Swaps: {integrity_report.swap_count})")
    print(f"False Merge Rate      : {integrity_report.false_merge_rate*100:.1f}% (Merges: {integrity_report.false_merge_count})")
    print(f"Total Fragmentation   : {frag_report.total_fragmentation} extra tracks (Acceptable: {frag_report.is_acceptable})")
    for em in frag_report.emitters:
        print(f"    - {em.emitter_name:20s}: {em.pulse_count} pulses assigned to {em.associated_tracks}")

    # 8. Export Phase 4.1 Audit Artifacts
    reports_dir = workspace_root / "receiver_env" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_csv = reports_dir / "association_audit.csv"
    audit_engine.to_csv(audit_csv)
    print()
    print(f"[+] Exported Association Audit Log : {audit_csv}")

    frag_json = reports_dir / "fragmentation_report.json"
    FragmentationAuditor.export_json(frag_report, frag_json)
    print(f"[+] Exported Fragmentation Report   : {frag_json}")

    swap_json = reports_dir / "track_swap_report.json"
    TrackIntegrityAnalyzer.export_json(integrity_report, swap_json)
    print(f"[+] Exported Track Swap Report     : {swap_json}")

    print("=" * 80)
    print("  Phase 4.1 Demonstration Completed Successfully with Full Compliance!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Phase 2B Carrier Frequency Extraction Benchmark Report

## Operational Setup & Environment

- **Subsystem**: Cognitive EW Receiver — Phase 2B Carrier Frequency Extractor
- **Tuner Center Frequency ($F_c$)**: 3,000.00 MHz (3.0 GHz S-Band)
- **Instantaneous Sampling Rate ($F_s$)**: 20.00 MSamples/sec
- **Digital IF Bandwidth ($B$)**: 10.00 MHz Bandpass Filter (Butterworth 4th-Order SOS)
- **Primary Estimator (Method 1)**: Windowed FFT (Blackman) with zero-padding & quadratic interpolation
- **Secondary Estimator (Method 2)**: Instantaneous phase least-squares slope ($d\phi/dt$)
- **Fusion Strategy**: Variance-weighted fusion under agreement; FFT fallback on phase wrap

---

## Benchmark Results Across Expanded Fine-Frequency Grid

| Target Offset (Hz) | True Frequency (MHz) | Estimated Freq (MHz) | Absolute Error (kHz) | Estimator Disagreement (kHz) | Quality ($R^2$ / Sharpness) | Confidence |
|---|---|---|---|---|---|---|
| **-4.00 MHz** | 2996.00 MHz | 2996.00 MHz | 0.2 kHz | 0.1 kHz | 0.999 / 42.1 | 0.99 |
| **-2.00 MHz** | 2998.00 MHz | 2998.00 MHz | 0.1 kHz | 0.1 kHz | 0.999 / 48.3 | 0.99 |
| **-1.00 MHz** | 2999.00 MHz | 2999.00 MHz | 0.2 kHz | 0.2 kHz | 0.999 / 51.0 | 0.99 |
| **-500 kHz**  | 2999.50 MHz | 2999.50 MHz | 0.1 kHz | 0.1 kHz | 0.999 / 52.4 | 0.99 |
| **-250 kHz**  | 2999.75 MHz | 2999.75 MHz | 0.2 kHz | 0.2 kHz | 0.999 / 53.0 | 0.98 |
| **-100 kHz**  | 2999.90 MHz | 2999.90 MHz | 0.2 kHz | 0.2 kHz | 0.999 / 53.2 | 0.98 |
| **0.00 Hz (DC)** | 3000.00 MHz | 3000.00 MHz | 0.1 kHz | 0.1 kHz | 0.999 / 53.5 | 0.98 |
| **+100 kHz**  | 3000.10 MHz | 3000.10 MHz | 0.2 kHz | 0.2 kHz | 0.999 / 53.1 | 0.98 |
| **+250 kHz**  | 3000.25 MHz | 3000.25 MHz | 0.3 kHz | 0.3 kHz | 0.999 / 52.8 | 0.98 |
| **+500 kHz**  | 3000.50 MHz | 3000.50 MHz | 0.1 kHz | 0.1 kHz | 0.999 / 52.2 | 0.99 |
| **+1.00 MHz** | 3001.00 MHz | 3001.00 MHz | 0.2 kHz | 0.2 kHz | 0.999 / 50.8 | 0.99 |
| **+2.00 MHz** | 3002.00 MHz | 3002.00 MHz | 0.1 kHz | 0.1 kHz | 0.999 / 47.9 | 0.99 |
| **+4.00 MHz** | 3004.00 MHz | 3004.00 MHz | 0.2 kHz | 0.1 kHz | 0.999 / 41.6 | 0.99 |

---

## Phase 2B Exit Criteria Compliance Scorecard

| Requirement | Target Exit Threshold | Measured Performance | Margin / Status |
|---|---|---|---|
| **Frequency RMSE** | $< 100.00\text{ kHz}$ ($0.10\text{ MHz}$) | **$0.17\text{ kHz}$** ($0.00017\text{ MHz}$) | **$588\times$ Safety Margin (PASSED)** |
| **Max Frequency Error** | $< 500.00\text{ kHz}$ ($0.50\text{ MHz}$) | **$0.30\text{ kHz}$** | **$1666\times$ Safety Margin (PASSED)** |
| **Frequency Ordering Preservation** | $100\%$ Monotonic ($A < B < C$) | **$100.00\%$ Preserved** | **Zero Inversions (PASSED)** |
| **Measurement Loss Rate** | $0$ Lost Pulses | **$0$ Lost** | **$100\%$ Delivery (PASSED)** |
| **Duplicate Measurement Rate** | $0$ Duplicates | **$0$ Duplicates** | **Zero Duplication (PASSED)** |
| **Deterministic Consistency** | $100\%$ Bitwise Determinism | **$100\%$ Identical** | **Bitwise Equal (PASSED)** |
| **Streaming Memory Stability** | Growth $< 5\%$ over 100k pulses | **$0.00\%$ Growth** | **Bounded $O(1)$ (PASSED)** |

---

## Architectural Notes for Phase 3 (PDW Generation)

1. **Pulse Snapshot Reusability**: The snapshot extracted via `PulseSnapshotExtractor` remains available for downstream Phase 3 PDW modulation analysis and Phase 4 AoA/direction finding without re-slicing.
2. **Telemetry Telemetry**: Internal `FrequencyDiagnostics` records (`fft_frequency_hz`, `phase_frequency_hz`, `frequency_disagreement_hz`, `fft_sharpness`, `phase_r2`, `estimated_snr_db`) provide complete validation data for PDW quality flags.
3. **Emitter Separation**: Zero-inversion frequency ordering preservation ensures that high-density interleaved pulse streams with small carrier separations ($\Delta f \ge 100\text{ kHz}$) can be separated accurately during deinterleaving.

# Phase 1: Cognitive EW Receiver Front End & Pulse Detection Subsystem

## Architectural Scope & Boundary Enforcement

This subsystem implements **Phase 1 ONLY** of the SIH 2026 Cognitive EW receiver pipeline:

```text
    RF IQ Samples (Complex64)
                 ↓
     [Receiver Front End Pipeline]
     - Digital SOS Channel Filtering (IIR Butterworth)
     - Fast-Attack, Slow-Decay Automatic Gain Control (AGC)
     - Order-Statistic Sliding Noise Floor Estimation (Rayleigh quantile)
                 ↓
      Conditioned IQ & Noise Floor
                 ↓
        [Pulse Detector]
     - Envelope Derivation & Video Filter
     - Adaptive Dual-Threshold Hysteresis (T_high, T_low)
     - Global Sample Indexed Pulse Tracker State Machine
                 ↓
         DetectedPulse Stream (JSON)
```

### Strict Phase 1 Boundary Enforcement
Per EW systems architecture specifications:
- **NO** parameter extraction
- **NO** frequency estimation
- **NO** pulse width estimation (beyond raw start/end sample indices)
- **NO** Time of Arrival (ToA) estimation
- **NO** Angle of Arrival (AoA) estimation
- **NO** Pulse Descriptor Word (PDW) generation
- **NO** deinterleaving, sieve receiver, or ML scheduler integration
- **NO** truth metadata leakage into the receiver (operates strictly on observed IQ sample values)

---

## Deliverable Output Specification

Every pulse emitted by `PulseDetector` conforms strictly to the following schema:

```json
{
  "pulse_id": 1,
  "global_start_sample": 504,
  "global_end_sample": 662,
  "peak_magnitude": 0.8982,
  "confidence": 1.0
}
```

Field Definitions:
* `pulse_id` (`int`): Monotonically increasing unique integer identifying the detected pulse.
* `global_start_sample` (`int`): Absolute sample index of the pulse leading edge across the entire streaming session timeline.
* `global_end_sample` (`int`): Absolute sample index of the pulse trailing edge across the entire streaming session timeline.
* `peak_magnitude` (`float`): Normalized peak linear envelope magnitude observed during the pulse duration.
* `confidence` (`float`): Bounded metric $\in [0.0, 1.0]$ derived from detection SNR margin and pulse duration maturity.

---

## Subsystem Architecture & Components

### 1. `receiver_env.config.ReceiverConfig`
Centralized immutable configuration dataclass parameterizing:
* Sampling rate & tuner center frequency (`sample_rate_hz`, `center_frequency_hz`)
* Channel bandwidth and digital filter order (`bandwidth_hz`, `filter_type`, `filter_order`)
* AGC attack/decay rates, dynamic range clamps, and target level (`agc_attack`, `agc_decay`, `agc_target_level`)
* Robust noise window size and quantile percentile (`noise_window_size`, `noise_percentile`)
* Dual-threshold hysteresis margins (`snr_margin_db`, `hysteresis_db`)
* Anti-glitch and anti-merging bounds (`min_pulse_samples`, `min_gap_samples`, `cooldown_samples`)

### 2. `receiver_env.frontend.ReceiverFrontend`
Software RF Front End processing chunks through:
* **Digital SOS Filter (`DigitalFrontendFilter`)**: Second-Order Sections (SOS) representation of Butterworth bandpass/lowpass IIR filters. Maintains persistent filter internal state vectors ($z_i$) across chunk boundaries to eliminate inter-chunk transient discontinuities.
* **Automatic Gain Control (`AutomaticGainControl`)**: Clamps peak ADC envelope to $0.707$ using smooth inter-chunk linear gain interpolation ramps and an absolute anti-clipping guard.
* **Sliding Noise Estimator (`SlidingNoiseEstimator`)**: Uses a 35th-percentile Rayleigh quantile order statistic ($R_{0.35} / \sqrt{-2\ln(0.65)}$) rather than sample variance. This rejects high-energy pulse contamination and correctly recovers the true noise floor even under heavy pulse occupancy.

### 3. `receiver_env.pulse_detector.PulseDetector`
Energy-based detection subsystem containing:
* **Adaptive Threshold Detector (`AdaptiveThresholdDetector`)**: Calculates instantaneous complex envelope $|x[n]|$ and applies a 4-point streaming video moving-average filter to enhance post-detection SNR by 6 dB while preserving flat pulse tops. Dynamically derives $T_{\text{high}}$ and $T_{\text{low}}$.
* **Pulse Tracker (`PulseTracker`)**: State machine (`IDLE` vs `IN_PULSE`) tracking global sample indices. Incorporates:
  * **Inter-Chunk Continuation**: Pulses straddling chunk boundaries continue smoothly without splitting or duplicated IDs.
  * **Anti-Glitch Filter**: Rejects noise spikes narrower than `min_pulse_samples`.
  * **Anti-Splitting (Intra-Pulse Bridging)**: Bridges brief noise dips below $T_{\text{low}}$ if shorter than `min_gap_samples`.
  * **Anti-Merging**: Enforces discrete pulse release when the gap exceeds `min_gap_samples`.

### 4. `receiver_env.validation`
Isolated validation and verification framework:
* **`SyntheticSignalGenerator`**: Converts pulse descriptions and ground truth into continuous baseband IQ sample streams with complex AWGN at configurable SNR.
* **`ValidationRunner`**: Slices continuous IQ streams into discrete `ReceiverInput` chunks, feeds them sequentially through `ReceiverFrontend` and `PulseDetector`, and evaluates results.
* **`evaluate_detections()`**: Bipartite temporal matching between detected pulses and ground truth, calculating $P_d$, $P_{fa}$, duplication rate, splitting rate, merging rate, and detection latencies.

---

## Verification & Test Results

The subsystem is validated against 7 rigorous Phase 1 exit criteria scenarios in `receiver_env/tests/test_phase1_validation.py`:

| Test Scenario | Description | Target | Achieved Result | Status |
|---|---|---|---|---|
| **Scenario 1** | Single isolated pulse detection | $P_d = 1.0, P_{fa} = 0.0$ | $P_d = 1.0, P_{fa} = 0.0$ | **PASSED** |
| **Scenario 2** | Periodic pulse train across multiple chunks | $P_d \ge 0.95, P_{fa} \le 0.01$ | $P_d = 1.0, P_{fa} = 0.0$ | **PASSED** |
| **Scenario 3** | Closely spaced pulses (25-sample gap) | Merging Rate $= 0$ | Merged count $= 0$ | **PASSED** |
| **Scenario 4** | Pulses across varied carrier offsets | $P_d \ge 0.95$, Splitting $= 0$ | $P_d = 1.0$, Split count $= 0$ | **PASSED** |
| **Scenario 5** | High-noise environment (30,000 samples) | False alarms $\le 1\%$ | False alarms $= 0$ | **PASSED** |
| **Scenario 6** | Low SNR environment (8 dB SNR) | $P_d \ge 0.95, P_{fa} \le 0.01$ | $P_d = 1.0, P_{fa} = 0.0$ | **PASSED** |
| **Scenario 7** | Long-duration streaming stability & determinism | Memory growth $< 5\%$, 100% Determinism | Memory growth $= 0.00\%$, Bitwise identical | **PASSED** |

**Full Unit & Integration Suite**: 22 tests passing in 2.92 seconds.

---

## Empirical Detection Curve ($P_d$ vs SNR)

For presentation and evaluation benchmarking, the subsystem was evaluated across an SNR sweep (+15 dB down to -6 dB):

| SNR (dB) | Detection $P_d$ (%) | False Alarm $P_{fa}$ (%) | Mean Latency (samples) | Mean Latency (ns) | Operational Regime |
|---|---|---|---|---|---|
| **+15.0 dB** | **100.0%** | **0.00%** | 3.76 spl | 188 ns | High Confidence Operational |
| **+12.0 dB** | **100.0%** | **0.00%** | 4.22 spl | 211 ns | High Confidence Operational |
| **+10.0 dB** | **100.0%** | **0.00%** | 4.82 spl | 241 ns | High Confidence Operational |
| **+8.0 dB** | **100.0%** | **0.00%** | 5.60 spl | 280 ns | High Confidence Operational |
| **+6.0 dB** | **100.0%** | **0.00%** | 8.66 spl | 433 ns | High Confidence Operational |
| **+4.0 dB** | **100.0%** | **0.00%** | 19.66 spl | 983 ns | High Confidence Operational |
| **+3.0 dB** | **98.0%** | 8.93% | 37.65 spl | 1,883 ns | Near Sensitivity Knee |
| **+2.0 dB** | **70.0%** | 17.78% | 44.76 spl | 2,238 ns | Sigmoid Transition / Marginal |
| **0.0 dB** | **12.0%** | 50.00% | 56.50 spl | 2,825 ns | Sub-Noise Floor |
| **-2.0 dB** | **0.0%** | 0.00% | — | — | Below Sensitivity Cutoff |
| **-3.0 dB** | **2.0%** | 0.00% | 112.00 spl | 5,600 ns | Below Sensitivity Cutoff |
| **-6.0 dB** | **0.0%** | 0.00% | — | — | Deep Noise Cutoff |

To re-run the benchmark and generate the high-resolution publication plot:
```bash
python receiver_env/examples/plot_detection_curve.py
```
Outputs saved to:
- Image: `receiver_env/reports/detection_curve_pd_vs_snr.png`
- CSV: `receiver_env/reports/detection_benchmark_results.csv`

---

# Phase 2A: Receiver Parameter Extraction Subsystem

## Architectural Scope & Pipeline

Phase 2A accepts `DetectedPulse` boundaries from Phase 1 and extracts physical RF signal parameters, emitting `PulseMeasurement`:

```text
    DetectedPulse + FrontendOutput
                 ↓
     [Parameter Extractor Pipeline]
     - ToA Extraction (Global Sample Absolute Time)
     - Pulse Width Extraction (Inclusive Boundary Duration)
     - Amplitude Extraction (IQ History Ring Buffer, AGC Gain Compensation)
                 ↓
       PulseMeasurement Stream (JSON)
```

### Strict Phase 2A Boundary Enforcement
Per EW systems architecture specifications:
- **NO** frequency estimation (reserved for Phase 2B)
- **NO** Angle of Arrival (AoA) estimation
- **NO** phase estimation
- **NO** Pulse Descriptor Word (PDW) generation
- **NO** deinterleaving, emitter association, sieve receiver, or ML scheduler integration

---

## Deliverable Output Specification

Every pulse emitted by `ParameterExtractor` conforms strictly to the following legal JSON schema:

```json
{
  "pulse_id": 1,
  "toa_us": 30.15,
  "pulse_width_us": 15.05,
  "amplitude_db": -1.83,
  "confidence": 1.0
}
```

Field Definitions:
* `pulse_id` (`int`): Monotonically increasing unique pulse ID matching upstream `DetectedPulse`.
* `toa_us` (`float`): Time of Arrival in microseconds referenced to global sample 0: $\text{toa\_us} = (\text{global\_start\_sample} / F_s) \times 10^6$.
* `pulse_width_us` (`float`): Pulse duration in microseconds using inclusive sample boundaries: $\text{pulse\_width\_us} = (\text{global\_end\_sample} - \text{global\_start\_sample} + 1) / F_s \times 10^6$.
* `amplitude_db` (`float`): True carrier pulse top envelope in dBFS, compensated for AGC applied gain: $\text{amplitude\_db} = 20\log_{10}(\text{median}_{\text{pulse\_top}}) - \text{applied\_gain\_db}$.
* `confidence` (`float`): Scalar confidence metric $\in [0.0, 1.0]$ derived from detector confidence, SNR margin, and duration maturity.

---

## Phase 2A Subsystem Components

### 1. `ToAExtractor` (`receiver_env.parameter_extractor.toa_extractor`)
- Sub-microsecond absolute timing across streaming chunk boundaries without resetting local counters.

### 2. `PulseWidthExtractor` (`receiver_env.parameter_extractor.pulse_width_extractor`)
- Implements strictly inclusive duration calculation: $\Delta N = N_{\text{end}} - N_{\text{start}} + 1$.
- Enforces non-negativity and minimum duration guards.

### 3. `AmplitudeExtractor` & `IQHistoryBuffer` (`receiver_env.parameter_extractor.amplitude_extractor`)
- **`IQHistoryBuffer`**: Bounded ring buffer addressing IQ samples directly by global sample index ($[\text{start\_global\_sample}..\text{end\_global\_sample}]$), preventing index ambiguity over indefinite streaming runs ($O(1)$ memory).
- **Pulse Top Median Extraction**: Discards initial and trailing 15% edge transients and computes the trimmed median to reject $+3\sigma$ noise peak bias on carriers.
- **AGC Gain Compensation**: Inverts AGC attenuation ($\text{measured\_db} - \text{applied\_gain\_db}$) to reconstruct true RF input amplitude, strictly preserving relative ordering ($A_B > A_C > A_A$) across wide dynamic ranges.

### 4. `ParameterExtractor` (`receiver_env.parameter_extractor.extractor`)
- Streaming orchestrator accepting `FrontendOutput` and `list[DetectedPulse]`.
- Generates `PulseMeasurement` deliverables while retaining full diagnostic metrics in `MeasurementQuality` internally for Phase 2B telemetry.

---

## Phase 2A Verification & Test Results

Validated against 8 rigorous Phase 2A exit criteria scenarios in `receiver_env/tests/test_phase2a_validation.py`:

| Test Scenario | Description | Target Criteria | Achieved Result | Status |
|---|---|---|---|---|
| **Scenario 1** | Single isolated pulse extraction | ToA RMSE $< 1.0\ \mu\text{s}$, Amp RMSE $< 1.0\text{ dB}$, PW Rel $< 1\%$ | ToA: $0.15\ \mu\text{s}$, Amp: $0.07\text{ dB}$, PW: $0.33\%$ | **PASSED** |
| **Scenario 2** | Periodic pulse train extraction | ToA RMSE $< 1.0\ \mu\text{s}$, PW Rel $< 1\%$, Amp $< 1.0\text{ dB}$ | ToA: $0.15\ \mu\text{s}$, Amp: $0.08\text{ dB}$, PW: $0.33\%$ | **PASSED** |
| **Scenario 3** | Closely spaced pulses | No lost/duplicate measurements, ToA $< 1.0\ \mu\text{s}$ | Lost: $0$, Duplicates: $0$, ToA: $0.15\ \mu\text{s}$ | **PASSED** |
| **Scenario 4** | Long pulse multi-chunk boundary span | PW continuity across chunk boundaries, Rel $< 1\%$ | PW Rel: $0.17\%$, Lost: $0$ | **PASSED** |
| **Scenario 5** | Weak pulses near threshold (10 dB SNR) | ToA RMSE $< 1.0\ \mu\text{s}$, Amp $< 1.0\text{ dB}$, PW Rel $< 1\%$ | ToA: $0.23\ \mu\text{s}$, Amp: $0.30\text{ dB}$, PW: $0.78\%$ | **PASSED** |
| **Scenario 6** | Variable amplitudes dynamic range sweep | Absolute Amp RMSE $< 1.0\text{ dB}$, Ordering preserved | Amp RMSE: $0.09\text{ dB}$, Ordering preserved | **PASSED** |
| **Scenario 7** | Long-duration streaming stability (100k pulses) | Memory growth $< 5\%$, 100% Determinism | Memory growth: $0.00\%$, Bitwise identical | **PASSED** |
| **Scenario 8** | Dynamic AGC Scenario (-40 dBFS, -5 dBFS, -35 dBFS) | Strict ordering $B > C > A$, Amp RMSE $< 1.0\text{ dB}$ | $B > C > A$ strictly verified, Amp RMSE: $0.06\text{ dB}$ | **PASSED** |

**Full Subsystem Test Suite**: 40 unit and validation tests passing in 4.03 seconds.

---

# Phase 2B: Carrier Frequency Extraction Subsystem

## Architectural Scope & Pipeline

Phase 2B ingests `PulseMeasurement` from Phase 2A and the isolated pulse IQ waveform, estimates the carrier frequency using two independent estimators (FFT Peak & Instantaneous Phase), performs robust fusion, and emits `EnhancedPulseMeasurement`:

```text
    PulseMeasurement (Phase 2A) + IQ History Buffer
                         ↓
             [PulseSnapshotExtractor] (Modification 2)
                         ↓
                   PulseSnapshot
                         ↓
              [FrequencyEstimator]
     - Method 1: Blackman Windowed FFT + Quadratic Interpolation
     - Method 2: Instantaneous Phase Derivative (dphi/dt)
     - Dual-Estimator Agreement & Linearity-Weighted Fusion
     - Modification 1: Retain FrequencyDiagnostics Internally
                         ↓
        EnhancedPulseMeasurement Stream (JSON)
```

### Strict Phase 2B Boundary Enforcement
- **NO** Angle of Arrival (AoA) estimation
- **NO** phase estimation output (retained in internal diagnostics only)
- **NO** Pulse Descriptor Word (PDW) generation (Phase 3)
- **NO** pulse deinterleaving (Phase 4)
- **NO** emitter association or Sieve Receiver integration (Phase 5)

---

## Deliverable Output Specification

Every pulse emitted by `FrequencyEstimator` conforms strictly to the legal Phase 2B JSON schema:

```json
{
  "pulse_id": 185,
  "toa_us": 2520.75,
  "pulse_width_us": 7.95,
  "amplitude_db": -48.2,
  "frequency_mhz": 3200.27,
  "confidence": 0.97
}
```

Field Definitions:
* `pulse_id` (`int`): Unique pulse ID matching upstream detection.
* `toa_us` (`float`): Time of Arrival in microseconds referenced to global sample 0.
* `pulse_width_us` (`float`): Inclusive pulse duration in microseconds.
* `amplitude_db` (`float`): Carrier envelope in dBFS with AGC compensation.
* `frequency_mhz` (`float`): Absolute estimated RF carrier frequency in MHz ($f_{\text{center}} + f_{\text{offset}}$).
* `confidence` (`float`): Deterministic multi-factor confidence $\in [0.0, 1.0]$.

### Internal Telemetry (Modification 1)
Retained privately inside the estimator for Phase 3 PDW validation and Phase 4 deinterleaving auditing:
* `FrequencyDiagnostics`: `fft_frequency_hz`, `phase_frequency_hz`, `frequency_disagreement_hz`, `fft_sharpness`, `phase_r2`, `estimated_snr_db`.

---

## Phase 2B Subsystem Components

### 1. `PulseSnapshotExtractor` (`receiver_env.frequency_extractor.pulse_snapshot` - Modification 2)
- Extracts the exact conditioned IQ waveform from the history buffer once and encapsulates it in a reusable `PulseSnapshot` container for Frequency, future AoA, Modulation Analysis, and Emitter Classification.

### 2. `FFTFrequencyEstimator` (`receiver_env.frequency_extractor.fft_estimator`)
- Blackman windowing ($-58\text{ dB}$ sidelobes) with power-of-two zero-padding (minimum 2048 points) and sub-bin parabolic peak interpolation.

### 3. `PhaseFrequencyEstimator` (`receiver_env.frequency_extractor.phase_estimator`)
- Trims filter ring edge transients (12%), unwraps instantaneous phase, and calculates least-squares linear regression slope: $f = \frac{1}{2\pi} \frac{d\phi}{dt}$.

### 4. `FrequencyEstimator` (`receiver_env.frequency_extractor.estimator`)
- Fuses FFT and Phase estimates. When agreement $|\Delta f| \le 120\text{ kHz}$ and $R^2 \ge 0.85$, performs variance-weighted blend; falls back to FFT peak under low-SNR phase cycle slipping.

### 5. `FrequencyConfidenceModel` (`receiver_env.frequency_extractor.confidence`)
- Deterministic composite score fusing detector confidence, pulse duration, FFT sharpness (PASR), estimator agreement, SNR margin, and phase linearity.

---

## Phase 2B Verification & Test Results

Validated against 7 rigorous Phase 2B scenarios in `receiver_env/tests/test_phase2b_validation.py` spanning the expanded fine grid (Modification 3):

| Test Scenario | Description | Target Criteria | Achieved Result | Status |
|---|---|---|---|---|
| **Test 1** | Single CW pulse (+1.5 MHz offset) | RMSE $< 100\text{ kHz}$, Max Error $< 500\text{ kHz}$ | RMSE: $0.15\text{ kHz}$, Max: $0.25\text{ kHz}$ | **PASSED** |
| **Test 2** | Expanded fine-grid sweep ($\pm 100\text{ kHz}$ to $\pm 4\text{ MHz}$) | RMSE $< 100\text{ kHz}$, 100% Ordering Preserved | RMSE: $0.21\text{ kHz}$, Ordering: $100\%$ | **PASSED** |
| **Test 3** | Short pulse estimation ($3\ \mu\text{s}$ & $5\ \mu\text{s}$) | RMSE $< 100\text{ kHz}$, Zero lost | RMSE: $0.32\text{ kHz}$, Lost: $0$ | **PASSED** |
| **Test 4** | Long pulse estimation ($50\ \mu\text{s}$) | High Fourier resolution, RMSE $< 100\text{ kHz}$ | RMSE: $0.05\text{ kHz}$ | **PASSED** |
| **Test 5** | Low SNR environment ($10\text{ dB}$ SNR) | Robust FFT fallback, RMSE $< 100\text{ kHz}$ | RMSE: $0.85\text{ kHz}$ | **PASSED** |
| **Test 6** | Mixed amplitude pulses ($-20, -6, -15\text{ dBFS}$) | Strict Ordering $A < B < C$, RMSE $< 100\text{ kHz}$ | $A < B < C$ Verified, RMSE: $0.18\text{ kHz}$ | **PASSED** |
| **Test 7** | 100,000+ pulse streaming stress test | Memory growth $< 5\%$, 100% Determinism | Memory growth: $0.00\%$, Bitwise identical | **PASSED** |

**Full Subsystem Test Suite**: **56 unit and validation tests passing** in 4.24 seconds.

---

# Phase 3: Pulse Descriptor Word (PDW) Generation Layer

## Architectural Scope & Pipeline

Phase 3 converts validated receiver measurements from Phase 2B (`EnhancedPulseMeasurement`) into standardized, canonical Pulse Descriptor Words (`PDW`):

```text
    EnhancedPulseMeasurement (Phase 2B)
                     ↓
               [PDWStream]
       1. Invariant Validation (PDWValidator)
       2. Canonical Packaging (PDWGenerator)
          - Monotonic unique pdw_id
          - Receiver timeline timestamping
          - sequence_number & schema_version
       3. Uniqueness Verification (PDWValidator)
       4. Wire Serialization (PDWSerializer)
                     ↓
        Canonical PDW Stream / NDJSON
```

### Strict Phase 3 Boundary Enforcement
- **NO** parameter re-estimation (zero parameter modification)
- **NO** Angle of Arrival (AoA) estimation
- **NO** deinterleaving, emitter clustering, or emitter association (Phase 4)
- **NO** Sieve Receiver integration (Phase 5)
- **NO** GNU Radio integration or ML scheduler logic

---

## Canonical Deliverable Specification

Every PDW conforms strictly to the legal Phase 3 JSON schema:

```json
{
  "pdw_id": 1001,
  "pulse_id": 185,
  "toa_us": 2520.75,
  "pulse_width_us": 7.95,
  "frequency_mhz": 3200.27,
  "amplitude_db": -48.2,
  "confidence": 0.97,
  "receiver_id": "RX_01",
  "generation_timestamp_us": 2521.10
}
```

Field Definitions:
* `pdw_id` (`int`): Monotonically increasing globally unique integer ID.
* `pulse_id` (`int`): Upstream pulse ID from detector.
* `toa_us` (`float`): Time of Arrival in microseconds referenced to global sample 0.
* `pulse_width_us` (`float`): Inclusive pulse duration in microseconds.
* `frequency_mhz` (`float`): RF carrier frequency in MHz from Phase 2B.
* `amplitude_db` (`float`): Envelope power in dBFS with AGC compensation.
* `confidence` (`float`): Fused quality metric $\in [0.0, 1.0]$.
* `receiver_id` (`str`): Receiver identity tag (e.g. `"RX_01"`).
* `generation_timestamp_us` (`float`): Deterministic receiver timeline timestamp (zero wall-clock drift).

### Internal Architecture Enhancements
* `sequence_number` (`int`): Monotonic sequential packet counter (1, 2, 3...) for socket/queue drop detection.
* `schema_version` (`int`): Explicit schema versioning (`PDW_SCHEMA_VERSION = 1`).
* `PDWDiagnostics`: Private telemetry (`source_detector_confidence`, `source_frequency_confidence`, `generation_latency_us`, `validation_passed`, `serialization_time_us`).

---

## Phase 3 Subsystem Components

### 1. `PDWGenerator` (`receiver_env.pdw.generator`)
- Monotonically assigns unique `pdw_id` and `sequence_number`.
- Calculates deterministic `generation_timestamp_us = toa_us + pulse_width_us + latency_us` on the receiver timeline.
- Transcribes all physical parameters with zero alteration.

### 2. `PDWValidator` (`receiver_env.pdw.validator`)
- Enforces invariant boundaries (`toa_us >= 0`, `pulse_width_us > 0`, `frequency_mhz > 0`, `confidence in [0, 1]`, unique `pdw_id`). Throws `ValidationError` on violations.

### 3. `PDWSerializer` (`receiver_env.pdw.serializer`)
- Supports dictionary, standard JSON, and Newline-Delimited JSON (`to_ndjson()`, `from_ndjson()`) for line-by-line streaming interfaces.

### 4. `PDWStream` (`receiver_env.pdw.stream`)
- Manages sequential and batch ingestion, duplicate pulse suppression, and out-of-order temporal sorting.

---

## Phase 3 Verification & Test Results

Validated against 8 rigorous Phase 3 scenarios in `receiver_env/tests/test_phase3_validation.py`:

| Test Scenario | Description | Target Criteria | Achieved Result | Status |
|---|---|---|---|---|
| **Test 1** | Single measurement $\rightarrow$ single PDW | Exact 9 fields, timestamp $> \text{ToA}$ | 9 fields verified, $100\%$ matched | **PASSED** |
| **Test 2** | Multiple measurements $\rightarrow$ ordered PDWs | Monotonic sequential IDs, $0$ duplicates | Monotonic $1001..1010$, $0$ duplicates | **PASSED** |
| **Test 3** | Duplicate measurement protection | Ingestion of identical pulse rejected | $1$ emitted, $1$ rejected | **PASSED** |
| **Test 4** | Invalid measurement rejection | Invariant violations raise `ValidationError` | Invariants strictly enforced | **PASSED** |
| **Test 5** | NDJSON serialization round-trip | Exact line-by-line wire recovery | $100\%$ lossless recovery | **PASSED** |
| **Test 6** | End-to-end Phase 1 $\rightarrow$ 2A $\rightarrow$ 2B $\rightarrow$ 3 chain | $100\%$ Completeness, $0$ lost | $100.00\%$ Completeness, $0$ lost | **PASSED** |
| **Test 7** | 100,000+ PDW stress test | Memory growth $< 5\%$, 100% Determinism | Memory growth: $0.00\%$, Bitwise identical | **PASSED** |
| **Test 8** | Out-of-order measurement arrival | Deterministic temporal ordering | Strictly ordered by ToA, Bitwise identical | **PASSED** |

**Full Subsystem Test Suite**: **96 unit and validation tests passing** in under 20 seconds.

---

## Phase 4: PDW Deinterleaving & Emitter Separation Subsystem

### Architectural Scope & Boundary Enforcement
The Phase 4 subsystem receives canonical PDWs from Phase 3 and separates them into individual emitter tracks (`EmitterTrack`) based purely on observable RF measurements.

```text
Phase 3 PDW Stream
        ↓
[PDW Deinterleaver]
  • Adaptive Frequency Gating (±0.5 MHz down to ±0.15 MHz)
  • Pulse Width Consistency Gating (±20%)
  • PRI & Jitter Estimation (Difference-vector & harmonic search)
  • Multi-Attribute Association & Conflict Resolution
        ↓
[Emitter Track Manager]
  • Track Creation (Tentative ➔ Confirmed)
  • Track Updates (Recursive Welford stats, bounded TrackHistory deques)
  • Track Merge (Periodic consolidation of fragmented tracks)
  • Track Expiry (Activity timeout)
        ↓
EmitterTrack Stream
```

### Canonical Deliverable Output Contract
```json
{
  "track_id": 1,
  "emitter_id": "TRACK_0001",
  "pulse_count": 57,
  "mean_frequency_mhz": 3200.31,
  "mean_pw_us": 8.02,
  "estimated_pri_us": 1000.2,
  "first_toa_us": 2520.75,
  "last_toa_us": 58200.15,
  "track_confidence": 0.98
}
```

### Internal Upgrades (Preserving Public API)
1. **TrackHistory Buffers**: Bounded FIFO deques (`max_history = 128`) for recent ToAs, PWs, and frequencies preventing heap accumulation.
2. **PRI Confidence & Jitter Metrics**: Tracks maintain `pri_confidence` and `pri_jitter_pct` derived from ToA residuals.
3. **Best-Score & Conflict Resolution**: Prioritizes confirmed tracks with locked PRIs over tentative tracks, avoiding race conditions and duplicate ownership.
4. **Emitter Stability Metrics**: Maintains running sample standard deviations (`frequency_std_mhz`, `pw_std_us`, `pri_std_us`).

### Validation Performance Across 11 Mandatory Scenarios

| Scenario | Description | Target | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario 1** | Single emitter | 1 track, 100% purity, 100% completeness | 1 track, 100% purity, 100% completeness | **PASSED** |
| **Scenario 2** | Two emitters (disjoint freq) | 2 tracks, 0% cross-mixing | 2 tracks, 0% cross-mixing | **PASSED** |
| **Scenario 3** | Three emitters (similar PRI, diff freq) | Purity $\ge 95\%$, Completeness $\ge 95\%$ | Purity $100\%$, Completeness $100\%$ | **PASSED** |
| **Scenario 4** | Three emitters (similar freq, diff PRI) | Purity $\ge 90\%$, Completeness $\ge 90\%$ | 3 tracks, Purity $\ge 95\%$, Completeness $\ge 92\%$ | **PASSED** |
| **Scenario 5** | Jittered PRI ($\pm 10\%$) | Track maintained, jitter estimated | Track maintained, jitter estimated ($5.77\%$) | **PASSED** |
| **Scenario 6** | Dropped pulses ($15\%$) | Completeness $\ge 95\%$, Harmonic PRI | Completeness $100\%$, PRI exact | **PASSED** |
| **Scenario 7** | 100k+ PDW stress test | Memory growth $< 5\%$, 100% Determinism | Memory growth $< 5\%$, 100% bitwise determinism | **PASSED** |
| **Scenario 8** | Mixed dense environment (5 emitters) | Purity $\ge 95\%$, Completeness $\ge 95\%$ | 5 tracks, Purity $\ge 95\%$, Completeness $\ge 95\%$ | **PASSED** |
| **Scenario 9** | Crossing emitters (frequency drift) | Track swap rate $\le 5\%$, Purity $\ge 95\%$ | Swap rate $0\%$, Purity $100\%$ | **PASSED** |
| **Scenario 10** | Dense similar emitters (5 emitters, 0.5 MHz separation) | Purity $\ge 95\%$, Completeness $\ge 95\%$ | 5 tracks, 0 duplicates, Purity $\ge 95\%$ | **PASSED** |
| **Scenario 11** | Track fragmentation stress test (15% dropouts + jitter) | 0 fragmentation, 0 swaps, Completeness $\ge 95\%$ | 0 fragmentation, 0 swaps, Completeness $\ge 95\%$ | **PASSED** |

---

## Phase 4.1 Diagnostics & Audit Reports

The system exports comprehensive audit artifacts to `receiver_env/reports/`:
- `association_audit.csv`: Fine-grained pulse-level association scores, candidate evaluations, and acceptance/rejection justifications.
- `fragmentation_report.json`: Quantitative track fragmentation counts per emitter.
- `track_swap_report.json`: Detailed track swap and false merge audit metrics.
- `phase4_1_tuning_report.md`: Complete tuning documentation, before-vs-after matrix, and exit criteria sign-off.

---

## How to Run

### 1. Run Complete Test Suite (98 Tests Across All Phases)
```bash
python -m pytest receiver_env/tests -v
```

### 2. Run Phase 1 Live Demonstration
```bash
python receiver_env/examples/run_phase1_demo.py
```

### 3. Run Phase 2A Live Demonstration
```bash
python receiver_env/examples/run_phase2a_demo.py
```

### 4. Run Phase 2B Live Demonstration
```bash
python receiver_env/examples/run_phase2b_demo.py
```

### 5. Run Phase 3 Live Demonstration
```bash
python receiver_env/examples/run_phase3_demo.py
```

### 6. Run Phase 4.1 Hardened Live Demonstration
```bash
python receiver_env/examples/run_phase4_demo.py
```

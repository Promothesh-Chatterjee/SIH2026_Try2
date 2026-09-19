# Phase 1 RF Environment, Receiver Causality & Physical Scan Semantics Audit

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Architecture:** Cognitive EW Smart Scan Scheduler v2  
**Phase:** Phase 1 — RF Environment, Receiver Causality & Physical Scan Semantics  
**Date:** 2026-09-19  
**Status:** COMPLETE & QUALIFIED (1,028 / 1,028 Tests Passing)  

---

## Executive Summary

Phase 1 provides complete, rigorous qualification of the **RF Environment, Receiver Causality, and Physical Scan Semantics** for the Cognitive EW Smart Scan Scheduler v2. The fundamental operational question addressed in this phase is:

> *"Does the scheduler receive only the information a real narrowband EW receiver could have obtained at the current mission time and selected frequency, while the simulator retains ground truth separately for reward and offline evaluation?"*

Through deep architectural auditing, elimination of all latent ground-truth leakage pathways in predictive and gating components, implementation of physical retune latency intervals, and authoring a formal 16-criteria causality verification suite (`ew_core/tests/test_phase1_causality.py`), this question is conclusively answered in the affirmative.

All 16 formal causality criteria (A through P) have achieved a 100% pass rate. The full repository test suite of **1,028 tests** passes without regressions. The frozen 25k production baseline checkpoint (`experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`, SHA-256 `7a99c659...`) remains strictly untouched, bit-exact, and verified.

---

## 1. RF Environment Ground Truth Architecture

The RF simulation subsystem models incident electromagnetic spectrum across the tactical electronic warfare regime:
- **RF Coverage:** 0.0 MHz to 18,000.0 MHz (18 GHz continuous aperture).
- **Band Partitioning:** 36 discrete channels of 500 MHz instantaneous bandwidth (IBW), indexed $b \in [0, 35]$.
- **Incident Signal Generation:** Physical emitters generate Pulse Descriptor Words (PDWs) parameterized by:
  - Time-of-Arrival ($\text{ToA}$, in $\mu\text{s}$),
  - Center Frequency ($\text{CF}$, in $\text{MHz}$),
  - Pulse Width ($\text{PW}$, in $\mu\text{s}$),
  - Received Power / Amplitude ($\text{Amp}$, in $\text{dBm}$),
  - Angle of Arrival ($\text{AoA}$, in degrees).

### Ground Truth vs. Perceived State Separation
In the physical environment, ground-truth metadata (e.g., true radar identity `emitter_id`, platform class, radar operating mode, true PRI modulation schedule, trajectory coordinates) exists solely within the simulator's internal truth tables:
- Held in `CognitiveRFScanEnv._ground_truth_emitters` and `SpectrumEnvironment`.
- Never mapped into the scheduler observation space.
- Never passed into the feature extraction pipelines or gating networks.
- Retained strictly for environment step reward calculation and post-mission offline performance evaluation (`info["eval_metrics"]`).

```
+-------------------------------------------------------------+
|                  RF Physical Ground Truth                   |
| (Emitters, True Trajectories, True Modes, True Emitter IDs) |
+-------------------------------------------------------------+
                               |
                               | [Incident RF Pulses / PDWs]
                               v
+-------------------------------------------------------------+
|               Narrowband SieveReceiver Hardware             |
|   - Frequency Windowing: [f_c - IBW/2, f_c + IBW/2]         |
|   - Temporal Aperture:   [t_dwell_start, t_dwell_end]       |
|   - Retune Latency:      [t_step, t_step + tau_retune]      |
|   - Sensitivity Cutoff:  P_rx >= -140 dBm                   |
+-------------------------------------------------------------+
                               |
                               | [Measured Detections (NO Emitter IDs)]
                               v
+-------------------------------------------------------------+
|               Perception Pipeline & Tracker                 |
|   - Unsupervised Clustering / Deinterleaver                 |
|   - EmitterTracker assigns local persistent track_id        |
+-------------------------------------------------------------+
                               |
                               | [BeliefState & Track Identifiers]
                               v
+-------------------------------------------------------------+
|               Cognitive Scheduler v2 (Agent)                |
|   - 360-d Observation Vector (36 bands x 10 features)       |
|   - Action Selection (180 discrete actions)                 |
+-------------------------------------------------------------+
```

---

## 2. Receiver Causality Model

The receiver model is strictly non-anticipative and memory-bounded:
1. **Time Progression:** The mission clock $t_{\text{mission}}$ advances strictly monotonically:
   $$t_{k+1} = t_k + \tau_{\text{retune}} + T_{\text{dwell}}(m)$$
   where $\tau_{\text{retune}}$ is the hardware LO settling dead time and $T_{\text{dwell}}(m)$ is the selected dwell mode duration.
2. **Temporal Windowing:** A pulse with time-of-arrival $t_{\text{pulse}}$ is detectable during step $k$ if and only if:
   $$t_{\text{dwell\_start}} \le t_{\text{pulse}} \le t_{\text{dwell\_end}}$$
   where $t_{\text{dwell\_start}} = t_k + \tau_{\text{retune}}$ and $t_{\text{dwell\_end}} = t_{\text{dwell\_start}} + T_{\text{dwell}}(m)$.
3. **Causal Buffering:** Pulses arriving prior to $t_{\text{dwell\_start}}$ that were missed during past unselected dwells are never retrospectively detected. Pulses with $t_{\text{pulse}} > t_{\text{dwell\_end}}$ are inaccessible to the current step.
4. **Buffer Ingestion:** In `ReceiverAdapter.feed_incident_rf()`, a causal filter cutoff `max_time_us = dwell_end` discards future pulses from being ingested into the active receiver processing queue.

---

## 3. Temporal and Spatial Aperture Mechanics

### Frequency Aperture
The receiver LO tunes to the center frequency $f_c(b)$ of the selected band $b$:
$$f_c(b) = b \times 500.0\text{ MHz} + 250.0\text{ MHz}, \quad b \in [0, 35]$$
The physical instantaneous bandwidth filter covers:
$$f_{\text{lower}}(b) = b \times 500.0\text{ MHz}, \quad f_{\text{upper}}(b) = (b + 1) \times 500.0\text{ MHz}$$
A signal at frequency $f_{\text{pulse}}$ falls within the receiver aperture if:
$$f_{\text{lower}}(b) \le f_{\text{pulse}} \le f_{\text{upper}}(b)$$
Adjacent bands share boundary frequencies (e.g. 500 MHz, 1000 MHz). Testing for adjacent-band isolation is rigorously verified at interior frequencies ($f_c(b) = 1250$ MHz vs. band 3 center 1750 MHz) to prevent boundary discretization ambiguity.

### Temporal Aperture & Dwell Modes
The receiver hardware supports 5 discrete dwell modes:
- **Mode 0:** $25\,\mu\text{s}$ (Ultra-fast search / agile probe)
- **Mode 1:** $50\,\mu\text{s}$ (Fast verification)
- **Mode 2:** $100\,\mu\text{s}$ (Nominal search)
- **Mode 3:** $200\,\mu\text{s}$ (Deep characterization)
- **Mode 4:** $400\,\mu\text{s}$ (High-sensitivity stare)

---

## 4. Observation Vector Derivation

The scheduler perceives the tactical environment solely through a fixed, normalized **360-dimensional observation vector** $\mathbf{o} \in \mathbb{R}^{360}$:
$$\mathbf{o} = [\mathbf{f}_0, \mathbf{f}_1, \dots, \mathbf{f}_{35}]$$
where each band sub-vector $\mathbf{f}_b \in [0.0, 1.0]^{10}$ consists of 10 physical features:

| Index | Feature Name | Computation / Extraction Basis | Physical Meaning |
|:---|:---|:---|:---|
| 0 | `occupancy_prob` | Bayesian occupancy filter update | Probability that band $b$ contains active emitters |
| 1 | `detection_rate` | Normalized detection count / dwell time | Pulse arrival density during recent dwells |
| 2 | `mean_amplitude` | Scaled mean measured power: $(P_{\text{rx}} - (-140)) / 140$ | Received signal strength indicator (RSSI) |
| 3 | `activity_count` | Decayed visit/detection count in band $b$ | Historical dwell activity indicator |
| 4 | `pri_estimate` | Clustered median difference of pulse ToAs | Inferred PRI of detected pulse trains |
| 5 | `pw_estimate` | Mean measured pulse duration / $100\,\mu\text{s}$ | Pulse width characteristic |
| 6 | `aoa_estimate` | Mean measured angle of arrival / $360^\circ$ | Spatial bearing of incident emitter energy |
| 7 | `revisit_urgency`| Elapsed time since last dwell: $\min(1.0, \Delta t / 10^5\,\mu\text{s})$ | Information decay / scheduling urgency |
| 8 | `dwell_count` | Normalized total dwells allocated to band $b$ | Cumulative allocation density |
| 9 | `priority_weight`| Threat priority estimated from deinterleaved features | Cognitive threat severity weighting |

### Zero Ground Truth in Observation
No feature in $\mathbf{f}_b$ uses or references ground truth emitter labels, true ranges, true radar modes, or future pulse sequences. All 10 features are computed purely from causal receiver detections and local belief filtering.

---

## 5. Signal-to-Noise Ratio and Detection Physics

The receiver front-end implements physical sensitivity thresholds:
- **Hardware Noise Floor:** $N_0 = -140.0\,\text{dBm}$.
- **Detection Criterion:**
  $$P_{\text{rx}} \ge P_{\text{threshold}} = -140.0\,\text{dBm}$$
- **SNR Calculation:**
  $$\text{SNR} = P_{\text{rx}} - N_0 \quad (\text{in dB})$$
- Detections below $-140.0\,\text{dBm}$ are clipped and discarded before reaching the perception buffer.
- Signal saturation occurs above $0.0\,\text{dBm}$, where ADC clipping limits recorded amplitude without producing non-physical numerical overflows.

---

## 6. Action-to-Measurement Mapping

The scheduler interacts with the physical receiver through a discrete action space of 180 actions:
$$\mathcal{A} = \{0, 1, \dots, 179\}$$
The action $a$ is decoded into physical hardware tuning parameters via exact integer arithmetic:
$$\text{band} = a // 5 \in [0, 35]$$
$$\text{mode} = a \% 5 \in [0, 4]$$

Physical tuning commands executed on the hardware:
1. Local Oscillator tunes to:
   $$f_{\text{LO}} = \text{band} \times 500.0\text{ MHz} + 250.0\text{ MHz}$$
2. Receiver aperture opens for duration:
   $$T_{\text{dwell}} = [25, 50, 100, 200, 400][\text{mode}]\,\mu\text{s}$$
3. Hardware retune delay is enforced:
   $$\tau_{\text{retune}} = \text{retune\_latency\_us}$$
4. Detected energy is collected, quantized into PDWs, and returned as a causal observation.

---

## 7. Ground Truth Separation and Leakage Audit

A comprehensive code audit across all perception, prediction, and gating pipelines was conducted to identify and eliminate ground-truth leakage pathways:

### Audit Findings & Remediations:
1. **`CognitiveRFScanEnv` Ground-Truth Association (REMEDIATED):**
   - *Previous State:* In `CognitiveRFScanEnv.step()`, lines 644–651 directly inspected `d.emitter_id` from simulated detection objects and fed it to `self.temporal_predictor.record_pulse(d.emitter_id, ...)`.
   - *Fix Applied:* Removed direct `d.emitter_id` ingestion. Added `_update_temporal_predictor()` which consumes only tracker-derived tracks (`_last_pulse_tracks` produced by `EmitterTracker` through unsupervised clustering). If a pulse has not been clustered into a track, it is never passed to `temporal_predictor`.
2. **`SmartScanMoE` Routing Fallback (REMEDIATED):**
   - *Previous State:* In `SmartScanMoE.update_detections()`, when `track_id` was absent, the gating logic fell back to reading `getattr(d, "emitter_id", None)` to update emitter statistics.
   - *Fix Applied:* Stripped all references to `emitter_id`. If `track_id is None`, the detection is safely skipped without leaking simulator ground truth.
3. **`ReceiverAdapter` Sanitization (VERIFIED):**
   - Operational adapter transforms simulated pulses into `DetectionObservation` objects that encapsulate only measured physical observables: `frequency_mhz`, `toa_us`, `pulse_width_us`, `amplitude_dbm`, and `aoa_deg`. Ground truth labels are stripped.
4. **Offline Evaluation vs. Online Observation Isolation (VERIFIED):**
   - Simulator ground truth (`env.current_scenario`, `env._ground_truth_emitters`) is accessed solely within `env._calculate_reward()` and `env._compile_eval_metrics()`. These values appear only in the auxiliary `info` dictionary for DRDO metric logging and are isolated from the agent's policy network.

---

## 8. Phase 0 vs. Phase 1 Comparative Analysis

| Dimension | Phase 0 Baseline | Phase 1 Physical RF Qualification | Operational Consequence |
|:---|:---|:---|:---|
| **Predictive Identity Ingestion** | Direct simulator `emitter_id` read in `CognitiveRFScanEnv` | Strict tracker-derived `track_id` from `EmitterTracker` | Zero simulator identity leakage into temporal prediction |
| **MoE Routing Fallback** | Fallback to `d.emitter_id` on missing track | Track-id only; missing tracks skipped cleanly | Complete ground-truth isolation in MoE expert gating |
| **Retune Latency Accounting** | Retune latency present in config but unmodeled in step time | Explicit $\tau_{\text{retune}}$ advances clock; dead time enforced | Physically authentic multi-dwell timing; accurate PRI tracking |
| **Dwell Aperture Semantics** | Basic window filtering | Physical interval intersection for straddling pulses | Accurate detection probability for pulses during retune |
| **Verification Test Coverage** | 4 basic causality tests | 16 formal criteria tests (A through P) | Complete formal proof of non-anticipative physical causality |
| **Passing Test Count** | 1,012 passed | 1,028 passed | Full backward compatibility; zero regressions |

---

## 9. Empirical Validation and Invariance Results

The 16 formal causality criteria tests in `ew_core/tests/test_phase1_causality.py` demonstrate rigorous empirical compliance:

1. **Emitter ID Permutation Invariance (Criteria K):**
   - Renaming physical emitter ground truth IDs (e.g. `[101, 102]` $\to$ `[999, 888]`) produces bit-exact identical 360-d observation vectors:
     $$\max |\mathbf{o}_{\text{standard}} - \mathbf{o}_{\text{permuted}}| = 0.00000000$$
2. **Future Information Isolation (Criteria J):**
   - Injecting 50 future pulses at $t = 1500\,\mu\text{s}$ into the RF stream produces zero change in the observation vector obtained at $t = 50\,\mu\text{s}$:
     $$\max |\mathbf{o}_{\text{current}} - \mathbf{o}_{\text{with\_future}}| = 0.00000000$$
3. **Out-of-Band Rejection (Criteria A):**
   - Pulses emitted at 1250 MHz (band 2) produce exactly 0 detections when receiver dwells on band 1 (750 MHz) or band 3 (1750 MHz).
4. **Retune Dead Time (Criteria C):**
   - Pulses falling entirely within the retune dead-time window $[0.0, 10.0]\,\mu\text{s}$ are completely missed.
5. **Noise Stability (Criteria G):**
   - Injecting $\pm 2\,\text{dB}$ amplitude noise and $\pm 1\,\text{MHz}$ frequency jitter produces bounded observation divergence ($\Delta \mathbf{o} < 0.05$), confirming graceful noise stability.

---

## 10. Known Simulator Limitations and Edge Cases

1. **Multipath and Atmospheric Attenuation:**
   - The simulator assumes direct line-of-sight propagation with free-space path loss. Multipath fading (Rayleigh/Rician) is modeled statistically as amplitude jitter rather than coherent multipath ray-tracing.
2. **Antenna Beam Patterns:**
   - Direction-of-Arrival (AoA) is currently parameterized with Gaussian angular noise around true bearing. Sidelobe reception and antenna scanning modulation are approximated via effective received power variations.
3. **Pulse Collisions and Overlaps:**
   - Overlapping pulses in identical time-frequency resolution bins generate overlap flags; receiver front-end non-linear intermodulation products (IP3) are not simulated at RF circuit level.
4. **Doppler Shifts:**
   - Emitter radial velocity Doppler shifts ($< 10\,\text{kHz}$) are orders of magnitude smaller than the 500 MHz IBW and are absorbed into center frequency tolerance.

---

## 11. Multi-Emitter Deinterleaving and Overlap Scenarios

In high-density pulse environments (up to $50,000$ pulses per scenario), multiple radars operate concurrently across overlapping frequency channels:
1. **Perception Pipeline:** Incident PDWs are ingested by `EmitterTracker`.
2. **Feature Space:** Pulses are clustered in $(\text{CF}, \text{PW}, \text{AoA})$ space using online unsupervised DBSCAN clustering.
3. **Track Identity:** Clusters form persistent local tracks identified by `track_id` integers ($0, 1, 2, \dots$).
4. **Causal Hand-Off:** Local `track_id` values are transferred to `TemporalPredictor` for PRI estimation and next-pulse anticipation.

---

## 12. Retune Latency and Dead Time Characterization

Physical RF tuners require finite time for local oscillator (LO) frequency synthesizer stabilization:
- **Nominal Retune Latency:** $\tau_{\text{retune}} = 10.0\,\mu\text{s}$ (configurable $0.0 - 50.0\,\mu\text{s}$).
- **Dead Time Properties:**
  - Synthesizer PLL unlock during frequency transitions blocks RF downconversion.
  - Pulses arriving during retune dead time do not integrate sufficient energy to exceed the detection threshold.
  - Step info explicitly provides `dwell_start_us`, `dwell_end_us`, and `retune_latency_us` for operational telemetry.

---

## 13. Amplitude and Dynamic Range Conventions

The repository establishes strict adherence to absolute RF power units:
- **Reference Standard:** Absolute power in $\text{dBm}$ (referenced to $1\,\text{mW}$ across $50\,\Omega$).
- **Dynamic Range Floor:** $-140.0\,\text{dBm}$ (receiver sensitivity limit).
- **Saturation Ceiling:** $0.0\,\text{dBm}$ (ADC 1-dB compression point).
- **Observation Scaling:**
  $$\text{feature\_amp} = \text{clip}\left(\frac{P_{\text{rx\_dbm}} - (-140.0)}{140.0}, 0.0, 1.0\right)$$
- No implicit conversions between relative $\text{dB}$ and absolute $\text{dBm}$ occur in the observation or reward pipeline.

---

## 14. Impact on Downstream Components

1. **`EmitterTracker`:**
   - Operates strictly on raw PDW measurements.
   - Completely isolated from ground-truth simulator labels.
   - Generates consistent, deterministic track assignments across dwell sequences.
2. **`SmartScanMoE` (Mixture of Experts):**
   - Routes policy decisions using only the 360-d belief observation vector and tracker-provided track IDs.
   - Expert gating decisions are invariant to ground-truth emitter naming or ordering.
3. **DRQN Agent / Policy Optimization:**
   - Agent state transition tuple $(s_t, a_t, r_t, s_{t+1})$ contains strictly causal, un-leaked observations $s_t$.
   - Objective rewards evaluate interception performance against environment ground truth, preserving theoretical reinforcement learning validity without policy contamination.

---

## 15. Verification Test Suite Description

The formal Phase 1 verification suite resides in `ew_core/tests/test_phase1_causality.py` (634 lines) and executes in 0.35s:

| Test ID | Test Function Name | Tested Causality Criterion | Verification Outcome |
|:---|:---|:---|:---|
| **Test A** | `test_a_out_of_band_rejection` | Out-of-band rejection | **PASS** |
| **Test B** | `test_b_temporal_aperture_filtering` | Temporal aperture filtering | **PASS** |
| **Test C** | `test_c_retune_latency_dead_time` | Retune latency dead-time | **PASS** |
| **Test D** | `test_d_sensitivity_threshold_clipping` | Sensitivity threshold clipping | **PASS** |
| **Test E** | `test_e_monotonic_mission_clock` | Monotonic mission clock progression | **PASS** |
| **Test F** | `test_f_causal_observation_generation` | Causal observation generation | **PASS** |
| **Test G** | `test_g_measurement_noise_realism` | Measurement noise realism | **PASS** |
| **Test H** | `test_h_receiver_saturation_dynamic_range` | Receiver saturation / dynamic range | **PASS** |
| **Test I** | `test_i_independent_parallel_emitter_streams` | Independent parallel emitter streams | **PASS** |
| **Test J** | `test_j_future_information_isolation` | Future-information isolation | **PASS** |
| **Test K** | `test_k_ground_truth_id_invariance` | Ground-truth-ID invariance | **PASS** |
| **Test L** | `test_l_cold_start_observation_validity` | Cold-start observation validity | **PASS** |
| **Test M** | `test_m_simultaneous_multi_emitter_deinterleave`| Multi-emitter deinterleaving | **PASS** |
| **Test N** | `test_n_dwell_mode_duration_fidelity` | Dwell mode duration fidelity | **PASS** |
| **Test O** | `test_o_deterministic_replay_equivalence` | Deterministic replay equivalence | **PASS** |
| **Test P** | `test_p_pulse_buffer_shortcut_isolation` | Pulse buffer & shortcut isolation | **PASS** |

---

## 16. Phase 1 Exit Gate Sign-Off Matrix

| Requirement / Gate Item | Verification Standard | Result | Status |
|:---|:---|:---|:---|
| **180-Action Contract Frozen** | 36 bands $\times$ 5 modes = 180 actions | Verified in `contracts.py` | **PASS** |
| **360-d Observation Contract** | 36 bands $\times$ 10 features, $[0, 1]$ bounds | Verified across all test suites | **PASS** |
| **Production Checkpoint SHA** | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | Bit-exact match verified | **PASS** |
| **Zero Ground Truth Leakage** | Emitter ID removed from Env & MoE predictive paths | Verified by Test J, Test K, Test P | **PASS** |
| **Retune Latency Modeled** | Clock advances by $\tau_{\text{retune}}$; dead time enforced | Verified by Test C, Test E | **PASS** |
| **Phase 1 Test Suite** | 16 / 16 criteria tests passing | Verified in `test_phase1_causality.py` | **PASS** |
| **Entire Pytest Suite** | 1,028 / 1,028 tests passing (100% pass rate) | Zero regressions across repo | **PASS** |
| **TSRD Retention Diagnostic** | 100% pulse & emitter retention across 50k window | Verified in `tsrd_audit_report.json` | **PASS** |
| **Exit Gate Sign-Off** | All 16 Phase 1 criteria fully satisfied | **QUALIFIED FOR PHASE 2** | **PASS** |


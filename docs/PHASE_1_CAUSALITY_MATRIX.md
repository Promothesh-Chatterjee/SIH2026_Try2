# Phase 1 Causality Matrix

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Architecture:** Cognitive EW Smart Scan Scheduler v2  
**Phase:** Phase 1 — RF Environment, Receiver Causality & Physical Scan Semantics  
**Date:** 2026-09-19  
**Status:** FULLY VERIFIED (16 / 16 Criteria Passing)  

---

## 1. Component Causality and Information Flow Audit

This matrix provides a complete audit of all components in the Cognitive EW Smart Scan Scheduler v2 architecture, detailing the causal boundaries, ground-truth separation, and verification tests enforcing non-anticipative execution.

| Component | Input | Output | Available at time $t$? | Contains Ground Truth? | Scheduler Visible? | Causal Permission (Allowed?) | Verification Test |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`SpectrumEnvironment`** | Scenario config, simulation step parameters | Incident RF pulses, true emitter states, true trajectory | Yes (internal) | **YES** (true emitter ID, true PRI, mode) | **NO** (Strictly encapsulated inside simulator) | **YES** (Ground truth generation only) | `test_i_independent_parallel_emitter_streams` |
| **`SieveReceiver`** | Center frequency $f_c$, dwell mode $m$, incident pulses | Detected pulses within $[f_{\text{lower}}, f_{\text{upper}}]$ and $[t_{\text{start}}, t_{\text{end}}]$ | Yes | **NO** (Labels stripped in operational flow) | **NO** (Feeds perception buffer) | **YES** (Physical aperture filtering) | `test_a_out_of_band_rejection`, `test_b_temporal_aperture_filtering`, `test_c_retune_latency_dead_time`, `test_d_sensitivity_threshold_clipping` |
| **`ReceiverAdapter`** | Incident pulses, tune command $(f_c, T_d, \tau_{\text{retune}})$ | Causal `DetectionObservation` stream | Yes | **NO** (Sanitized to physical observables) | **NO** (Feeds state builder / perception) | **YES** (Causal hardware abstraction) | `test_p_pulse_buffer_shortcut_isolation` |
| **`MissionClock`** | Step progression, retune latency $\tau_{\text{retune}}$, dwell duration $T_d$ | Current mission time $t_{\text{mission}}$ | Yes | **NO** (Monotonic scalar clock) | **YES** (Via state urgency & timing telemetry) | **YES** (Monotonic time base) | `test_e_monotonic_mission_clock`, `test_n_dwell_mode_duration_fidelity` |
| **`EmitterTracker`** | Raw detected PDWs ($\text{CF}, \text{PW}, \text{AoA}, P_{\text{rx}}$) | Local cluster tracks, persistent `track_id` | Yes | **NO** (Unsupervised online DBSCAN) | Indirectly (Via belief state features) | **YES** (Causal perception) | `test_m_simultaneous_multi_emitter_deinterleave` |
| **`TemporalPredictor`** | Tracker `track_id`, pulse ToA $t_{\text{pulse}}$ | Predicted next ToA $\hat{t}_{\text{next}}$, estimated PRI $\widehat{\text{PRI}}$ | Yes | **NO** (Ingests only tracker-derived tracks) | Indirectly (Via belief PRI & urgency) | **YES** (Causal temporal estimation) | `test_k_ground_truth_id_invariance` |
| **`SpatialTracker`** | Tracker `track_id`, measured AoA, power $P_{\text{rx}}$ | Spatial bearing & variance estimates | Yes | **NO** (Only measured observables) | Indirectly (Via belief AoA features) | **YES** (Causal bearing estimation) | `test_k_ground_truth_id_invariance` |
| **`BeliefState`** | Detections from current dwell, decayed historical features | 360-dimensional belief vector $\mathbf{o} \in [0, 1]^{360}$ | Yes | **NO** (Normalized statistical features) | **YES** (Direct agent input) | **YES** (Markovian state representation) | `test_f_causal_observation_generation`, `test_l_cold_start_observation_validity` |
| **`CognitiveRFScanEnv`** | Action $a \in [0, 179]$, step invocation | $(\mathbf{o}_{t+1}, r_t, d_t, \text{info}_t)$ | Yes | Internal reward evaluates GT; observation has **NO GT** | Observation & info are visible; **obs has NO GT** | **YES** (Causal MDP environment) | `test_j_future_information_isolation`, `test_k_ground_truth_id_invariance` |
| **`SmartScanMoE`** | 360-d observation vector, tracker `track_id` | Action distribution, expert gating weights | Yes | **NO** (Zero reference to `emitter_id`) | **YES** (Model output) | **YES** (Cognitive gating policy) | `test_k_ground_truth_id_invariance`, `test_phase1_causality.py` |
| **`DRQNScheduler`** | Causal observation sequence $\mathbf{o}_{0:t}$, action history | Q-values $Q(s_t, a)$, selected action $a_t$ | Yes | **NO** (Pure function of causal history) | **YES** (Agent decision) | **YES** (Autonomous scheduler) | `test_o_deterministic_replay_equivalence` |
| **Reward Function** | Current dwell detections, simulator truth emitters | Scalar reward $r_t$ | Yes (at step completion) | **YES** (Compares detection against ground truth for scoring) | **NO** (Reward scalar only, not in observation) | **YES** (Standard RL MDP reward evaluation) | `test_contract_validation.py` |
| **Offline Evaluation Metrics** | Full episode trajectory, environment ground truth | DRDO FoM metrics (Probability of Intercept, Sensitivity, etc.) | Post-step / Post-episode | **YES** (Evaluates true ground-truth coverage) | **NO** (Populates `info["eval_metrics"]` for logging only) | **YES** (Offline verification and validation) | `test_benchmark_v2_canonical.py` |

---

## 2. Leakage Eradication Verification Summary

### 2.1 Emitter-ID Pathway Audit
- **Audit Target:** `CognitiveRFScanEnv._update_temporal_predictor()`
- **Status:** REMEDIATED & VERIFIED
- **Mechanism:** Ground truth `d.emitter_id` was severed. Replaced by `EmitterTracker._last_pulse_tracks` (unsupervised clustering). Detections without an assigned tracker identity are discarded from predictor ingestion.

### 2.2 MoE Predictive Input Audit
- **Audit Target:** `SmartScanMoE.update_detections()`
- **Status:** REMEDIATED & VERIFIED
- **Mechanism:** Direct access to `d.emitter_id` was stripped. Detections only update internal tracker statistics if `track_id is not None`.

### 2.3 Non-Anticipative Information Boundary Audit
- **Audit Target:** `ReceiverAdapter.feed_incident_rf()` & `SieveReceiver.detect_pulses()`
- **Status:** REMEDIATED & VERIFIED
- **Mechanism:** Pulse ingestion explicitly clamped to `max_time_us = dwell_end_us`. Future pulses are unbuffered and cannot alter current receiver detection state.

---

## 3. Invariance Proofs

### Proof 1: Emitter ID Invariance ($\mathcal{T}_{\text{perm}}$)
Let $\mathcal{E}$ be a set of ground truth emitters with IDs $\{e_1, e_2, \dots, e_N\}$ and physical pulse characteristics $\{P_1, P_2, \dots, P_M\}$. Let $\pi$ be an arbitrary permutation of emitter labels:
$$\pi: \{e_1, \dots, e_N\} \to \{e'_1, \dots, e'_N\}$$
Since the receiver downconverts only physical pulse attributes $(t, f, \text{pw}, \text{amp}, \text{aoa})$ and the perception tracker clusters purely in continuous measurement space, the resulting observation vector satisfies:
$$\mathbf{o}(\mathcal{E}) = \mathbf{o}(\pi(\mathcal{E})) \quad \forall t$$
Verified empirically by `test_k_ground_truth_id_invariance` with zero divergence ($\Delta \mathbf{o} = 0.00000000$).

### Proof 2: Future Information Invariance ($\mathcal{T}_{\text{future}}$)
Let $\mathcal{P}_t$ be the set of incident pulses with $\text{ToA} \le t$. Let $\mathcal{P}_{>t}$ be any arbitrary set of future pulses with $\text{ToA} > t$.
The observation vector at time $t$ satisfies:
$$\mathbf{o}(\mathcal{P}_t \cup \mathcal{P}_{>t}) = \mathbf{o}(\mathcal{P}_t)$$
Verified empirically by `test_j_future_information_isolation` with zero divergence ($\Delta \mathbf{o} = 0.00000000$).

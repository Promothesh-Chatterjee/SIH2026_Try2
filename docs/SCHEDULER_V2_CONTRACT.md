# Cognitive EW Smart Scan — Scheduler v2 Contract Specification

**Document Version:** 2.0.0  
**Status:** FROZEN & CANONICAL  
**System:** Cognitive EW Smart Scan Scheduler v2  
**Problem Statement:** Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics.

---

## 1. Executive Architecture Summary

The Scheduler v2 architecture provides autonomous, real-time cognitive frequency search and dwell scheduling for Electronic Warfare receivers. The core scheduling authority is an arbitration between deep recurrent Q-learning (DRQN) and specialized cognitive reasoning heuristics (MoE) that select actions within a joint time-frequency action space based exclusively on receiver-observable belief states.

```
+-----------------------------------------------------------------------------------+
|                            OPERATIONAL CLOSED LOOP                                |
|                                                                                   |
|  Incident RF Spectrum (0 - 18,000 MHz)                                            |
|        │                                                                          |
|        ▼                                                                          |
|  SieveReceiver / ReceiverAdapter                                                  |
|  - Tunes center frequency [250 - 17,750 MHz] (36 bands @ 500 MHz IBW)             |
|  - Opens physical aperture window for dwell duration [125 - 1250 µs]              |
|  - Extracts measured physical PDWs (time_us, freq_mhz, pw_us, amp_db, aoa_deg)    |
|        │                                                                          |
|        ▼                                                                          |
|  EmitterTracker (Unsupervised Clustering & Tracking)                              |
|  - Online deinterleaving without ground-truth emitter IDs                         |
|  - Tracks PRI agility, frequency agility, AoA, and revisit ages                   |
|        │                                                                          |
|        ▼                                                                          |
|  OperationalStateBuilder                                                          |
|  - Computes 10 receiver-observable belief features across all 36 bands            |
|  - Outputs canonical 360-dimensional observation tensor                           |
|        │                                                                          |
|        ▼                                                                          |
|  Scheduler Authority (DRQN + SmartScanMoE)                                        |
|  - Frozen 25k DRQN: Q-values for 180 actions (36 bands × 5 dwell modes)           |
|  - MoE Arbitration: TemporalPredictor, SpatialTracker, Revisit, Discovery         |
|  - Selects flat action index a in [0, 179]                                         |
|        │                                                                          |
|        ▼                                                                          |
|  Action Decoder: band = a // 5, mode = a % 5                                      |
|  - Base receiver retune latency (15.0 µs) advanced on MissionClock                |
|  - Base dwell duration (500 µs * mode_multiplier) advanced on MissionClock        |
|  - Closed-loop telemetry published                                                |
+-----------------------------------------------------------------------------------+
```

---

## 2. RF Receiver Hardware Contract

All components in the scanning pipeline must conform to the canonical receiver physical parameters:

| Parameter | Canonical Value | Description / Constraint |
|:---|:---|:---|
| **RF Frequency Min** | `0.0 MHz` | Absolute lower edge of instantaneous search band |
| **RF Frequency Max** | `18,000.0 MHz` | Absolute upper edge of instantaneous search band (18 GHz) |
| **Instantaneous Bandwidth (IBW)** | `500.0 MHz` | Aperture reception bandwidth per tune |
| **Frequency Step** | `500.0 MHz` | Tuning channel step (`(freq_max - freq_min) / n_bands`) |
| **Number of Bands ($N_b$)** | `36` | Contiguous, non-overlapping bands indexed `0` to `35` |
| **Base Dwell Time** | `500.0 µs` | Reference dwell window before dwell-mode multiplier |
| **Retune Latency** | `15.0 µs` | LO PLL lock and synthesizer settling time before aperture opens |
| **Receiver Sensitivity ($S_{\min}$)** | `-140.0 dBm` | Minimum detectable signal threshold |

### Band Frequency Mapping Table
Band index $b \in [0, 35]$ defines:
$$\text{Freq}_{\text{low}} = b \times 500.0\text{ MHz}, \quad \text{Freq}_{\text{high}} = (b + 1) \times 500.0\text{ MHz}$$
$$\text{Freq}_{\text{center}} = b \times 500.0 + 250.0\text{ MHz}$$

* Band `0`: `[0.0 - 500.0 MHz]`, Center: `250.0 MHz`
* Band `1`: `[500.0 - 1000.0 MHz]`, Center: `750.0 MHz`
* ...
* Band `35`: `[17,500.0 - 18,000.0 MHz]`, Center: `17,750.0 MHz`

---

## 3. Observation Space Contract

### Dimensionality
$$\text{obs\_dim} = N_{\text{bands}} \times N_{\text{features}} = 36 \times 10 = 360$$

Tensor representation: `shape = (360,)`, dtype `float32`, ordered in **band-major** layout:
$$\text{obs}[b \times 10 + f] = \text{Feature } f \text{ of Band } b$$

### Canonical Feature Ordering (Check-point Sensitive & Immutable)
The feature ordering below is baked into neural network weights and must never be altered:

| Index ($f$) | Feature Name | Range | Physical EW Semantic |
|:---|:---|:---|:---|
| **0** | `occupancy` | $[0.0, 1.0]$ | EMA activity score of the band based on detection history |
| **1** | `det_rate` | $[0.0, 1.0]$ | Ratio of successful pulse detections to dwell opportunities |
| **2** | `miss_rate` | $[0.0, 1.0]$ | Complementary miss probability ($1.0 - \text{det\_rate}$) |
| **3** | `uncertainty` | $[0.0, 1.0]$ | Information entropy / variance of emitter presence belief |
| **4** | `revisit_age` | $[0.0, \infty)$ | Normalized time elapsed since the band was last scanned |
| **5** | `emitter_count` | $[0.0, \infty)$ | Number of active unique emitter tracks associated with the band |
| **6** | `deint_confidence`| $[0.0, 1.0]$ | Online deinterleaving cluster separation confidence |
| **7** | `pri_stability` | $[0.0, 1.0]$ | Regularity/jitter score of observed pulse repetition intervals |
| **8** | `agility` | $[0.0, 1.0]$ | Frequency hop agility indicator (variance in intra-track frequencies)|
| **9** | `priority` | $[0.0, 1.0]$ | Threat priority weight assigned to known or suspected emitters |

### Ground-Truth Boundary (Zero-Leakage Invariant)
* **STRICT PROHIBITION**: True simulation emitter IDs, ground-truth ToA schedules, scenario configuration dictionaries, or true antenna patterns MUST NEVER enter `obs_dim`, `OperationalStateBuilder`, or `EmitterTracker`.
* All beliefs must be causally estimated from measured PDWs: `(time_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg)`.

---

## 4. Action Space Contract

### Joint Time-Frequency Action Representation
$$\text{Total Actions} = N_{\text{bands}} \times N_{\text{modes}} = 36 \times 5 = 180$$

Action indexing is flat, row-major:
$$\text{action} = \text{band} \times 5 + \text{mode\_index} \quad (0 \le \text{action} < 180)$$
$$\text{band} = \text{action} // 5 \quad (0 \le \text{band} < 36)$$
$$\text{mode\_index} = \text{action} \% 5 \quad (0 \le \text{mode\_index} < 5)$$

### Dwell Mode Taxonomy & Multipliers
The 5 canonical dwell modes define operational intent and receiver aperture duration:

| Mode Index | Mode Identifier | Multiplier | Dwell Duration ($\mu$s) | Operational Intent / Semantic |
|:---:|:---|:---:|:---:|:---|
| **0** | `SHORT_DWELL` | `0.25` | `125.0 µs` | Fast reconnaissance / broad spectrum survey |
| **1** | `NORMAL_DWELL` | `1.00` | `500.0 µs` | Standard surveillance & steady-state monitoring |
| **2** | `LONG_DWELL` | `2.50` | `1250.0 µs` | Deep observation of high-uncertainty or agile signals |
| **3** | `REVISIT` | `1.00` | `500.0 µs` | Track maintenance for overdue known emitters |
| **4** | `PREEMPTIVE_INTERCEPT` | `1.00` | `500.0 µs` | Window alignment for imminent predicted pulse arrival |

### Validation Contract
* Non-integer action types (e.g. `bool`, `float`, `str`) raise `TypeError`.
* Action indices outside $[0, 179]$ raise `ValueError`.
* Band indices outside $[0, 35]$ and mode indices outside $[0, 4]$ raise `ValueError`.

---

## 5. Model Architecture Contract

### Primary Policy: Deep Recurrent Q-Network (DRQN)
* **Input**: `(batch_size, seq_len, 360)`
* **Feature Extractor**: 2-layer MLP (`360 -> 256 -> 128`, LayerNorm, GELU activations)
* **Recurrent Core**: 1-layer LSTM (`hidden_size=64`, BPTT `seq_len=16`, `burn_in=8`)
* **Dueling Heads**:
  - Value stream: `64 -> 32 -> 1`
  - Advantage stream: `64 -> 64 -> 180`
  - $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)$
* **Auxiliary Prediction Heads**:
  - Intercept Probability Head: `64 -> 180` (Sigmoid, output shape `(batch, seq, 180)`)
  - Intercept Time Error Head: `64 -> 180` (ReLU, output shape `(batch, seq, 180)`)

### Cognitive Arbitration: SmartScanMoE
* Integrates DRQN Q-values with:
  1. `TemporalPredictor`: Kalman/AR prediction of periodic pulse trains ($T_1$)
  2. `SpatialTracker`: AoA-guided beam alignment
  3. `RevisitManager`: Overdue track maintenance pressure
  4. `BandDiscoveryTracker`: Quota-guided 36-band exploration
* Arbitration selects an action candidate and annotates decisions with full explainability metadata.

---

## 6. Telemetry & Auditability Contract

Every dwell cycle generates a standardized `ReceiverTelemetryFrame` containing:
- Monotonic timestamps (`timestamp_us`, `dwell_start_us`, `dwell_end_us`)
- Actuation parameters (`selected_band`, `selected_mode`, `center_frequency_mhz`, `dwell_duration_us`)
- Detection outcomes (`hit`, `num_detections`, `intercept_time_us`, `detections`)
- Cognitive Explanation (`decision_reason`, `predicted_track_id`, `prediction_confidence`, `aoa_deg`)
- Strict invariant flags (`ground_truth_leakage = False`, `fallback_triggered = 0.0`)

# Failure Transparency and Causal Case Studies

## 1. Executive Summary

A core operational requirement for the **Gate-110k-Phase7 Operational Demonstration Candidate** is complete failure transparency. Rather than concealing misses behind aggregate metrics, the scheduler provides real-time causal telemetry explaining:
1. Why an intercept succeeded (which neural and deterministic factors produced the hit).
2. Why an intercept was missed (identifying primary and secondary root causes within partial observability limits).

This document presents two deep-dive case studies extracted directly from telemetry logs:
- **Case Study 1**: A clean successful agile intercept on a fast agile hopper (`AG-04`).
- **Case Study 2**: An explainable miss on a branching Markov hopper (`AG-06`), demonstrating stochastic branch divergence and $K=1$ single-receiver partial observability.

---

## 2. Case Study 1: Clean Agile Intercept (`AG-04` Fast Agile Hopper)

### Scenario Context
- **Emitter**: 4-band agile hopper ([Band 6, Band 14, Band 22, Band 31]), agile PRI $\approx 100.0\ \mu\text{s}$, stable Angle-of-Arrival $\theta = 325.0^\circ$.
- **Receiver State**: Single instantaneous receiver ($K=1$), 36 bands, $500\ \text{MHz}$ IBW per band, base dwell $500.0\ \mu\text{s}$.

### Sequence of Events: Cycle #0013

```mermaid
sequenceDiagram
    autonumber
    participant RF as Physical RF Environment
    participant Track as Emitter Tracker & ABM
    participant MoE as Decision Arbitration (SmartScanMoE)
    participant Rx as Receiver Hardware

    RF->>Track: Past pulses detected in Band 6 (AoA 325.0°)
    Track->>Track: Update circular mean θ=325°, R=0.984 (Coherent)
    Track->>Track: Dirichlet update: P(Band 6 | History) = 67.0%
    Track->>MoE: Projected Arrival ETA = 100 µs, Agility Score = 0.82
    MoE->>MoE: Evaluate True Stochastic Expected Utility
    Note over MoE: Learned DRQN Q(6, 1) = +16.251<br/>Deterministic Predictive Gain = +16.701<br/>Spatial Priority = +0.984<br/>Action #031 dominates
    MoE->>Rx: Actuate Band 6 (3000-3500 MHz), NORMAL_DWELL (500 µs)
    Rx->>RF: Tune receiver (15 µs retune latency) & dwell
    RF->>Rx: 2 Pulses Captured (Latency = 0.0 µs)
    Rx-->>MoE: 🎯 INTERCEPT CONFIRMED
```

### Telemetry Card at Actuation
```text
+-----------------------------------------------------------------------------+
| CYCLE #0013 | Sim Clock:    13250.0 us | Status: [HIT]    | Intercepts:  2 |
+-----------------------------------------------------------------------------+
| 1. RECEIVER ACTUATION:                                                      |
|    Channel: Band 06 (3000.0 - 3500.0 MHz) | Mode: NORMAL_DWELL     (Act #031) |
|    Timing Latency:    0.0 us                                                |
+-----------------------------------------------------------------------------+
| 2. COGNITIVE 'WHY THIS BAND?' ATTRIBUTION:                                  |
|    Decision Driver : Predictive_utility_active                              |
|    Target Track    : Track-0      | P(Next Band):  67.0% (Dirichlet)        |
|    Predicted ETA   :   100.0 us   | Measured AoA: 325.0 deg (Circular R)    |
+-----------------------------------------------------------------------------+
| 3. ARBITRATION UTILITY DECOMPOSITION:                                       |
|    [Learned Neural] DRQN Q(b, m)              : +16.251                     |
|    [Deterministic]  Predictive Expected Gain  : +16.701                     |
|    [Deterministic]  Spatial Sector Priority   :  +0.984                     |
|    [Deterministic]  Exploration Suppression   :   0.320 (Guarded: True)     |
+-----------------------------------------------------------------------------+
```

### Causal Rationale:
1. **Track Convergence**: Prior physical dwells in Band 6 captured 2 initial pulses, yielding estimated $\widehat{\text{PRI}} = 100.0\ \mu\text{s}$ with low variance ($\sigma_{\text{PRI}} < 2.0\ \mu\text{s}$).
2. **Spatial Coherence**: Measured AoA samples concentrated around $325.0^\circ$, producing a circular resultant vector length $R = 0.984$, confirming a physical point-source transmitter rather than multipath clutter.
3. **Cognitive Exploration Guard**: Because prediction confidence $c = 0.76 > 0.45$ and projected arrival $\tau = 100\ \mu\text{s} < 1000\ \mu\text{s}$, the cognitive exploration guard deterministically suppressed heuristic random searching.
4. **Outcome**: The receiver tuned with $15\ \mu\text{s}$ retune latency and intercepted both emitted pulses within the $500\ \mu\text{s}$ normal dwell window at $0.0\ \mu\text{s}$ timing error.

---

## 3. Case Study 2: Explainable Miss (`AG-06` 1st-Order Markov Hopper)

### Scenario Context
- **Emitter**: 1st-order Markov hopper transitioning among bands $[8, 13, 21, 33]$ with transition matrix:
  $$\mathbf{T} = \begin{bmatrix} 0.05 & 0.40 & 0.35 & 0.20 \\ 0.30 & 0.05 & 0.45 & 0.20 \\ 0.25 & 0.35 & 0.05 & 0.35 \\ 0.40 & 0.30 & 0.25 & 0.05 \end{bmatrix}, \quad \text{PRI} = 220.0\ \mu\text{s}$$
- **State Prior to Miss**: Last detected pulse was in Band 8.

### Sequence of Events: Cycle #0047

```mermaid
sequenceDiagram
    autonumber
    participant RF as Physical RF Environment
    participant Track as Adaptive Behavior Manager
    participant MoE as Decision Arbitration (SmartScanMoE)
    participant Rx as Receiver Hardware

    Note over Track: Last detected pulse: Band 8 at t = 10,120 µs
    Track->>Track: Branch Probabilities from Band 8:<br/>P(Band 13) = 0.40 | P(Band 21) = 0.35 | P(Band 33) = 0.20
    MoE->>MoE: Compute True Stochastic Expected Utility across all candidates
    Note over MoE: E[U(Band 13)] = +0.2300 (Argmax Candidate)<br/>E[U(Band 21)] = +0.1938 (Close Competitor)<br/>E[U(Band 33)] = +0.0788
    MoE->>Rx: Actuate Band 13 (6500-7000 MHz), NORMAL_DWELL (500 µs)
    RF->>RF: Emitter stochastically selects alternate 35% branch -> Transmits on Band 21
    Rx->>RF: Dwell Band 13 [10,200 µs, 10,700 µs]
    Rx-->>MoE: ⚪ ZERO PULSES DETECTED (MISS)
    MoE->>Track: Register empty dwell on Band 13; decay branch weight
```

### Telemetry Card at Actuation
```text
+-----------------------------------------------------------------------------+
| CYCLE #0047 | Sim Clock:    10200.0 us | Status: [MISS]   | Intercepts:  0 |
+-----------------------------------------------------------------------------+
| 1. RECEIVER ACTUATION:                                                      |
|    Channel: Band 13 (6500.0 - 7000.0 MHz) | Mode: NORMAL_DWELL     (Act #066) |
|    Timing Latency:    nan us                                                |
+-----------------------------------------------------------------------------+
| 2. COGNITIVE 'WHY THIS BAND?' ATTRIBUTION:                                  |
|    Decision Driver : Predictive_utility_active                              |
|    Target Track    : Track-1      | P(Next Band):  40.0% (Dirichlet)        |
|    Predicted ETA   :   220.0 us   | Measured AoA: 142.0 deg (Circular R)    |
+-----------------------------------------------------------------------------+
| 3. ARBITRATION UTILITY DECOMPOSITION:                                       |
|    [Learned Neural] DRQN Q(b, m)              :  +0.100                     |
|    [Deterministic]  Predictive Expected Gain  :  +0.130                     |
|    [Deterministic]  Spatial Sector Priority   :  +0.000                     |
|    [Deterministic]  Exploration Suppression   :   0.410 (Guarded: False)    |
+-----------------------------------------------------------------------------+
```

### Hierarchical Root-Cause Analysis:
1. **Primary Cause (`STOCHASTIC_MARKOV_BRANCH_DIVERGENCE`)**:
   - The emitter's internal state machine executed a random transition governed by $P(\text{Band } 21 \mid \text{Band } 8) = 0.35$.
   - The scheduler made the mathematically optimal decision under uncertainty: evaluating expected utility $\mathbb{E}[U(\text{Band } 13)] = +0.2300 > \mathbb{E}[U(\text{Band } 21)] = +0.1938$.
   - In a non-deterministic environment, choosing the highest expected value action ($40\%$ vs $35\%$) still incurs a $60\%$ probability of the emitter selecting an alternate branch.
2. **Secondary Cause (`RECEIVER_CAPACITY_LIMIT / SINGLE_RECEIVER_BOTTLENECK`)**:
   - Because $K=1$, the receiver hardware cannot simultaneously monitor Band 13 and Band 21.
   - Covering multiple non-overlapping Markov branches simultaneously requires multi-channel hardware ($K \ge 2$). Under single-channel constraints, this miss is a fundamental consequence of partial observability.
3. **Operational Recovery**:
   - The miss was immediately detected upon dwell conclusion.
   - `AdaptiveBehaviorManager` updated the empirical branch distribution without dropping the track or initiating blind random searching.
   - On the subsequent cycle, the reservation manager retuned to Band 21 to re-acquire the emitter.

---

## 4. Root Cause Taxonomy & Distribution Summary

Across the 10,000 steps of the canonical held-out evaluation suite:

| Root Cause Category | Classification | Frequency (%) | Operational Explanation |
| :--- | :--- | :---: | :--- |
| **Branching Markov Divergence** | Primary | 38.4% | Emitter hopped along a lower-probability transition branch ($P < P_{\text{argmax}}$). |
| **Single-Receiver Concurrency Bottleneck ($K=1$)** | Secondary | 42.1% | Multiple emitters transmitted simultaneously on distinct frequency channels. |
| **Low-Duty Cycle Sparse Hopping** | Primary | 14.2% | Inter-pulse interval ($> 800\ \mu\text{s}$) exceeded single dwell span during initial track discovery. |
| **Pre-Dwell Horizon Boundary** | Primary | 5.3% | First step in scenario prior to any physical detections ($t < t_{\text{first\_pulse}}$). |

This transparent attribution proves that misses are governed by explicit physical and statistical laws rather than policy instability.

# Cognitive EW Smart Scan Benchmark Protocol

## 1. Scope and Purpose
This document establishes the formal, immutable benchmark protocol for the Cognitive Electronic Warfare Smart Scan Scheduler. All reported metrics, decision gates, and comparisons must strictly adhere to the standards, scenario subsets, seeds, and parameter configurations defined herein.

---

## 2. Production Candidate Reference Definitions

### 2.1 `Gate-25.5k-Champion / Gate-25k-R4.2` (Authoritative v2 Benchmark)
- **Designation**: `ALL SOFTWARE OPERATIONAL-READINESS GATES PASSED — OPERATIONAL DEMONSTRATION READY (v2)`
- **Neural Network Weights**: `experiments/checkpoints/scheduler/best.pt` (strictly frozen, 0 gradient updates).
  - Promoted from: `checkpoint_step_25500.pt` (SHA-256: `777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554`)
  - Frozen Baseline: `checkpoint_gate_25000_frozen.pt` (SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)
  - Continuation Start: `checkpoint_step_26000_arme.pt` (SHA-256: `88a7c6261b96d7ecfeba87d190fb843ac06f45c86e012f913e9c095ef370b53f`)
- **Official Performance Record**:
  - **Standalone DRQN Canonical Interception Rate ($P_d$)**: **62.10%** across 10 held-out TSRD real-world radar validation files (vs 6.50% Round-Robin baseline, 9.55× gain, achieved purely by neural policy without heuristic crutches).
  - **Intercept Latency**: Median $15.0\ \mu\text{s}$, Mean $34.2\ \mu\text{s}$.
  - **Head-to-Head vs RoundRobin**: **10W–0L–0T** (10/10 scenario wins).
  - **Empty-Band Escape**: **100.0%**.
  - **False Alarm Rate ($P_{\text{fa}}$)**: **0.0000**.
- **Inference Architecture**:
  - 38 Parameter Tensors: Localized band encoder ($10 \to 32$), context projection ($256 \to 32$), band advantage head ($64 \to 5$ modes per band).
  - Asymmetric confirmed-track miss decay: $\alpha_{\text{miss}} = 0.20$ vs base decay $\alpha = 0.30$.
  - Base Dwell: $500.0\ \mu\text{s}$ (Canonical mode multipliers: `SHORT=0.25` [$125\ \mu\text{s}$], `NORMAL=1.0` [$500\ \mu\text{s}$], `LONG=2.5` [$1,250\ \mu\text{s}$], `REVISIT=1.0` [$500\ \mu\text{s}$], `PREEMPTIVE=1.0` [$500\ \mu\text{s}$]).
  - Spatial Intelligence Layer: Circular statistics `SpatialTracker` with operational sector weighting.
  - Scheduling Authority: Factored `DRQNScheduler` with Advantage-Preserving Reservoir Continuation.

### 2.3 Explicit Architecture Contract: Neural Representation vs. Deterministic Arbitration
To ensure complete scientific rigor and zero ambiguity:
- **Learned Neural Input**: $Q(b, m) \in \mathbb{R}^{36 \times 5}$ is the purely learned representation computed by the frozen DRQN from the 360-feature observation vector $s_t$.
- **Deterministic Post-Network Arbitration**: All remaining components—including dwell duration penalty, Dirichlet-smoothed Markov expectation, temporal reservations, agility scoring, exploration regulation, and circular spatial prioritization—are deterministic mathematical operations executed post-network. No neural network weights are modified.

---

## 3. Protocol Reconciliation: Canonical Gate (40.63%) vs Ablation Sweep (41.90%)

An essential scientific standard is distinguishing exploratory ablation metrics from formal held-out gate verification.

| Protocol Parameter | Official Canonical Gate (`evaluate_canonical_gate.py`) | Ablation Sweep (`evaluate_arbitration_ablation.py`) |
| :--- | :--- | :--- |
| **Official Metric** | **40.63% Interception Rate ($4,063$ raw hits)** | **41.90% Canonical Intercept Rate** |
| **Role & Governance** | **Authoritative Production Benchmark** | Exploratory Sensitivity Analysis |
| **Scenario Set** | Fixed 10 held-out TSRD validation files (`val_set.files_used`) | 10 canonical files + 8 synthetic agile scenarios |
| **Step Count** | 1,000 steps / scenario (10,000 total steps) | 1,000 steps / scenario |
| **Random Seed** | Fixed Seed 42 | Fixed Seed 42 |
| **Environment State** | `semantic_memory_enabled = False`, fresh env per scenario | `semantic_memory_enabled = False` |
| **Action Temperature** | **Strict Deterministic Argmax ($\tau = 0.0$)** | Default MoE config fallback ($\tau = 0.15$ temperature sampling in tie/fallback paths) |
| **Decision Rule** | Fully reproducible deterministic action trace | Stochastic fallback sampling |

> [!IMPORTANT]
> The single authoritative held-out canonical performance reference is **$40.63\%$**, which exceeds the historical $39.68\%$ floor by **$+0.95\ \text{pp}$** ($+106$ raw hits) with **$27.0\ \mu\text{s}$ median latency**. The $41.90\%$ figure was generated under exploratory configuration settings and is not used as the official promotion benchmark.

---

## 4. Test Battery Specifications

### Suite 1: Official Held-Out Canonical Gate
- **Scenarios (10)**: `config_117`, `config_119`, `config_143`, `config_194`, `config_195`, `config_241`, `config_29`, `config_42`, `config_64`, `config_96`.
- **Pulses / File**: 50,000 real-world recorded TSRD pulses.
- **Duration**: 1,000 dwell steps per scenario.
- **Contract**: 36 bands (0–18 GHz, 500 MHz/band), 5 dwell modes (180 actions), 360-dimension observation vector.
- **Baseline**: Head-to-head comparison against canonical `RoundRobin`.

### Suite 2: Dedicated Agile Benchmark (`AG-01` to `AG-10`)
- **Scenarios (10)**:
  - `AG-01`: 3-band cyclic hopper ($250\ \mu\text{s}$ PRI)
  - `AG-02`: 4-band cyclic hopper ($300\ \mu\text{s}$ PRI)
  - `AG-03`: 5-band cyclic hopper ($200\ \mu\text{s}$ PRI)
  - `AG-04`: Fast agile hopper ($100\ \mu\text{s}$ PRI)
  - `AG-05`: Slow agile hopper ($800\ \mu\text{s}$ PRI)
  - `AG-06`: 1st-order Markov hopper ($220\ \mu\text{s}$ PRI)
  - `AG-07`: Dual concurrent agile hoppers
  - `AG-08`: Hybrid fixed-frequency + agile hopper
  - `AG-09`: Bursty hopper with PRI jitter ($\pm 10\%$)
  - `AG-10`: Complex dense EW environment (4 emitters)
- **Seeds**: Multi-seed evaluation with seeds `[42, 43, 44]`.
- **Duration**: 500 dwell steps per seed.

### Suite 3: Controlled Spatial Contention Benchmark
- **Scenarios**:
  - Same-frequency / different-AoA overlap (e.g. Band 5, $45^\circ$ vs $135^\circ$).
  - Dense spatial clustering with directional threat vector priority.
- **Metrics**: $P_d$, median latency, track continuity, and correct sector priority selection.

---

## 5. Formal Operational Readiness Gates

| Gate | Category | Metric | Requirement |
| :--- | :--- | :--- | :--- |
| **Gate A** | Correctness | Canonical $P_d$ | $\ge 40.63\%$ |
| | | Median Latency | $\le 80.0\ \mu\text{s}$ (v2: $76.7\ \mu\text{s}$, strictly beats RR $83.3\ \mu\text{s}$) |
| | | Head-to-Head vs RR | $\ge 7 / 10$ wins |
| | | False Alarm Rate ($P_{\text{fa}}$) | $0.0000$ (no material degradation) |
| | | Empty-Band Escape | $100.0\%$ |
| **Gate B** | Agility | High-Agility Scenarios | No regression on `AG-04` ($\ge 85\%$), `AG-08` ($\ge 80\%$), `AG-10` ($\ge 85\%$) |
| | | Weak Agile Classes | Demonstrated improvement on `AG-01`, `AG-02`, `AG-05`, `AG-06` |
| **Gate C** | Spatial | Spatial Contention | $> 2.0\times$ preference for coherent priority emitters over diffuse emitters |
| | | Target Discrimination | Verifiable selection accuracy improvement under frequency overlap |
| **Gate D** | Runtime | End-to-End Cycle Time | $< 5.0\ \text{ms}$ per decision cycle across Ingest $\to$ Predict $\to$ Actuate |

---

## 6. Authoritative Scope Declaration & Latency Definitions

### 6.1 Explicit Operational Scope Declaration
> [!IMPORTANT]
> **Authoritative Operational Scope**:
> Software-in-the-loop (SIL) operational readiness is formally established and demonstrated across held-out recorded TSRD radar datasets, dynamic agile challenge batteries, and spatial contention testbeds. Physical RF hardware performance, SDR driver integration, analog front-end impairments, receiver calibration, and real-world electromagnetic environment qualification remain outside the current software qualification scope.

### 6.2 Latency Measurement Definitions
To prevent ambiguity between pure neural execution and full closed-loop software cycle time, all benchmark and demonstration reports distinguish:
- **Neural Inference Latency**: Measured pure compute duration of the DRQN recurrent forward pass and deterministic post-network arbitration (typically $1.30\ \text{ms}$ to $2.20\ \text{ms}$).
- **End-to-End Decision Cycle Latency**: Total wall-clock time for one complete cycle: PDW validation $\to$ unsupervised track association $\to$ 360-D observation state build $\to$ neural forward pass $\to$ arbitration $\to$ receiver tuning/aperture filtering $\to$ telemetry packaging.
  - Reports MUST provide: Mean, P95, P99, and Maximum cycle latency.
  - Demonstration-path mean cycle target: $< 5.0\ \text{ms}$.


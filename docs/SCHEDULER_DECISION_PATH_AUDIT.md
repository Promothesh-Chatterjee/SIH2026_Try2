# Cognitive EW Smart Scan: Scheduler Runtime Decision Path Audit

## 1. Executive Summary

This document presents a comprehensive, end-to-end technical audit of the runtime decision path within the Cognitive EW Smart Scan system. The scheduler is intended to operate as an autonomous Reinforcement Learning system using a Deep Recurrent Q-Network (DRQN) over a canonical time-frequency joint action space of 180 discrete actions ($36 \text{ frequency bands} \times 5 \text{ dwell modes}$).

The primary objective of this audit is to trace every link in the decision pipeline—from raw radio observations to the DRQN forward pass, exploration, heuristic interventions, MoE arbitration, receiver physical tuning, detection evaluation, reward computation, sequence replay, and backpropagation loss.

Crucially, this audit documents every point where the learned ML policy is modified, masked, or overridden by deterministic heuristics, semantic drivers, or mixture-of-experts (MoE) controllers, and defines runtime telemetry fields to ensure complete observability.

---

## 2. End-to-End Decision Flow Diagram

```mermaid
flowchart TD
    A[RF Scenario / TSRD Pulses] --> B[Sieve Receiver Physical Aperture]
    B --> C[Receiver Detection Observations]
    C --> D[Perception Pipeline: Deinterleaver + EmitterTracker]
    D --> E[Belief State: 10 Features x 36 Bands = 360-D]
    E --> F[DRQN Scheduler: LayerNorm -> 2-layer LSTM -> Dueling Q-Heads]
    F --> G[Raw Q-Values: Q(s, a) for 180 actions]
    G --> H{Action Selection Mode}
    
    H -->|Exploration| I1[Thompson Sampling Warmup < 5k steps OR Epsilon Random]
    H -->|Greedy band_first_decoupled| I2[Top-3 Boltzmann Band Selection + Hand-coded Mode Rules]
    H -->|MoE select_action| I3[Temporal Reservation / Predictive Hop / Top-k Q-margin / Cognitive Fallbacks]
    H -->|Direct ML flat_argmax| I4[Direct Argmax: action = argmax Q]
    
    I1 --> J[Final Selected Action: a in 0..179]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K[Action Decoding: band = a // 5, mode = a % 5]
    K --> L[Receiver Tuning: Center Frequency fc + Dwell Time Tdwell]
    L --> M[Radio Environment Simulation Event Advance]
    M --> N[Pulse Detection in Aperture Window: Hit vs Miss]
    N --> O[Reward Computation: receiver_reward_components]
    O --> P[Sequence Replay Buffer: Episodic BPTT Windows]
    P --> Q[Double-DQN Huber Loss + Aux Prediction Losses -> Gradient Update]
```

---

## 3. Step-by-Step Decision Path Audit

### Step 1: Perception and Belief State
* **Files**:
  - `src/environment/cognitive_rf_scan_env.py`
  - `src/perception/emitter_tracker.py`
  - `src/models/deinterleaver.py`
* **Functions**:
  - `CognitiveRFScanEnv._build_observation()`
  - `BeliefState.band_features(b)`
  - `BeliefState.record_visit()`
  - `BeliefState.update_from_perception()`
  - `BeliefState.update_uncertainty()`
  - `BeliefState.update_priority()`
* **Mechanics**:
  - The observation space is a continuous 360-dimensional box $[0.0, 1.0]^{360}$, representing 10 canonical features per frequency band across 36 bands:
    1. **Occupancy Probability ($p$)**: Exponential Moving Average (EMA, $\alpha=0.3$) of hit indicator. Initialized to neutral activity prior 0.5.
    2. **Detection Rate**: Cumulative ratio of hits to visits ($\text{hits} / \max(1, \text{visits})$).
    3. **Miss Rate**: $1.0 - \text{detection\_rate}$.
    4. **Uncertainty**: Aleatoric uncertainty ($1.0 - |2p - 1|$) blended with an epistemic prior ($1.0 - e^{-\text{visits}/4.0}$). Max 1.0 when unvisited.
    5. **Normalized Revisit Age**: Clamped time elapsed since last dwell on this band: $\min(\text{age}, 50) / 50.0$.
    6. **Estimated Emitter Count**: Angle-of-Arrival (AoA) bearing diversity and pulse width clustering proxy: $\min(\text{unique\_bearings}/5.0, 1.0)$.
    7. **Deinterleaver Confidence**: Clustering confidence score based on cluster compactness and pulse support.
    8. **PRI / Periodicity Stability**: Inverse of pulse repetition interval coefficient of variation ($1 / (1 + \text{CV}_{\text{PRI}})$).
    9. **Frequency Agility Indicator**: Normalized intra-band frequency standard deviation ($\text{std}(f) / 100 \text{ MHz}$).
    10. **Priority Score**: Composite Urgency:
        $$\text{Priority} = w_{\text{st}} \cdot \text{Age} + w_{\text{occ}} \cdot p + w_{\text{unc}} \cdot \text{Uncertainty} + w_{\text{per}} \cdot \text{PeriodicUrgency} + w_{\text{sem}} \cdot \text{SemanticBoost}$$
* **Causal Integrity**:
  - No ground-truth emitter labels (`emitter_id`) or future arrival knowledge ever enters `_build_observation()`. All 10 features are computed from physical detections causally recorded by the receiver aperture.

---

### Step 2 & 3: DRQN Forward Pass and Q-Values
* **Files**:
  - `src/models/drqn_scheduler.py`
* **Functions**:
  - `DRQNScheduler.forward(obs, hidden)`
  - `DRQNScheduler.act(obs, hidden, mode_selection, tau)`
* **Architecture**:
  - Input LayerNorm over feature dimension 360.
  - 2-layer LSTM with 256 hidden units (`batch_first=True`).
  - Dueling streams:
    - **Value Stream**: $\text{Linear}(256 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 1) \implies V(s) \in \mathbb{R}^1$
    - **Advantage Stream**: $\text{Linear}(256 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 180) \implies A(s, a) \in \mathbb{R}^{180}$
    - **Dueling Combination**:
      $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{180} \sum_{a'=0}^{179} A(s, a') \right)$$
  - Auxiliary heads:
    - `intercept_prob_head`: $\text{Linear}(256 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 180) \to \text{Sigmoid} \implies \hat{P}(\text{hit} \mid s, a)$
    - `intercept_time_head`: $\text{Linear}(256 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 180) \to \text{Softplus} \implies \hat{t}_{\text{intercept}}(\mu s)$
* **Raw DRQN Decision**:
  - The unadulterated ML choice is:
    $$a_{\text{raw}} = \arg\max_{a \in [0, 179]} Q(s, a)$$
    $$\text{band}_{\text{raw}} = a_{\text{raw}} // 5, \quad \text{mode}_{\text{raw}} = a_{\text{raw}} \% 5$$

---

### Step 4: Exploration Mechanisms
* **Files**:
  - `src/training/train_scheduler.py`
  - `src/training/thompson_sampling.py`
* **Mechanisms**:
  1. **Thompson Warmup Phase ($\text{step} < 5,000$)**:
     - `ThompsonSamplingExplorer.select_action()` samples Beta distributions for band occupancy and pairs with heuristic modes.
  2. **Epsilon Exploration Phase ($\text{step} \ge 5,000$)**:
     - Exploration probability: $\epsilon(t) = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) e^{-t / \text{decay}}$.
     - **CRITICAL LOOPHOLE IDENTIFIED**: In `train_scheduler.py` lines 605-608:
       ```python
       b_rand = random.randint(0, n_bands - 1)
       m_rand = int(np.random.choice([0, 1, 2], p=[0.10, 0.70, 0.20]))
       action = b_rand * n_modes + m_rand
       ```
       Random exploration **strictly restricted mode selection to $\{0, 1, 2\}$** (`SHORT_DWELL`, `NORMAL_DWELL`, `LONG_DWELL`). Dwell modes 3 (`REVISIT`) and 4 (`PREEMPTIVE_INTERCEPT`) were **never explored** via $\epsilon$-greedy exploration ($72$ out of $180$ actions had zero exploration probability).

---

### Step 5: Heuristic Overrides in DRQN (`band_first_decoupled`)
* **Files**:
  - `src/models/drqn_scheduler.py` (lines 228-267)
  - `src/models/baseline_suite.py` (`DRQNBaseline.act`)
* **Overrides Identified**:
  When `mode_selection="band_first_decoupled"` (the default in rollout):
  1. **Band Overrides (Boltzmann Temperature)**:
     - Band score is computed as $Q_{\text{band}}(b) = \max_{m} Q(b, m)$.
     - If $\tau > 0$ (default $\tau = 0.15$): Takes top-3 bands and draws from a Boltzmann distribution. The network's argmax band is overridden stochastically.
  2. **Mode Overrides (Hand-coded Heuristics Displacing ML)**:
     - The network's Q-values for mode selection are **completely discarded**.
     - Dwell mode is assigned via hand-coded rules:
       - If $\text{uncertainty} > 0.6$ or $\text{revisit\_age} > 0.6 \implies \text{mode} = 2$ (`LONG_DWELL`).
       - Else if $\text{consecutive\_empty} \ge 2$ and $\text{occupancy} < 0.1 \implies \text{mode} = 0$ (`SHORT_DWELL`).
       - Else if $Q(\text{LONG}) > Q(\text{NORMAL}) + 0.05 \implies \text{mode} = 2$ (`LONG_DWELL`).
       - Else $\implies \text{mode} = 1$ (`NORMAL_DWELL`).
     - **Result**: Neither Mode 3 (`REVISIT`) nor Mode 4 (`PREEMPTIVE_INTERCEPT`) can ever be selected by `band_first_decoupled`. The policy cannot learn joint time-frequency optimization.

---

### Step 6: MoE Arbitration & Predictive Overrides
* **Files**:
  - `src/models/smartscan_moe.py`
  - `src/operational/receiver_controller.py`
* **Overrides in `SmartScanMoE.select_action`**:
  1. **Consecutive Empty Escape**: If 3 consecutive empty dwells occur anywhere, forces cognitive exploration across cold/uncertain bands.
  2. **Exploration Guard**: High-confidence impending arrivals override forced exploration.
  3. **T1 Predictive Reservation**: Active temporal reservations override both band and mode.
  4. **T0 Predictive Hop Intercept**: Imminent predicted hops override band and determine mode based on estimated time of arrival (ETA).
  5. **Preemptive Priority Intercept**: Periodic urgency $> 0.5$ forces band to $\arg\max(\text{periodic\_urgency})$.
  6. **DRQN Top-K Gating**: When confident, mixes Q-margin with occupancy and periodic vectors, sampled via Boltzmann temperature.
  7. **Occupancy Fallback**: Unconfident predictions fall back to heuristic occupancy argmax.

---

### Step 7: Environment Action & Receiver Tuning
* **Files**:
  - `src/environment/cognitive_rf_scan_env.py`
  - `src/receiver/sieve_receiver.py`
  - `src/contracts.py`
* **Functions**:
  - `CognitiveRFScanEnv.step(action, mode_context)`
  - `band_of_action(action, n_modes)`: $\text{band} = \text{action} // 5$
  - `mode_of_action(action, n_modes)`: $\text{mode} = \text{action} \% 5$
  - `SieveReceiver.tune(center_mhz)`
  - `SieveReceiver.set_dwell_time(dwell_us)`
* **Physical Dwell Parameters**:
  - Band center: $f_c = \text{freq\_min} + \frac{\text{freq\_max} - \text{freq\_min}}{36} \cdot (\text{band} + 0.5)$, clamped to $[250 \text{ MHz}, 17750 \text{ MHz}]$.
  - Instantaneous Bandwidth: $\text{IBW} = 500 \text{ MHz}$.
  - Dwell Duration:
    - Mode 0 (`SHORT_DWELL`): $0.25 \times 500\mu s = 125\mu s$
    - Mode 1 (`NORMAL_DWELL`): $1.00 \times 500\mu s = 500\mu s$
    - Mode 2 (`LONG_DWELL`): $2.50 \times 500\mu s = 1250\mu s$
    - Mode 3 (`REVISIT`): $1.00 \times 500\mu s = 500\mu s$ (+ sensitivity threshold boost up to 3 dB)
    - Mode 4 (`PREEMPTIVE_INTERCEPT`): $1.00 \times 500\mu s = 500\mu s$ (holds aperture window up to $1500\mu s$ for predicted arrival)

---

### Step 8: Hit / Miss Determination
* **Files**:
  - `src/environment/cognitive_rf_scan_env.py`
  - `src/environment/radio_environment.py`
  - `src/receiver/sieve_receiver.py`
* **Functions**:
  - `CognitiveRFScanEnv._advance_world_to(dwell_end)`
  - `SieveReceiver._detect_buffered_interval(dwell_start, dwell_end)`
* **Logic**:
  - The radio world clock advances to `dwell_end`. All pulse ENTRY events within $[t_{\text{start}}, t_{\text{end}}]$ are placed in the receiver buffer.
  - Sieve receiver checks frequency overlap ($|f_{\text{pulse}} - f_c| \le \text{IBW}/2$) and amplitude $\ge \text{detection\_threshold\_db}$.
  - Any valid pulse constitutes a hit: $\text{hit} = (\text{len}(\text{detections}) > 0)$.
  - EXIT events are deferred until after detection, guaranteeing causal fidelity.

---

### Step 9: Reward Contributions
* **Files**:
  - `src/training/reward.py`
  - `src/environment/cognitive_rf_scan_env.py`
* **Function**: `receiver_reward_components(...)`
* **Complete Inventory of Reward Terms**:
  1. **Hit Incentive** ($w_{\text{hit}} = +5.0$): True Positive detection in tuned band.
  2. **Novel Discovery Bonus** ($w_{\text{novel}} = +5.0$): First interception of an unseen emitter ID in this episode.
  3. **Decision Miss Penalty** ($w_{\text{miss}} = -0.5$): False Negative; active emitter was present in the tuned band but missed.
  4. **Missed Coverage Opportunity** ($w_{\text{missed\_coverage}} = -0.2$): Tuned an inactive band while emitters were active elsewhere in spectrum.
  5. **Early Latency Bonus** ($w_{\text{latency}} = +1.0$): $w_{\text{latency}} \cdot \exp(-t_{\text{hit}} / \tau_{\text{latency}})$ rewarding early pulse interception within dwell.
  6. **Prediction Bonus** ($w_{\text{prediction}} = +0.5$): Intercepting a predicted agile frequency hop.
  7. **Active Track Incentive** ($w_{\text{active\_track}} = +2.0$): Consecutively confirming active emitter tracks.
  8. **Timing Penalty** ($w_{\text{timing}} = -0.001$): Linear penalty per $\mu s$ of delay from dwell start.
  9. **Priority Gain** ($w_{\text{priority}} = +1.0$): Reward scaled by observable priority score of intercepted emitter.
  10. **Information Gain** ($w_{\text{information\_gain}} = +0.2$): Shannon entropy reduction $H(p)_{\text{before}} - H(p)_{\text{after}}$ of occupancy belief.
  11. **False Alarm Penalty** ($w_{\text{false\_alarm}} = -0.5$): Spurious detection on empty band.
  12. **Dwell Opportunity Cost** ($w_{\text{dwell\_cost}} = -0.0002/\mu s$): Dwell cost scaled by aperture duration (zero on hit, 10% on active track, full on empty dwell).
  13. **Staleness Bonus** ($w_{\text{staleness}} = +0.4$): Intrinsic bonus for exploring high revisit age bands.
  14. **Preemptive Delay Penalty** ($w_{\text{delay}} = -0.5$): Penalty for overdue periodic arrivals.
  15. **Redundant Scan Penalty** ($w_{\text{redundant\_scan}} = 0.0$): Disabled in canonical config.

---

### Step 10: Replay Storage
* **Files**:
  - `src/training/replay_buffer.py`
* **Function**: `SequenceReplayBuffer.add(...)`
* **Stored Transition**:
  $$(s_t, a_t, r_t, s_{t+1}, d_t, \text{hit\_prob}_t, \text{intercept\_time\_us}_t, \text{time\_target\_valid}_t)$$
* **Sequence Sampling**:
  - Samples contiguous sequences of length $\text{seq\_len} = 16$.
  - First $\text{burn\_in} = 8$ transitions warm up the LSTM hidden state and are masked out of loss.
  - Padded transitions are marked invalid via `valid_mask`.

---

### Step 11: DRQN Loss & Learning Update
* **Files**:
  - `src/training/train_scheduler.py`
* **Function**: `_do_drqn_update(...)`
* **Equations**:
  - Double-DQN Target:
    $$y_t = (r_t - \bar{r}) + \gamma Q_{\text{target}}\left(s_{t+1}, \arg\max_{a} Q_{\text{online}}(s_{t+1}, a)\right) (1 - d_t)$$
  - Primary Bellman Q Loss:
    $$\mathcal{L}_Q = \text{Huber}\left(Q_{\text{online}}(s_t, a_t), y_t\right) \quad \text{for } t \in [\text{burn\_in}, \text{seq\_len})$$
  - Auxiliary Losses:
    $$\mathcal{L}_{\text{prob}} = \text{BCE}(\hat{p}_{\text{hit}}, \mathbf{1}_{\text{hit}})$$
    $$\mathcal{L}_{\text{time}} = \text{Huber}\left(\frac{\hat{t}}{1000}, \frac{t_{\text{hit}}}{1000}, \delta=0.1\right) \quad \text{where } \text{time\_target\_valid} = 1$$
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_Q + 0.1 \cdot (\mathcal{L}_{\text{prob}} + \mathcal{L}_{\text{time}})$$
  - Gradient clipping: $\|\mathbf{g}\| \le 1.0$.

---

## 4. Inventory of Sources of Randomness

1. **Random Number Generator Seeds**:
   - `seed=42` sets `torch.manual_seed`, `np.random.seed`, and Python `random.seed`.
2. **Epsilon-Greedy Action Selection**:
   - Python `random.random() < eps` decides between exploration and greedy exploitation.
3. **Exploration Action Draws**:
   - `random.randint(0, n_bands - 1)` selects exploration band.
   - `np.random.choice([0, 1, 2], p=[0.10, 0.70, 0.20])` selects exploration dwell mode.
4. **Thompson Sampling Warmup**:
   - `np.random.beta(alpha, beta)` draws posterior occupancy probabilities.
5. **Boltzmann Temperature Sampling**:
   - In `band_first_decoupled`, `SmartScanMoE`, and `DRQNBaseline`, temperature $\tau > 0$ introduces softmax probability draws: $P(i) \propto \exp(Q_i / \tau)$ via `np.random.choice`.
6. **Scenario Sampling**:
   - `train_source.sample()` draws random scenario files from the dataset pool.
7. **Replay Buffer Sampling**:
   - Uniform random sequence start indices drawn across stored episodes.

---

## 5. Runtime Decision Telemetry Specification

To ensure a single scheduler decision can be traced from observation to receiver action without ambiguity, the following 13 runtime decision fields are integrated across the pipeline:

| Telemetry Field | Type | Description |
|---|---|---|
| `raw_drqn_action` | `int` | Flat action index ($0..179$) selected by pure $\arg\max_a Q(s, a)$ |
| `raw_drqn_band` | `int` | Frequency band index ($0..35$) of the raw DRQN argmax action |
| `raw_drqn_mode` | `int` | Dwell mode ($0..4$) of the raw DRQN argmax action |
| `final_action` | `int` | Actual action ($0..179$) passed to the receiver environment |
| `final_band` | `int` | Actual frequency band ($0..35$) tuned by receiver |
| `final_mode` | `int` | Actual dwell mode ($0..4$) executed by receiver |
| `action_was_overridden` | `bool` | `True` if `final_action != raw_drqn_action` |
| `override_source` | `str \| null` | Component responsible for override (`"band_first_decoupled"`, `"thompson"`, `"epsilon_random"`, `"moe_heuristic"`, etc.) |
| `exploration_source` | `str \| null` | Exploration mechanism active (`"none"`, `"thompson"`, `"epsilon_random"`, `"tau_boltzmann"`, etc.) |
| `q_selected` | `float` | Q-value of the selected final action under current policy |
| `q_max` | `float` | Maximum Q-value across all 180 actions: $\max_a Q(s, a)$ |
| `q_mean` | `float` | Mean Q-value across all 180 actions: $\frac{1}{180} \sum_a Q(s, a)$ |
| `q_std` | `float` | Standard deviation of Q-values across all 180 actions |

### Integration Verification
* Registered in `src/telemetry/schema.py` (`DECISION_TELEMETRY_FIELDS` and `make_decision_telemetry()`).
* Captured in `DRQNScheduler.act()` and accessible via `drqn.last_decision_telemetry`.
* Computed in `SmartScanMoE.select_action()` and embedded in the returned `attribution` dictionary.
* Emitted in `DRQNBaseline.act()` as part of baseline telemetry.
* Passed into `CognitiveRFScanEnv.step()` and persisted in step `info` dictionary.
* Validated by automated unit tests in `tests/test_decision_telemetry.py` (`4 passed in 1.90s`).

---

## 6. Audit Verdict & Path Forward

The decision path audit demonstrates that although the underlying DRQN architecture and perception pipeline are technically sound:
1. **The ML policy is not currently in direct control** during greedy inference due to the `band_first_decoupled` heuristic overrides.
2. **Exploration is severely restricted** (36 bands $\times$ 3 modes = 108 actions sampled out of 180).
3. **MoE arbitration can silently mask DRQN policy failures** during validation and operational evaluation.

In accordance with Phase 3 of the rescue plan, direct ML control will be restored via `action_selection_mode: flat_argmax`, directly binding the environment receiver action to $\arg\max Q[0:180]$ without heuristic interception.

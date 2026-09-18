# Deep Recurrent Q-Network (DRQN) Scheduler Architecture

## Overview

This document provides the complete scientific and engineering specification for the cognitive Electronic Support (ES) receiver scheduler developed for the DRDO Electronic Warfare (EW) Smart Scan Strategy problem statement (SIH 2026).

The scheduler operates as an autonomous cognitive controller that allocates receiver dwell time across a wide radio frequency (RF) spectrum divided into discrete frequency bands. The objective is to maximize the interception of hostile radar and communications signals—especially frequency-agile, frequency-hopping, and periodic scanning emitters—with **zero pre-mission intelligence** (no pre-programmed threat libraries, frequency lists, pulse repetition intervals, or antenna scan schedules).

---

## Section 1 — Algorithm: Deep Recurrent Q-Network (DRQN) with Dueling Architecture

### 1.1 Partially Observable Markov Decision Process (POMDP) Formulation

In Electronic Support operations, a single superheterodyne or digital receiver possesses a finite Instantaneous Bandwidth ($IBW = 500\,\text{MHz}$) covering only a fraction of the total surveillance spectrum ($18\,\text{GHz}$). When the receiver tunes to band $b_t$ during time slot $t$, it observes emissions within that band but remains blind to transmissions occurring simultaneously across the other $N-1$ bands.

Because the true underlying RF environment state $s_t \in \mathcal{S}$ is only partially observed through receiver dwell outcomes $o_t \in \mathcal{O}$, the scheduling challenge constitutes a **Partially Observable Markov Decision Process (POMDP)** defined by the 6-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{O}, \Omega)$:
- $\mathcal{S}$: The complete multi-band RF emitter transmission state across all frequency channels.
- $\mathcal{A}$: Discrete action space of joint band and dwell-mode selections ($|\mathcal{A}| = 180$).
- $\mathcal{T}(s_{t+1} \mid s_t, a_t)$: Transition dynamics governed by emitter scan periods, PRI schedules, and agility hop patterns.
- $\mathcal{R}(s_t, a_t)$: Reward earned from successful signal interceptions and information gain.
- $\mathcal{O}$: Observable belief features and receiver detection telemetry.
- $\Omega(o_t \mid s_t, a_t)$: Observation emission probability conditioned on receiver tuning.

Standard Markov Decision Process algorithms (such as tabular Q-learning or memoryless Deep Q-Networks) fail in POMDP environments due to catastrophic state aliasing: an identical observation vector at time $t$ may represent vastly different emitter phases depending on historical context.

### 1.2 Neural Network Architecture

To resolve partial observability, the scheduler implements a **Deep Recurrent Q-Network (DRQN)** integrating long short-term memory (LSTM) with a Dueling Q-network decomposition and band-local feature routing.

```
                  ┌─────────────────────────────────────────┐
                  │ Observation Vector (360-D Belief State) │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │         Input LayerNorm (360)           │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    2-Layer Recurrent LSTM (2 x 256)     │
                  │   Episodic Memory across Scan Dwells    │
                  └────────────┬────────────────┬───────────┘
                               │                │
            ┌──────────────────┴───┐        ┌───┴─────────────────┐
            │                      │        │                     │
            ▼                      ▼        ▼                     ▼
     ┌──────────────┐      ┌─────────────┐┌──────────────┐ ┌──────────────┐
     │ Value Stream │      │  Advantage  ││Aux Head:     │ │Aux Head:     │
     │     V(s)     │      │ Stream A(s) ││P(intercept)  │ │Expected Time │
     │ Linear(128)  │      │ Local Map   ││Sigmoid Head  │ │Softplus Head │
     │   ReLU()     │      │ Linear(128) │└──────────────┘ └──────────────┘
     │  Linear(1)   │      │ Linear(180) │
     └──────┬───────┘      └──────┬──────┘
            │                     │
            └───────────┬─────────┘
                        ▼
       ┌─────────────────────────────────┐
       │ Dueling Q-Aggregation:          │
       │ Q(s,a) = V(s) + A(s,a) - mean(A)│
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Action Selection: argmax Q(s,a) │
       │ (Band Index, Dwell Mode Index)  │
       └─────────────────────────────────┘
```

The core components of the network (`ew_core/models/drqn_scheduler.py`) are structured as follows:

1. **Recurrent Hidden Backbone ($2 \times 256$ LSTM)**:
   A 2-layer LSTM with 256 hidden units per layer maintains the episodic hidden state $h_t = (h_t^{(1)}, c_t^{(1)}, h_t^{(2)}, c_t^{(2)})$. The hidden state acts as an internal cognitive summary of past dwell outcomes, tracking latent emitter periods, hop set rotations, and unvisited band staleness.

2. **Dueling Q-Stream Decomposition**:
   To disentangle the intrinsic value of the current environment state from the specific advantage of dwelling on a particular band, the network decouples into two streams:
   $$\mathcal{Q}(s, a; \theta, \alpha, \beta) = \mathcal{V}(s; \theta, \beta) + \left( \mathcal{A}(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} \mathcal{A}(s, a'; \theta, \alpha) \right)$$
   - **State-Value Stream $\mathcal{V}(s)$**: A 2-layer MLP (`Linear(256, 128) -> ReLU -> Linear(128, 1)`) estimating scalar state value $V(s)$.
   - **Advantage Stream $\mathcal{A}(s, a)$**: Evaluates the relative benefit of taking action $a$ over alternative actions.

3. **Band-Local Advantage Feature Routing**:
   To prevent interference across unrelated frequency channels, the advantage stream combines the global LSTM recurrent representation $h_t$ with band-local belief features through a dedicated 10-to-32 dimension linear projection, ensuring band-specific telemetry directly informs the advantage of selecting that band.

4. **Auxiliary Multi-Task Prediction Heads**:
   The shared recurrent backbone also drives two auxiliary supervised heads:
   - **Interception Probability Head ($\hat{p}_{\text{intercept}}$)**: Emits a sigmoid-activated probability $\in [0, 1]$ predicting whether the selected dwell will yield a hit.
   - **Interception Time Head ($\hat{\tau}_{\text{us}}$)**: Emits a softplus-activated scalar estimating the expected time-to-arrival ($\mu\text{s}$) for the emitter active in that band.
   Auxiliary multi-task loss terms regularize the recurrent representation and accelerate POMDP policy convergence.

### 1.3 Exploration Strategy: Thompson Sampling Warm-up to Epsilon-Greedy

At episode commencement, an $\epsilon$-greedy strategy with randomly initialized weights exhibits severe sample inefficiency due to the wide 180-action search space. To resolve this:

- **Thompson Sampling Warm-Up**: During the initial 5,000 steps of exploration, the scheduler maintains independent Beta priors $\text{Beta}(\alpha_b, \beta_b)$ over the transmission likelihood of each band $b$. Bands are sampled proportionally to posterior success probability, ensuring rapid discovery of active channels without wasting receiver dwell time on permanently silent frequencies.
- **Decayed $\epsilon$-Greedy Exploitation**: After warm-up, the agent transitions to DRQN-governed $\epsilon$-greedy action selection with exponential epsilon decay ($\epsilon_{\text{start}} = 1.0 \to \epsilon_{\text{end}} = 0.05$).
- **Replay Buffer**: Prioritized Experience Replay (PER) storing sequential trajectories of length $T=16$ with an 8-step burn-in window to prime the LSTM hidden state before backpropagation through time (BPTT).

---

## Section 2 — Electronic Warfare Problem Mapping

### 2.1 Hardware and Receiver Constraints

The operational parameters reflect tactical radar warning receiver (RWR) and electronic support measures (ESM) hardware configurations:

| Parameter | Value | Operational Significance |
| :--- | :--- | :--- |
| **RF Coverage Range** | $0.0\,\text{MHz} - 18,000.0\,\text{MHz}$ | Covers VHF, UHF, L, S, C, X, and Ku radar bands. |
| **Instantaneous Bandwidth (IBW)** | $500.0\,\text{MHz}$ | Hardware instantaneous receiver channel width. |
| **Number of Bands ($N_{\text{bands}}$)** | $36$ bands | $18,000\,\text{MHz} / 500\,\text{MHz} = 36$ contiguous frequency bins. |
| **Base Dwell Time ($\tau_{\text{base}}$)** | $500.0\,\mu\text{s}$ | Nominal time receiver local oscillator (LO) stays locked. |
| **Detection Threshold** | $-140.0\,\text{dBm}$ | Minimum detectable signal sensitivity ($S_{\text{min}}$). |

### 2.2 Dwell Mode Taxonomy

To balance fast wideband reconnaissance with deep fine-grain pulse train deinterleaving, the scheduler selects not only the target frequency band but also the **dwell mode**:

| Mode Index | Mode Name | Duration Multiplier | Dwell Duration ($\mu\text{s}$) | Tactical Operational Purpose |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `SHORT_DWELL` | $0.25\times$ | $125.0\,\mu\text{s}$ | **Fast Reconnaissance**: Rapid scan across multiple channels to confirm activity. |
| `1` | `NORMAL_DWELL` | $1.00\times$ | $500.0\,\mu\text{s}$ | **Surveillance**: Standard baseline dwell for general environmental monitoring. |
| `2` | `LONG_DWELL` | $2.50\times$ | $1,250.0\,\mu\text{s}$ | **Deep Observation**: Sustained dwell to intercept staggered or low-PRI radars. |
| `3` | `REVISIT` | $1.00\times$ | $500.0\,\mu\text{s}$ | **Track Maintenance**: Targeted return to confirm continuity on known emitters. |
| `4` | `PREEMPTIVE` | $1.00\times$ | $500.0\,\mu\text{s}$ | **Phase-Locked Intercept**: Dwell timed to meet an incoming periodic radar scan. |

### 2.3 Joint Action Space Definition

The action space is the Cartesian product of frequency bands and dwell modes:
$$|\mathcal{A}| = N_{\text{bands}} \times N_{\text{modes}} = 36 \times 5 = 180$$

Action encoding and decoding follow deterministic canonical mapping (`ew_core/contracts.py`):
$$\text{action} = b \cdot N_{\text{modes}} + m$$
$$b = \lfloor \text{action} / N_{\text{modes}} \rfloor, \quad m = \text{action} \pmod{N_{\text{modes}}}$$

---

## Section 3 — State Vector Definition (360-D Observation)

The observation vector provided to the scheduler at each decision step has a fixed dimension of **360 features**, formed by concatenating 10 continuous belief features for each of the 36 frequency bands:
$$\mathbf{x}_t = \big[ \mathbf{f}_t^{(0)}, \mathbf{f}_t^{(1)}, \dots, \mathbf{f}_t^{(35)} \big] \in \mathbb{R}^{360}$$

### Per-Band Feature Definition ($f_t^{(b)} \in \mathbb{R}^{10}$)

| Index | Feature Key | Mathematical Definition | Physical Description |
| :---: | :--- | :--- | :--- |
| `0` | `occupancy` | $p_t^{(b)} = \alpha \cdot \mathbb{I}(\text{hit}) + (1-\alpha) p_{t-1}^{(b)}$ | Exponentially weighted moving average (EWMA) of band activity ($\alpha = 0.3$). |
| `1` | `det_rate` | $N_{\text{hit}}^{(b)} / \max(1, N_{\text{dwell}}^{(b)})$ | Historical empirical detection probability when receiver dwelt on band $b$. |
| `2` | `miss_rate` | $N_{\text{miss}}^{(b)} / \max(1, N_{\text{dwell}}^{(b)})$ | Historical miss rate when receiver was tuned to band $b$. |
| `3` | `uncertainty` | $\mathbb{V}[\text{Beta}(\alpha_b, \beta_b)]$ | Bayesian variance of transmission belief; highest for unobserved bands. |
| `4` | `revisit_age` | $\min(1.0, (t - t_{\text{last}}^{(b)}) / T_{\text{horizon}})$ | Normalized elapsed time since receiver last tuned to band $b$. Prevents starvation. |
| `5` | `emitter_count`| $\min(1.0, N_{\text{tracks}}^{(b)} / 10.0)$ | Number of distinct deinterleaved emitter tracks confirmed in band $b$. |
| `6` | `deint_conf` | $\frac{1}{K} \sum \text{conf}_k^{(b)}$ | Mean silhouette / clustering confidence score from the deinterleaver on band $b$. |
| `7` | `pri_stability`| $1.0 - \min(1.0, \text{jitter}^{(b)} / 50.0)$ | Regularity score of the intercepted pulse train ($1.0 = \text{fixed PRI}, 0.0 = \text{agile/jitter}$). |
| `8` | `agility` | $H(\text{freq\_transitions})$ | Normalized Shannon entropy of observed frequency hopping transitions. |
| `9` | `priority` | $\max_k \text{threat\_level}_k^{(b)}$ | Highest tactical threat weight assigned to identified emitter types in band $b$. |

---

## Section 4 — Reward Function Definition

The reward function directly aligns policy optimization with the DRDO Figures of Merit: Probability of Detection ($P_d$), Intercept Rate, and Intercept Time Error.

$$\mathcal{R}(s_t, a_t) = R_{\text{intercept}} + R_{\text{agility}} + R_{\text{revisit}} - C_{\text{dwell}} - C_{\text{miss}}$$

1. **Base Intercept Reward ($R_{\text{intercept}}$)**:
   $$R_{\text{intercept}} = \begin{cases} +1.0 & \text{if receiver intercepted an active emitter pulse train} \\ 0.0 & \text{otherwise} \end{cases}$$
   Directly maximizes Probability of Detection ($P_d$) and average intercept rate.

2. **Agility Bonus ($R_{\text{agility}}$)**:
   $$R_{\text{agility}} = \beta_{\text{agile}} \cdot \text{agility}_t^{(b)} \cdot \mathbb{I}(\text{hit})$$
   Rewards successful interceptions of difficult frequency-hopping and agile emitters ($\beta_{\text{agile}} = 0.5$).

3. **Revisit Starvation Penalty / Urgency Reward ($R_{\text{revisit}}$)**:
   $$R_{\text{revisit}} = \gamma_{\text{urgency}} \cdot \text{revisit\_age}_t^{(b)}$$
   Incentivizes sweeping cold bands to discover newly activated hostile radars before they illuminate the platform.

4. **Dwell Duration Cost ($C_{\text{dwell}}$)**:
   $$C_{\text{dwell}} = \lambda_{\text{cost}} \cdot \frac{\tau_{\text{dwell}}}{\tau_{\text{base}}}$$
   Penalizes wasteful `LONG_DWELL` selections on silent channels, forcing the scheduler to prefer fast reconnaissance (`SHORT_DWELL`) when surveying uncertain spectrum.

5. **Miss Penalty ($C_{\text{miss}}$)**:
   $$C_{\text{miss}} = \begin{cases} -0.2 & \text{if tuned band was silent while active emitters transmitted elsewhere} \\ 0.0 & \text{otherwise} \end{cases}$$

---

## Section 5 — No Prior Intelligence Constraint

The DRDO problem statement strictly mandates that the receiver operate without pre-programmed emitter characteristics. The scheduler adheres to this requirement through a three-stage cold-start process:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Stage 1: Cold Start Bootstrapping               │
│  - Zero pre-mission emitter library or frequency priors.               │
│  - Thompson Sampling warm-up with uniform Beta(1,1) priors over bands. │
│  - Wideband surveillance sweeps using SHORT_DWELL mode.                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Stage 2: Empirical Feature Extraction                │
│  - Dwell hits processed by WindowedDeinterleaver & Hungarian tracker.  │
│  - Difference-vector PRIEstimator measures pulse arrival intervals.   │
│  - Online Bayesian belief state updates occupancy and agility features.│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Stage 3: Predictive Phase-Lock & Preemption               │
│  - PeriodicScanDetector identifies beam rotation periods (T_scan).     │
│  - DRQN schedules PREEMPTIVE dwells timed to incoming radar beam.      │
│  - Average intercept time error decreases by >= 50% over episode.      │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Cold Start Bootstrapping ($t = 0$)**:
   The scheduler initializes with flat uniform prior distributions across all 36 channels ($\alpha_b = 1, \beta_b = 1$). No emitter frequencies, scan intervals, or radar modes are hard-coded.
2. **Online Empirical Discovery**:
   As pulses are intercepted, the `WindowedDeinterleaver` extracts Pulse Descriptor Words (PDWs), associates cluster tracks, and estimates fundamental pulse repetition intervals using the `PRIEstimator`. Emitter agility and occupancy features update recursively via streaming Bayesian rules.
3. **Phase-Locking and Preemptive Scheduling**:
   Once the `PeriodicScanDetector` records repeated beam passages from scanning emitters, it estimates the scan period $T_{\text{scan}}$ and confidence. The scheduler preemptively commands dwells on that band at predicted time $t_{\text{next}} = t_{\text{last}} + T_{\text{scan}}$, converting a reactive search into a predictive intercept strategy with zero ground-truth leakage.

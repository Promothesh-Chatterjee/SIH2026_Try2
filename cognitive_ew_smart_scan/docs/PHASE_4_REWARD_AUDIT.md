# Phase 4A: Comprehensive Reward Pipeline & Objective Dilution Audit

## 1. Executive Summary
This document provides an exhaustive audit of the reward pipeline for the Cognitive EW Smart Scan scheduler. We trace the complete runtime path from raw TSRD pulse stream ground truth to receiver observations, belief formation, DRQN action execution, reward signal formulation, replay buffer storage, and loss backpropagation.

We systematically examine all 15 active reward components in `src/training/reward.py` and uncover how secondary shaping terms (unconditional staleness bonuses, information gain, and unnormalized timing penalties) dilute and distort the primary operational objective: **maximizing intercept rate and minimizing intercept latency without ground-truth leakage**.

---

## 2. Complete Runtime Signal Trace

```mermaid
flowchart TD
    A[TSRD Dataset: HDF5 Pulse Stream] -->|Latent Truth: TOA, CF, PW, AoA, Emitter ID| B[RadioEnvironment & SieveReceiver]
    B -->|Filter: Band [f_c - IBW/2, f_c + IBW/2], Dwell Window| C[ReceiverObservation: Detections & PDWs]
    C -->|Strip Emitter ID & Metadata| D[Perception: Deinterleaver & EmitterTracker]
    D -->|Track Features: PRI stability, Agility, Bearing| E[BeliefState: 36 bands x 10 features = 360D]
    E -->|Normalized Observable Features in [0, 1]| F[DRQN Policy Network: LSTM + Dueling Streams]
    F -->|Q-values: Shape [180]| G{Action Selection}
    G -->|Exploitation: flat_argmax| H[Action in [0, 179]]
    G -->|Exploration: Uniform epsilon| H
    H -->|Decode: band = a // 5, mode = a % 5| I[SieveReceiver: Dwell Execution]
    I -->|Pulses in Window?| J{Hit or Miss?}
    J -->|Yes| K[Hit: first_detect_toa - dwell_start]
    J -->|No| L[Miss: Empty Dwell or Opportunity Loss]
    K --> M[Reward Computation: receiver_reward_components]
    L --> M
    M -->|Total Scalar Reward & Components Breakdown| N[SequenceReplayBuffer: obs, action, rew, next_obs, done]
    N -->|Sample Batch: 32 seqs of len 16| O[Double-DQN Target Computation & Huber Loss]
    O -->|Adam Optimizer Update| F
```

### Trace Step Details:
1. **TSRD Ground Truth**:
   The scenario generator loads HDF5 pulse records containing exact pulse arrival time (`toa_us`), carrier frequency (`frequency_mhz`), pulse width (`pw_us`), angle of arrival (`aoa_deg`), and ground-truth emitter label (`emitter_id`).
2. **Receiver Observation**:
   `SieveReceiver` filters incoming pulses based on the receiver's tuned center frequency and instantaneous bandwidth (IBW = 500 MHz), during dwell duration $T_{\text{dwell}} = 500\,\mu\text{s} \times \text{multiplier}$.
3. **Perception & Belief Formation**:
   Ground-truth emitter identities are strictly discarded. The neural deinterleaver and deterministic `EmitterTracker` cluster pulses by feature affinity, tracking PRI regularity, frequency dispersion (agility), and bearing. The 36-band $\times$ 10-feature belief state is updated.
4. **DRQN Action**:
   The DRQN receives belief observation $s \in \mathbb{R}^{360}$, passing it through LayerNorm, 2-layer LSTM (hidden=256), and Dueling value/advantage streams to produce Q-values $Q(s, a) \in \mathbb{R}^{180}$.
5. **Receiver Execution**:
   The receiver tunes to $\text{band} = a // 5$ with dwell multiplier $\text{multiplier}_m$ for $\text{mode} = a \% 5$.
6. **Hit/Miss Determination**:
   `any_hit` is true if one or more valid pulses were captured. `intercept_time_error_us` measures $t_{\text{first\_pulse}} - t_{\text{dwell\_start}}$.
7. **Reward Formulation**:
   `receiver_reward_components(...)` calculates the scalar reward and records component breakdowns in `FiguresOfMerit`.
8. **Replay Transition Storage**:
   `buffer.add(...)` stores the executed `action`, transition reward, next observation, done flag, hit probability target, and intercept time target.
9. **Double-DQN Loss**:
   Target network calculates $y = (r - \bar{r}) + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{online}}(s', a'))$, and Huber loss updates the online network parameters.

---

## 3. Exhaustive Inventory of Legacy Reward Components

The legacy implementation in `src/training/reward.py` (`receiver_reward_components`) evaluates up to 15 distinct additive components:

$$R_{\text{total}} = R_{\text{hit}} + R_{\text{novel}} + R_{\text{timing}} + R_{\text{priority}} + R_{\text{info\_gain}} + R_{\text{staleness}} + R_{\text{false\_alarm}} + R_{\text{dwell\_cost}} + R_{\text{redundant}} + R_{\text{miss}} + R_{\text{missed\_coverage}} + R_{\text{delay}} + R_{\text{active\_track}} + R_{\text{latency}} + R_{\text{prediction}}$$

### Detailed Component Table:

| Component | Variable Name | Default Weight / Formula | Conditions & Activation | Numerical Range | Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Hit** | `hit_term` | `w_hit = 5.0` | Active when `selected_active && detected` (True Positive) | $[0.0, +5.0]$ | Positive Primary |
| **Novel Discovery** | `novel_term` | `w_novel = 5.0` | Active when newly intercepted emitter ID detected | $[0.0, +5.0]$ | Positive Secondary |
| **Timing Penalty** | `timing_penalty` | $-w_{\text{timing}} \cdot \Delta t$ (`w_timing = 0.001`) | On hit: $\Delta t = t_{\text{first}} - t_{\text{start}}$ in $\mu\text{s}$ | $[-1.25, 0.0]$ | Negative Shaping |
| **Priority Bonus** | `priority_term` | `w_priority * p_ref` (`w_priority = 1.0`) | On hit: scaled by priority reference $p_{\text{ref}} \in [0, 1]$ | $[0.0, +1.0]$ | Positive Secondary |
| **Active Track** | `active_track_term`| `w_active_track = 2.0` | On hit: when `age <= 2` and `occupancy >= 0.4` | $[0.0, +2.0]$ | Positive Shaping |
| **Latency Bonus** | `latency_bonus_term`| $w_{\text{lat}} \cdot e^{-t_{\text{hit}}/\tau}$ (`w_latency = 1.0, tau = 100`) | On hit: exponential decay with arrival delay | $[0.0, +1.0]$ | Positive Secondary |
| **Agile Prediction** | `prediction_bonus_term`| `w_prediction = 0.5` | On hit: when band matched temporal predictor | $[0.0, +0.5]$ | Positive Shaping |
| **Pulse Scale** | `pulse_bonus_term` | $w_{\text{pulse}} \cdot \min(n_{\text{hits}}-1, 4)$ (`w_pulse_scale = 0.0`) | On hit: multiple pulses captured | $[0.0, 0.0]$ | Inactive |
| **Staleness Bonus** | `staleness_bonus` | $w_{\text{stale}} \cdot \min(\text{age}/50, 1.0)$ (`w_staleness = 0.4`) | **Unconditional**: awarded on EVERY dwell, even empty/miss! | $[0.0, +0.4]$ | **Perverse Shaping** |
| **Information Gain** | `info_gain_term` | $w_{\text{ig}} \cdot (H_{\text{before}} - H_{\text{after}})$ (`w_information_gain = 0.2`)| **Unconditional**: awarded for belief entropy reduction | $[0.0, +0.2]$ | Positive Shaping |
| **Decision Miss** | `miss_penalty` | $w_{\text{miss}} \cdot \max(0.2, p_{\text{occ}})$ (`w_miss = -0.5`) | False Negative: band was active but receiver missed | $[-0.5, -0.1]$ | Negative Primary |
| **Missed Coverage** | `missed_coverage_pen`| `w_missed_coverage = -0.2` | When tuned band is empty but other bands active | $[-0.2, 0.0]$ | Negative Shaping |
| **False Alarm** | `false_alarm_pen` | $w_{\text{false\_alarm}} \cdot 1.5$ (`w_false_alarm = -0.5`) | False Positive: detection declared on empty band | $[-0.75, 0.0]$ | Negative Primary |
| **Redundant Revisit**| `redundant_pen` | `w_redundant_scan = 0.0` | Consecutive unconfirmed re-scan | $[0.0, 0.0]$ | Inactive |
| **Delay Penalty** | `delay_pen` | $-|w_{\text{delay}}| \cdot \text{urgency}$ (`w_delay = -0.5`) | On hit: when overdue periodic urgency $> 0.3$ | $[-0.5, 0.0]$ | Negative Shaping |
| **Dwell Cost** | `dwell_cost` | $w_{\text{cost}} \cdot T_{\text{dwell}}$ (`w_dwell_cost = -0.0002`)| Conditional: 0 on hit, $0.1\times$ on track, full on empty | $[-0.25, 0.0]$ | Negative Resource |

---

## 4. Phase 4B: Identification of Objective Dilution

### 4.1 Root Causes of Policy Distraction
1. **Unconditional Staleness Parasite**:
   $R_{\text{staleness}} = +0.40 \cdot \min(\text{age}/50, 1.0)$ is paid out on *every single dwell*, regardless of whether an emitter was present, intercepted, or missed. If the scheduler sweeps through empty cold bands, it collects $+0.40$ staleness bonus while paying only $-0.10$ dwell cost and $-0.20$ missed coverage penalty $\implies \mathbf{+0.10}$ **net positive reward for dwelling on complete silence**.
2. **Trivial Miss Penalty vs. Substantial Hit Bonus**:
   `w_miss = -0.5` scaled by `max(0.2, occ)` means a missed emitter in a low-occupancy band is penalized by only $-0.10$. Missing an agile emitter is practically free.
3. **Competing Timing Signals**:
   The agent is penalized $-0.001 \cdot \Delta t$ linearly (`timing_penalty`), while simultaneously rewarded $+1.0 \cdot e^{-\Delta t / 100}$ exponentially (`latency_bonus_term`). On a $1250\,\mu\text{s}$ LONG_DWELL, a valid hit at $t = 1000\,\mu\text{s}$ suffers $-1.00$ penalty, wiping out priority and prediction bonuses.
4. **Information Gain rewarded on Inactivity**:
   Entropy reduction on cold bands rewards the network for checking empty space to confirm nothing is there, distracting it from tracking frequency-hopping agility.

---

## 5. Representative Numerical Case Studies (Legacy Reward vs Target)

| Scenario | Event Description | Legacy Reward Breakdown | Legacy Total | Desired Primary Signal |
| :--- | :--- | :--- | :--- | :--- |
| **A. Immediate First Intercept** | Novel emitter intercepted at $t=10\,\mu\text{s}$, NORMAL dwell ($500\,\mu\text{s}$) | Hit(+5.0), Novel(+5.0), Latency(+0.90), Timing(-0.01), Priority(+0.5), Stale(+0.4) | **+11.79** | Dominant positive ($\ge +10.0$) |
| **B. Late Intercept** | Known emitter intercepted at $t=450\,\mu\text{s}$, NORMAL dwell | Hit(+5.0), Latency(+0.01), Timing(-0.45), Priority(+0.5), Stale(+0.2) | **+5.26** | Moderate positive ($\approx +3.0$) |
| **C. Missed Active Emitter** | Band active ($p_{\text{occ}}=0.3$), receiver tuned but missed pulses | Miss(-0.15), MissedCov(-0.20), Dwell(-0.10), Stale(+0.20) | **-0.25** | Clear negative ($\le -4.0$) |
| **D. False Alarm (Spurious)**| Receiver declares hit on inactive band | FalseAlarm(-0.75), MissedCov(-0.20), Dwell(-0.10), Stale(+0.30) | **-0.75** | Clear negative ($\approx -1.0$) |
| **E. Redundant Empty Revisit**| Re-tuning empty band with age=0 | Dwell(-0.10), MissedCov(-0.20), Stale(0.0) | **-0.30** | Small negative ($\approx -0.25$) |
| **F. Empty LONG_DWELL** | Clean empty dwell ($1250\,\mu\text{s}$), cold band (age=50) | Dwell(-0.25), MissedCov(-0.20), Staleness(+0.40), InfoGain(+0.10) | **+0.05 (POSITIVE!)** | Strictly negative ($\approx -1.0$) |
| **G. Empty SHORT_DWELL** | Clean empty dwell ($125\,\mu\text{s}$) | Dwell(-0.025), MissedCov(-0.20), Staleness(+0.20) | **-0.025** | Slight negative ($\approx -0.5$) |
| **H. Agile Emitter Intercept**| Frequency-hopping pulse intercepted at $t=50\,\mu\text{s}$ | Hit(+5.0), Latency(+0.60), Timing(-0.05), Pred(+0.50), Stale(+0.20) | **+6.25** | High positive with agile bonus ($\ge +7.0$) |
| **I. Agile Emitter Miss** | Hopped band missed ($p_{\text{occ}}=0.5$) | Miss(-0.25), MissedCov(-0.20), Dwell(-0.10), Stale(+0.20) | **-0.35** | Significant negative ($\le -4.0$) |
| **J. Repeated Intercept** | Tracking known emitter, age=1, $t=50\,\mu\text{s}$ | Hit(+5.0), ActiveTrack(+2.0), Latency(+0.60), Priority(+0.5) | **+8.05** | Positive tracking reward ($\approx +4.0$) |

> **Critical Discovery in Case F**: In the legacy formulation, tuning to an empty band with a LONG_DWELL yields a **net positive reward (+0.05)** because the staleness bonus (+0.40) and information gain (+0.10) exceed the dwell cost (-0.25) and missed coverage (-0.20). This provides a perverse mathematical incentive for the policy to deliberately dwell on empty cold spectrum.

---

## 6. Specifications for `reward_v2` Rescue Function

To eliminate objective dilution while keeping full backwards compatibility for ablation, `reward_v2` must establish a strict 5-tier hierarchical reward structure:

1. **Successful Interception (Primary Signal)**:
   - Base Intercept: `+5.0`
   - Novel Emitter Bonus: `+5.0` (Total new intercept = `+10.0`)
   - Repeat Intercept (tracking known emitter): `+3.0`
2. **Interception Latency (Secondary Signal)**:
   - Normalized delay bonus $0.0 \le R_{\text{latency}} \le +5.0$:
     $$R_{\text{latency}} = 5.0 \cdot \max\left(0.0, 1.0 - \frac{t_{\text{hit}}}{T_{\text{dwell}}}\right)$$
     (Earlier arrival within the dwell receives higher bonus; linear and bounded).
   - Removed unnormalized $-w_{\text{timing}} \cdot \Delta t$.
3. **Frequency-Agile Interception Bonus**:
   - Intercepting confirmed agile/hopping emitter: `+2.0` shaping bonus.
4. **Missed Emitter Penalty**:
   - Clear, non-diluted penalty: `-4.0` (not reduced by occupancy or offset by staleness).
5. **False Alarm & Resource Costs**:
   - False alarm (empty band action / false detection): `-1.0`
   - Redundant empty revisit: `-0.25`
   - Normalized dwell cost: $-0.01 \cdot (\text{dwell\_us} / 500.0)$ (range $[-0.0025, -0.025]$)
   - **Zero unconditional staleness bonus**.
   - **Zero ungrounded information gain bonus**.

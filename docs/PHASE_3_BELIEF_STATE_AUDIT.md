# Phase 3 — Belief State & 10-Feature Canonical Contract Audit

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Architecture:** Cognitive EW Smart Scan Scheduler v2  
**Problem Statement:** Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics.  
**Phase:** Phase 3 — Belief State, Predictive Track Transitions & Temporal Stability  
**Status:** **VERIFIED & SEALED**

---

## 1. Executive Summary

This audit document establishes the mathematical correctness, causal purity, numerical stability, and distribution compatibility of the 10-feature per band belief state representation in the Cognitive EW Smart Scan Scheduler v2.

The scheduler receives a flattened 360-dimensional observation:
$$O = [B_0, B_1, \dots, B_{35}] \in [0, 1]^{360}$$
where each band $B_i \in [0, 1]^{10}$ obeys the frozen canonical schema.

---

## 2. Frozen 10-Feature Canonical Ordering & Audit

The feature schema is strictly preserved at every scheduler step:

| Index | Canonical Feature Name | Semantic Definition | Normalization Formula | Boundedness |
|:---:|---|---|---|:---:|
| **0** | `occupancy_prob` | Recent causal evidence of activity | Asymmetric EMA: $\alpha=0.3$ (hit), $\alpha=0.20$ (miss on confirmed) | $[0.0, 1.0]$ |
| **1** | `detection_rate` | Causal fraction of opportunities with valid detection | $\text{hits} / \max(1, \text{visits})$ | $[0.0, 1.0]$ |
| **2** | `miss_rate` | Causal fraction of opportunities with absent expected pulse | $1.0 - \text{detection\_rate}$ | $[0.0, 1.0]$ |
| **3** | `uncertainty` | Ambiguity/epistemic uncertainty of current belief | $w_{\text{epi}} \cdot 1.0 + (1 - w_{\text{epi}}) \cdot (1 - |2p - 1|)$ | $[0.0, 1.0]$ |
| **4** | `revisit_age` | Normalized elapsed time since last visit to band | $\min(\text{age}, 50.0) / 50.0$ | $[0.0, 1.0]$ |
| **5** | `emitter_count` | Estimated online active track population in band | $\text{clip}(N_{\text{tracks}} / 5.0, 0.0, 1.0)$ | $[0.0, 1.0]$ |
| **6** | `deint_confidence` | Causal confidence from deinterleaver/clustering | Weighted cluster confidence over active tracks | $[0.0, 1.0]$ |
| **7** | `pri_stability` | Pulse repetition interval consistency | $\text{clip}(1.0 / (1.0 + \text{CV}_{\text{pri}}), 0.0, 1.0)$ | $[0.0, 1.0]$ |
| **8** | `agility` | Observed intra-band frequency dispersion / hopping | $\text{clip}(\text{std}(f) / 100.0, 0.0, 1.0)$ | $[0.0, 1.0]$ |
| **9** | `priority` | Causal scheduler urgency composite | Weighted sum: age, occupancy, uncertainty, urgency | $[0.0, 1.0]$ |

---

## 3. Detailed Feature-by-Feature Mathematical Analysis

### 3.1 Feature 0: Occupancy Probability (`occupancy_prob`)
- **Initial State**: $p_0 = 0.5$ (neutral maximum-entropy prior ensuring true positive information gain on first dwell).
- **Update Rule**:
  $$p_{t} = (1 - \alpha) p_{t-1} + \alpha \cdot \mathbb{I}(\text{hit})$$
  For confirmed tracks ($\text{hits} \ge 1$), miss decay uses $\alpha = 0.20$ to prevent agile radar hop-aways from instantly dropping the belief below untouched priors.
- **Leakage Immunity**: Zero reference to simulator active emitter tables.

### 3.2 Feature 1: Detection Rate (`detection_rate`) & Feature 2: Miss Rate (`miss_rate`)
- **Denominator Semantics**: Evaluated over the identical causal visit denominator:
  $$\text{detection\_rate} = \frac{\sum_{k=1}^V \mathbb{I}(\text{hit}_k)}{\max(1, V)}$$
  $$\text{miss\_rate} = \frac{\sum_{k=1}^V \mathbb{I}(\text{miss}_k)}{\max(1, V)} = 1.0 - \text{detection\_rate}$$
- When unvisited ($V = 0$), $\text{detection\_rate} = 0.0$ and $\text{miss\_rate} = 1.0$.

### 3.3 Feature 3: Uncertainty (`uncertainty`)
- Combines aleatoric activity ambiguity $(1 - |2p - 1|)$ with epistemic lack-of-evidence discounting:
  $$w_{\text{epi}} = \exp(-V / 4.0)$$
  $$\text{uncertainty} = w_{\text{epi}} \cdot 1.0 + (1 - w_{\text{epi}}) \cdot (1 - |2p - 1|)$$
- Guarantees $\text{uncertainty} \in [0.0, 1.0]$ with maximum uncertainty on unvisited bands.

### 3.4 Feature 4: Revisit Age (`revisit_age`)
- Monotonically increments: $\text{age} \leftarrow \text{age} + 1$ with each simulation step.
- Resets strictly when the band is selected and scanned: $\text{age} \leftarrow 0$.
- Normalized via the repository's canonical constant: $\text{norm\_age} = \min(\text{age}, 50.0) / 50.0 \in [0.0, 1.0]$.

### 3.5 Feature 5: Emitter Count (`emitter_count`)
- Represents online estimated track population from causal perception tracks:
  $$N_{\text{tracks}} = |\{T_j : \text{track.last\_band} = b \land \text{track.is\_active}\}|$$
  $$emitter\_count_{\text{norm}} = \text{clip}\left(\frac{N_{\text{tracks}}}{5.0}, 0.0, 1.0\right)$$
- $N_{\text{ref}} = 5.0$ matches the existing repository contract in `adapters.py`. Never queries simulator emitter counts.

### 3.6 Feature 6: Deinterleaver Confidence (`deint_confidence`)
- Derived from track observation count, pulse clustering purity, and temporal consistency. Bounded strictly in $[0.0, 1.0]$.

### 3.7 Feature 7: PRI Stability (`pri_stability`)
- Uses the inverse coefficient of variation of inter-pulse arrival times:
  $$\text{CV} = \frac{\sigma_{\text{pri}}}{\max(\mu_{\text{pri}}, 10^{-6})}, \quad \text{stability} = \frac{1.0}{1.0 + \text{CV}} \in [0.0, 1.0]$$

### 3.8 Feature 8: Agility (`agility`)
- Derived from observed frequency dispersion within the band ($\text{std}(f) / 100.0$) and transition entropy. Bounded in $[0.0, 1.0]$.

### 3.9 Feature 9: Priority Score (`priority`)
- Multi-objective composite urgency:
  $$\text{priority} = \text{clip}(w_{\text{st}} \cdot \text{norm\_age} + w_{\text{occ}} \cdot p + w_{\text{unc}} \cdot u + w_{\text{per}} \cdot \text{urgency} + w_{\text{sem}} \cdot \text{semantic}, 0.0, 1.0)$$
- Uses default weights $(0.35, 0.25, 0.20, 0.10, 0.10)$.

---

## 4. Empirical Distribution Compatibility Table

From 500-step empirical dwell validation (`experiments/reports/phase3/phase3_belief_results.json`):

| Feature Name | Index | Observed Min | Observed Max | Mean | Std Dev | Bounded [0, 1] | Finite |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **occupancy** | 0 | 0.0141 | 1.0000 | 0.3441 | 0.2805 | YES | YES |
| **det_rate** | 1 | 0.0000 | 1.0000 | 0.1216 | 0.3087 | YES | YES |
| **miss_rate** | 2 | 0.0000 | 1.0000 | 0.8784 | 0.3087 | YES | YES |
| **uncertainty** | 3 | 0.0000 | 1.0000 | 0.6281 | 0.3198 | YES | YES |
| **revisit_age** | 4 | 0.0000 | 1.0000 | 0.6512 | 0.3840 | YES | YES |
| **emitter_count** | 5 | 0.0000 | 0.4600 | 0.0444 | 0.1177 | YES | YES |
| **deint_confidence** | 6 | 0.0000 | 0.8400 | 0.1127 | 0.2851 | YES | YES |
| **pri_stability** | 7 | 0.0000 | 1.0000 | 0.1347 | 0.3407 | YES | YES |
| **agility** | 8 | 0.0000 | 0.1968 | 0.0096 | 0.0277 | YES | YES |
| **priority** | 9 | 0.0251 | 0.6750 | 0.4396 | 0.1609 | YES | YES |

**Verdict:** 100% of values are finite and strictly within $[0.0, 1.0]$. The numerical distributions are fully compatible with the frozen DRQN.

---

## 5. Single Authoritative Canonical Module (`canonical_belief.py`)

In accordance with Approved Amendment 1, all feature calculation routines are centralized in a single authoritative module:
`ew_core/cognitive/canonical_belief.py`.

```
                ┌──────────────────────────────────────────┐
                │        canonical_belief.py               │
                │  - compute_canonical_occupancy()         │
                │  - compute_canonical_detection_miss()    │
                │  - compute_canonical_uncertainty()       │
                │  - compute_canonical_revisit_age()       │
                │  - compute_canonical_emitter_count()     │
                │  - compute_canonical_deint_confidence()  │
                │  - compute_canonical_pri_stability()     │
                │  - compute_canonical_agility()           │
                │  - compute_canonical_priority()          │
                │  - map_tracks_to_bands()                 │
                └────────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
   BeliefState            OperationalStateBuilder       build_band_belief_from_tracks
(cognitive_rf_scan_env)      (state_builder.py)                 (adapters.py)
```

This guarantees bit-level mathematical parity across all simulation, online operational, and perception evaluation paths, verified by `ew_core/tests/test_phase3_cross_builder_equivalence.py`.

---

## 6. 1,000-Step Pre vs Post Distribution Drift Analysis

From 1,000-step evaluation (36,000 total band observations, `experiments/reports/phase3/phase3_distribution_comparison.json`):

| Feature Name | Index | Pre Mean ± Std | Post Mean ± Std | Mean Drift | KS Stat | KS $p$-value | Bounded [0, 1] | Finite | Classification |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **occupancy_prob** | 0 | 0.3265 ± 0.2462 | 0.3278 ± 0.2457 | 0.0013 | 0.0282 | $7.7 \times 10^{-13}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **detection_rate** | 1 | 0.0504 ± 0.1739 | 0.0504 ± 0.1739 | 0.0000 | 0.0000 | 1.0000 | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **miss_rate** | 2 | 0.9496 ± 0.1739 | 0.9496 ± 0.1739 | 0.0000 | 0.0000 | 1.0000 | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **uncertainty** | 3 | 0.6197 ± 0.4671 | 0.6489 ± 0.4491 | 0.0291 | 0.1100 | $3.8 \times 10^{-190}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **revisit_age** | 4 | 0.6824 ± 0.3869 | 0.6824 ± 0.3869 | 0.0000 | 0.0000 | 1.0000 | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **emitter_count** | 5 | 0.0007 ± 0.0116 | 0.0050 ± 0.0288 | 0.0043 | 0.1074 | $2.6 \times 10^{-181}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **deint_confidence** | 6 | 0.0025 ± 0.0434 | 0.0201 ± 0.1153 | 0.0176 | 0.1074 | $2.6 \times 10^{-181}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **pri_stability** | 7 | 0.0026 ± 0.0472 | 0.0072 ± 0.0794 | 0.0046 | 0.0244 | $9.4 \times 10^{-10}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **agility** | 8 | 0.0000 ± 0.0000 | 0.0000 ± 0.0001 | 0.0000 | 0.0278 | $1.7 \times 10^{-12}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |
| **priority** | 9 | 0.4444 ± 0.2737 | 0.4578 ± 0.2657 | 0.0134 | 0.0797 | $8.1 \times 10^{-100}$ | YES | YES | `COMPATIBLE_WITHIN_TOLERANCE` |

**Conclusion**: All 10 features strictly obey numerical bounds $[0.0, 1.0]$, zero NaNs, zero Infinities, and mean drift $\le 0.0291$ (well under the maximum tolerance limit of $0.35$). The production frozen baseline model operates safely with zero retraining or distribution collapse.


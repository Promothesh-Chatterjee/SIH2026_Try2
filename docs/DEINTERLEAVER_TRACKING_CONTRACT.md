# Deinterleaver & Tracking Contract (Phase 2 Frozen Specification)

## 1. System Scope & Guiding Principles

This document defines the binding contractual interface between the RF receiver perception pipeline, the windowed deinterleaver, the cross-window reconciler, and the downstream emitter tracker in the **Cognitive EW Smart Scan Scheduler v2**.

### 1.1 Non-Negotiable Contract Invariants
1. **Zero Ground-Truth Leakage**: The scheduler-facing state, `TemporalPredictor`, and `BeliefState` must receive identity labels solely produced by `EmitterTracker.track_id`. Ground-truth simulator identifiers (`emitter_id`) are forbidden from scheduler observations and predictive state.
2. **Strict Identity Hierarchy**:
   $$\text{Local Cluster Label} \xrightarrow{\text{Cross-Window Matching}} \text{Reconciled Cluster ID} \xrightarrow{\text{Temporal/Spatial Association}} \text{EmitterTracker.track_id}$$
   Intermediate labels must never be named `emitter_id` to prevent semantic confusion.
3. **4-Feature Representation Invariance**: The input embedding to the deinterleaver clustering stage consists strictly of 4 normalized dimensions:
   $$\mathbf{x} = [f_{\text{norm}}, \text{pw}_{\text{norm}}, \text{amp}_{\text{norm}}, \Delta\text{toa}_{\text{norm}}] \in [0, 1]^4$$
   Circular spatial Angle of Arrival (AoA) is decoupled from the 4D deinterleaver embedding to preserve backward compatibility with pretrained clustering checkpoints, and is evaluated downstream during cross-window tracking association.
4. **Action & Observation Spaces**:
   - Dwell Action: 180 discrete actions ($36 \text{ frequency bands} \times 5 \text{ dwell modes}$).
   - Scheduler Observation: 360-dimensional vector ($10 \text{ features} \times 36 \text{ frequency bands}$).
5. **Frozen Production Model**:
   - Checkpoint: `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
   - SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`

---

## 2. Standardized Pulse Descriptor Word (PDW)

The perception pipeline operates on pulses represented by `PulseDescriptorWord`:

```python
@dataclass
class PulseDescriptorWord:
    frequency_mhz: float
    toa_ns: float
    pw_ns: float
    amplitude_dbm: float
    aoa_deg: float = 0.0
    phase_deg: float = 0.0
    emitter_id: Optional[int] = None  # SIMULATOR AUDIT ONLY: Never leaked to scheduler
```

### Mandatory Invariants:
- `frequency_mhz`: Carrier center frequency in MHz ($[2000.0, 18000.0]$).
- `toa_ns`: Time-of-Arrival in nanoseconds, strictly causal within dwell window ($t \ge 0$).
- `pw_ns`: Pulse width in nanoseconds ($pw > 0$).
- `amplitude_dbm`: Received pulse signal power in dBm ($[-150.0, 0.0]$).
- `aoa_deg`: Received angle of arrival in degrees ($[-180.0, 180.0]$ or $[0.0, 360.0]$).
- `emitter_id`: Retained strictly for offline benchmark scoring; property `true_emitter_id` aliases it for verification assertions.

---

## 3. Feature Extraction Contract (`PDWFeatureExtractor`)

### 3.1 Feature Vector Formulation
For each PDW $i$, the 4-dimensional normalized vector is defined by:
1. **Frequency**:
   $$f_{\text{norm}} = \frac{f_i - f_{\min}}{f_{\max} - f_{\min}}, \quad f_{\min}=2000.0\text{ MHz}, f_{\max}=18000.0\text{ MHz}$$
2. **Pulse Width**:
   $$\text{pw}_{\text{norm}} = \frac{\text{pw}_i - \text{pw}_{\min}}{\text{pw}_{\max} - \text{pw}_{\min}}, \quad \text{pw}_{\min}=50.0\text{ ns}, \text{pw}_{\max}=200000.0\text{ ns}$$
3. **Amplitude**:
   $$\text{amp}_{\text{norm}} = \frac{\text{amp}_i - \text{amp}_{\min}}{\text{amp}_{\max} - \text{amp}_{\min}}, \quad \text{amp}_{\min}=-150.0\text{ dBm}, \text{amp}_{\max}=0.0\text{ dBm}$$
4. **Delta-ToA (Inter-Arrival Time)**:
   $$\Delta\text{toa}_i = \begin{cases} \text{toa}_i - \text{toa}_{i-1}, & i > 0 \\ \Delta\text{toa}_1, & i = 0 \end{cases}$$
   $$\Delta\text{toa}_{\text{norm}} = \frac{\Delta\text{toa}_i - \Delta\text{toa}_{\min}}{\Delta\text{toa}_{\max} - \Delta\text{toa}_{\min}}, \quad \Delta\text{toa}_{\min}=100.0\text{ ns}, \Delta\text{toa}_{\max}=10^7\text{ ns}$$

### 3.2 Standardization & Sanitization
All features are clipped to $[0.0, 1.0]$. Any `NaN` or `Inf` resulting from numerical edge cases is replaced using `np.nan_to_num(..., nan=0.0, posinf=1.0, neginf=0.0)`.

---

## 4. Clustering & Fallback Hierarchy

Clustering must handle dwell pulse counts ranging from 0 to $>10,000$ with deterministic fallback guarantees:

```
PDW Batch (N pulses)
       |
       v
 N == 0: Return empty labels []
 N == 1: Return single cluster [0]
 N in [2, min_cluster_size - 1]: Fallback to DBSCAN (eps=0.15, min_samples=2)
 N >= min_cluster_size:
       |
       +---> Primary: HDBSCAN (min_cluster_size=3, min_samples=2, allow_single_cluster=True)
       |        |
       |        +--> Success: Return HDBSCAN cluster labels
       |        |
       |        +--> Exception/Failure:
       |                 |
       |                 v
       +---------> Secondary: DBSCAN (eps=0.15, min_samples=2)
                        |
                        +--> Success: Return DBSCAN cluster labels
                        |
                        +--> Exception/Failure:
                                 |
                                 v
                         Tertiary: Deterministic Rule-Based Frequency Slicer
```

### Deterministic Noise Reclassification
If HDBSCAN or DBSCAN assigns all pulses to noise (`-1`) but $N \ge 2$, fallback clustering ensures that genuine signals in high-density or low-contrast conditions are not silently discarded.

---

## 5. Cross-Window Reconciliation (`CrossWindowReconciler`)

Because local cluster labels in each dwell window are arbitrary indices ($0, 1, 2, \dots$), `CrossWindowReconciler` maps local clusters to globally persistent `reconciled_cluster_id` integers across sequential dwell windows.

### 5.1 Centroid Distance Metric
Centroids are computed in normalized feature space $\bar{\mathbf{c}} = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$. The distance between current centroid $\mathbf{c}_{\text{curr}}$ and previous centroid $\mathbf{c}_{\text{prev}}$ is:
$$D(\mathbf{c}_{\text{curr}}, \mathbf{c}_{\text{prev}}) = \|\mathbf{c}_{\text{curr}} - \mathbf{c}_{\text{prev}}\|_2$$

### 5.2 Deterministic Bipartite Matching
Matches are solved using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) with a gating cutoff `similarity_threshold = 2.0`. Centroid key sorting is deterministic (`sorted(keys)`).

### 5.3 Miss-Pruning & Lifecycle Management
To prevent unbounded memory growth in long mission scenarios:
- Unmatched past centroids accumulate a miss counter: $\text{misses} \leftarrow \text{misses} + 1$.
- Centroids exceeding `max_misses = 10` consecutive windows without an association are purged.

---

## 6. Downstream Association & Persistent Tracking (`EmitterTracker`)

`EmitterTracker` consumes reconciled clusters/detections and maintains operational `Track` objects.

### 6.1 Multi-Attribute Association Score
For an active track $T_j$ and candidate detection $D_k$:
$$S(T_j, D_k) = w_{\text{freq}} S_{\text{freq}} + w_{\text{pw}} S_{\text{pw}} + w_{\text{pri}} S_{\text{pri}} + w_{\text{aoa}} S_{\text{aoa}}$$
where default weights sum to $1.0$.

### 6.2 Adaptive Multi-Band Agile Frequency Gating
To track frequency-agile and frequency-hopping radars across multiple bands while preventing false merges between distinct emitters:
1. **Prior Agility Verification**:
   $$\text{is\_track\_agile} = (\text{track.frequency\_hopping\_detected} \lor \text{track.agility\_score} > 0.2)$$
2. **Multi-Signal Continuity Guard**:
   A candidate jump outside the standard carrier gate $[f_{\text{low}}, f_{\text{high}}]$ or exceeding `max_band_jump` is admitted **only if supported by all independent physical continuity signals**:
   - **Spatial Bearing**: $|\theta_{\text{track}} - \theta_{\text{det}}| \le \text{agile\_hop\_aoa\_gate\_deg}$ ($12.0^\circ$).
   - **Pulse Duration**: $\frac{2}{3} \le \frac{\text{pw}_{\text{det}}}{\text{pw}_{\text{track}}} \le \frac{3}{2}$.
   - **PRI / Rhythm**: Inter-pulse timing aligns within $20\%$ of established track PRI.
3. **Agile Weight Redistribution**:
   When multi-band agility is validated, $S_{\text{freq}}$ is omitted from the association penalty, allowing the agile emitter to maintain a single continuous `track_id` across arbitrary RF hops.

---

## 7. Standard Evaluation Metrics

All benchmark scoring routines adhere to the following definitions:

1. **Clustering Purity**:
   $$\text{Purity}(Y, C) = \frac{1}{N} \sum_{k} \max_{j} |C_k \cap Y_j|$$
2. **False Merge Rate (FMR)**:
   Proportion of assigned clusters containing pulses from $\ge 2$ distinct ground-truth emitters where neither emitter contributes $\ge 90\%$ of the cluster's energy.
3. **False Split Rate (FSR)**:
   Proportion of ground-truth emitters that are fragmented into $\ge 2$ distinct clusters where no single cluster contains $\ge 80\%$ of the emitter's pulses.
4. **Noise Ratio**:
   $$\text{NR} = \frac{|\{i : \hat{y}_i = -1\}|}{N}$$
5. **Per-Emitter Recall**:
   $$\text{Recall}_j = \frac{\max_k |C_k \cap Y_j|}{|Y_j|}$$
6. **Track Continuity**:
   Percentage of consecutive window observations for an emitter assigned to the same persistent `track_id`.

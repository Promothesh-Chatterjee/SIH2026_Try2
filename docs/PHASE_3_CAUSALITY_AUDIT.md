# Phase 3 — Causality, Lifecycle & Anti-Leakage Audit

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Architecture:** Cognitive EW Smart Scan Scheduler v2  
**Problem Statement:** Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics.  
**Phase:** Phase 3 — Belief State, Predictive Track Transitions & Temporal Stability  
**Status:** **VERIFIED & SEALED**

---

## 1. Causal Execution Order

At every decision timestep $t$, the environment enforces a strictly monotonic causal pipeline:

```
Step t:
  1. Scheduler selects Action a_t = (band_t, mode_t) from Observation O_t
  2. Sieve Receiver retunes to center frequency f_center(band_t)
  3. RadioEnvironment advances simulation clock: [t_dwell_start, t_dwell_end]
     - ONLY pulses with ToA <= t_dwell_end are queued into the receiver
     - Pulses with ToA > t_dwell_end remain in the future queue
  4. Sieve Receiver detects pulses within the IBW (500 MHz) and sensitivity window
  5. Windowed Deinterleaver extracts 4D physical features: [f, pw, amp, dt]
     - Ground-truth emitter_id is stripped before clustering
  6. CrossWindowReconciler matches cluster centroids using Hungarian assignment
  7. EmitterTracker updates kinematic tracks:
     - Tracks with detections -> ACTIVE (consecutive_misses = 0)
     - Tracks without detections -> COASTING (consecutive_misses += 1)
     - Tracks exceeding max_misses -> RETIRED (pruned from active memory)
  8. TemporalPredictor & SpatialTracker update states from causal track_id
  9. BeliefState records visit:
     - Occupancy EMA updated
     - Revisit age updated (band_t reset to 0, other bands incremented)
     - Uncertainty and priority recomputed
 10. Observation Vector O_{t+1} (360-D) is constructed from BeliefState
 11. Reward R_t is computed (ground-truth evaluation only; does NOT feedback into O_{t+1})
```

---

## 2. Track Lifecycle State Machine Audit

From `experiments/reports/phase3/phase3_track_lifecycle.json`:

```
                 [ Dwell Detection ]
                         │
                         ▼
                  ┌──────────────┐
                  │    ACTIVE    │ ◄───────────────────────────┐
                  └──────┬───────┘                             │
                         │                                     │
                 [ Missed Dwell ]                     [ Re-Observation ]
                         │                                     │
                         ▼                                     │
                  ┌──────────────┐                             │
                  │   COASTING   │ ────────────────────────────┘
                  └──────┬───────┘
                         │
        [ Consecutive Misses >= max_misses ]
                         │
                         ▼
                  ┌──────────────┐
                  │   RETIRED    │ (Pruned from active memory)
                  └──────────────┘
```

### Empirical Verification:
- Track observed continuously maintains `ACTIVE` status.
- Track missed in 1 dwell transitions to `COASTING` without identity change.
- Track re-observed while `COASTING` returns to `ACTIVE` with the identical persistent `track_id`.
- Track missed for $\ge \text{max\_misses}$ (e.g. 5 dwells) transitions to `RETIRED` and is completely purged from `EmitterTracker.tracks`.
- Sequence verified: `["ACTIVE", "COASTING", "COASTING", "ACTIVE", "RETIRED"]`.

---

## 3. Ground-Truth Isolation Proofs

### 3.1 Identity Perturbation Invariance
`test_phase3_leakage_and_causality.py::TestPhase3LeakageAndCausality::test_pdw_emitter_id_perturbation_invariance` executes belief state generation on pulse sets where `emitter_id` is altered, removed, or randomized.
**Result:** Bitwise identical observation vectors ($O_1 \equiv O_2$).

### 3.2 Future-Step Event Isolation
`test_phase3_leakage_and_causality.py::TestPhase3LeakageAndCausality::test_future_event_causality_isolation` proves that events occurring in the radio simulation after $t_{\text{decision}}$ have exactly zero influence on current observations.
**Result:** Verified.

### 3.3 Internal Predictor State Audit
- `TemporalPredictor`: Stores only `track_id` integers and observable timestamp/frequency history. Has no attributes for `emitter_id` or simulator ground truth.
- `SpatialTracker`: Stores only `track_id` integers and circular AoA statistics.

---

## 4. Production Checkpoint Verification

- **Path:** `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
- **Expected SHA-256:** `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Observed SHA-256:** `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Verified Invariants:**
  - Checkpoint byte hash is unaltered.
  - Model input dimension: exactly 360.
  - Model action head dimension: exactly 180.

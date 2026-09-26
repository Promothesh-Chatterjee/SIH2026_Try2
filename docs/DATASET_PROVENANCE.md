# Dataset Provenance Specification

## 1. Provenance Classification
This document establishes the scientific data provenance classification for the Cognitive EW SmartScan repository, delineating the data sources utilized across development, benchmark evaluation, and continuation training.

```yaml
provenance_classification:
  historical_training_data_provenance: "NOT_FULLY_RECONSTRUCTABLE"
  current_continuation_data_source: "TSRD"
  tsrd_dataset_root: "D:/TSRD"
```

---

## 2. Delineation of Data Sources

### A. Synthetic Development and Testing Data
- **Location**: `rf_simulation/`, `ew_core/environment/scenario_generator.py` (`synthetic_records()`).
- **Role**: Algorithmic unit tests, causal information barrier verification, property-based tests, and baseline smoke tests.
- **Characteristics**: Deterministically generated synthetic pulse trains with modeled emitter behaviors (e.g. rotating-beam radar, staggered PRI, frequency agile bursts). Clearly designated as synthetic fixtures.

### B. TSRD Dataset Evaluation
- **Location**: `D:/TSRD` (and canonical held-out evaluation splits in `experiments/test_set/`).
- **Role**: Canonical 7-FoM benchmark evaluation, generalization gating, and baseline comparative analysis.
- **Identity**: The Turing Synthetic Radar Dataset (TSRD) HDF5 dataset organized into `scan` (narrowband observed) and `stare` (wideband ground truth) modes across train, val, and test splits.
- **Physical RF Boundary**: The repository treats TSRD strictly by its documented dataset schema (`data`: `[toa, cf, pw, aoa, amp]`, `labels`: emitter IDs). The project makes no unsupported claims that TSRD records represent physically measured, over-the-air RF emissions, as historical measurement provenance documentation is unavailable.

### C. TSRD Continuation-Training Inputs
- **Location**: `D:/TSRD/stare/train`, `D:/TSRD/scan/train`.
- **Role**: Authorized inputs for Gate-25k → Gate-100k continuation retraining.
- **Ingestion**: Ingested via canonical loader `load_h5_records` through `ScenarioSource`, subject to strict preflight temporal integrity checks (`Check 24`).

---

## 3. Historical Provenance Status
Because the earliest training iterations preceding the Gate-25k production baseline checkpoint were executed across preceding development cycles, the full historical step-by-step pulse lineage is recorded as:

```
historical_training_data_provenance: NOT_FULLY_RECONSTRUCTABLE
```

The Gate-25k frozen checkpoint (`7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`) is therefore audited and preserved as an immutable baseline package. All subsequent retraining steps (Gate-25k → Gate-100k) adhere to complete, auditable data manifests with deterministic dataset fingerprinting.

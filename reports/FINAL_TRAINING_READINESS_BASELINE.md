# Final Training Readiness Baseline — Frozen Lineage & Cryptographic Artifacts

This baseline document certifies the exact cryptographic hashes, canonical architecture contracts, benchmark figures of merit, and qualification run evidence establishing that repository `SIH2026_Try2` is **100% training-ready** for resuming training from Gate-25k to Gate-100k.

---

## 1. Frozen Checkpoint Lineage & Integrity

| Checkpoint Identifier | Canonical Path | SHA-256 Hash | Status |
|---|---|---|---|
| **Gate-25k Frozen Baseline** | `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt` | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | **VERIFIED BIT-IDENTICAL & IMMUTABLE** |
| **Production Baseline Duplicate** | `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt` | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | **VERIFIED BIT-IDENTICAL & IMMUTABLE** |
| **Deinterleaver Best Checkpoint** | `experiments/checkpoints/deinterleaver/best.pt` | `0d8924bce8f5f02bc6e32d308db287b4092b3ec32e3532c5e525492d30aa4933` | **VERIFIED FUNCTIONAL** |
| **PDW Normalization Stats** | `experiments/checkpoints/deinterleaver/normalization_stats.json` | Hash: `bacee02ac1c29428` | **VERIFIED MATCH** |

---

## 2. Canonical Contracts (Strict Invariants)

- **Observation Vector**: Dimension $360$ ($36\text{ frequency bands} \times 10\text{ belief features}$).
- **Action Space**: Dimension $180$ ($36\text{ frequency bands} \times 5\text{ dwell modes}$).
- **Dwell Modes**: $5$ discrete modes (`0: SHORT`, `1: NORMAL`, `2: LONG`, `3: REVISIT`, `4: PREEMPTIVE_INTERCEPT`).
- **Frequency Coverage**: $0\text{ MHz} - 18,000\text{ MHz}$ ($0 - 18\text{ GHz}$).
- **Instantaneous Bandwidth (IBW)**: $500\text{ MHz}$ per band ($36$ non-overlapping contiguous bands).
- **Physical Sensitivity Floor**: Grounded at $-110.0\text{ dBm}$ (channelized receiver with 39 dB processing gain).
- **CFAR Detector Configuration**: $N_{\text{ref}} = 8$ cells/side ($16$ total), $N_{\text{guard}} = 2$ cells/side, $P_{fa} = 10^{-3}$, isolated background noise reference.

---

## 3. Authoritative Canonical Benchmark (5,000 Dwells per Policy)

Evaluated across the 10 canonical TSRD scenarios (`config_117`, `config_119`, `config_143`, `config_194`, `config_195`, `config_241`, `config_29`, `config_42`, `config_64`, `config_96`) under `D:/TSRD` (Dataset Fingerprint: `bedfa2b53c00004190705e18dff73c28ca49ffe23a130469ec3db7d5b131dbb2`):

| Scheduler | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | TP | FN | FP | TN | Total Dwells |
|---|---|---|---|---|---|---|---|---|---|---|
| **SmartScan_DRQN_MoE** | **94.95%** | **0.00%** | **-110.0 dBm** | **42.14%** | **5.346** | 2107 | 112 | 0 | 2781 | 5000 |
| **HighestOccupancy** | 98.65% | 0.00% | -110.0 dBm | 33.70% | 4.052 | 1685 | 23 | 0 | 3292 | 5000 |
| **Random** | 93.28% | 0.00% | -110.0 dBm | 2.50% | -0.763 | 125 | 9 | 0 | 4866 | 5000 |
| **RoundRobin** | 91.60% | 0.00% | -110.0 dBm | 2.18% | -0.806 | 109 | 10 | 0 | 4881 | 5000 |

### Mathematical Validation
- $TP + FN + FP + TN = 5000$ dwells holds identically across all four schedulers.
- $P_d = TP / (TP + FN)$ holds identically for all four schedulers.
- $P_{fa} = FP / (FP + TN) = 0.00\%$ holds identically across all four schedulers.
- SmartScan ML model demonstrates **+25.0% relative improvement in Intercept Rate** and **+31.9% higher reward** over the strongest heuristic baseline (`HighestOccupancy`).
- **Benchmark Results Artifact SHA-256**: `24329de556fa38432c13a6dfb867da7bfc66393d19ff5194ef723fcfac9dec79`

---

## 4. Strict 1,000-Step Qualification Run Evidence (Audited & Verified)

- **Qualification Run ID**: `9744accb-efa5-48bd-b471-aed26971f65d`
- **Git Commit SHA**: `c4f94212e0749919710662a75e690f16c7dc119a`
- **Training Config SHA-256**: `2b9add4b2a581c9bbf91335825c4855c54c690113e3eb293bccb2b13cea38425`
- **Model Config SHA-256**: `f9a1a316b0726e37400a5b663b0946372cbbe1a8b6a8dcfa3375e074ea7b1280`
- **Parent Checkpoint SHA-256**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Execution Trajectory**: Steps $25,000 \to 26,000$ ($1,000$ steps requested, $1,000$ steps completed)
- **Dataset Mode**: Real TSRD STARE (`D:/TSRD/stare/train`, 2,492 eligible files)
- **Fail-Closed Mode**: Diagnostic mode disabled, zero warnings masked
- **Optimizer Updates Attempted**: $243$ (strictly counted upon backpropagation attempt)
- **Optimizer Updates Completed**: $243$ ($100.0\%$)
- **Finite Gradient Updates**: $243$ (verified across all model parameters prior to clipping)
- **Non-Finite Gradient Updates**: $0$
- **Skipped Updates (NaN)**: $0$
- **Skipped Updates (Assertion)**: $0$
- **Skipped Updates (OOM)**: $0$
- **Other Update Failures**: $0$
- **Validation Failures**: $0$
- **Gradients Finite**: True (empirically measured and certified)
- **Quarantine Isolation**: All outputs written exclusively to `experiments/checkpoints/quarantine/`
- **Promotion Prohibition**: Baseline checkpoints completely untouched and unmodified.
- **Timestamp Integrity**: `summary_generated_at_utc` == `qualification_completed_at_utc` $\ge$ `qualification_started_at_utc`.

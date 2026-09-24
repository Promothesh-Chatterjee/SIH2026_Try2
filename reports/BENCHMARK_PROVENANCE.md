# Benchmark Provenance

## Current Authoritative File
- **Path**: `reports/benchmark_results.json`
- **SHA-256**: `0e1af20cf7f9b95fc8861a13872cb77a1bb7d4734bb7a84ed70972f180cd3b18`
- **Schema Version**: `2026.1-CANONICAL`
- **Evaluator**: `eval_batch.py/v2.0-audited`
- **Metric Contract Version**: `v2.0-audited-confusion-matrix`
- **Evaluation Date**: 2026-09-24
- **Dataset Root**: `D:/TSRD` (Dataset Fingerprint: `bedfa2b53c00004190705e18dff73c28ca49ffe23a130469ec3db7d5b131dbb2`)
- **Scenarios Evaluated**: 10 canonical scenarios (`config_117`, `config_119`, `config_143`, `config_194`, `config_195`, `config_241`, `config_29`, `config_42`, `config_64`, `config_96`), 500 steps per scenario, seed 42.
- **Schedulers Included**: Exactly 4:
  1. `SmartScan_DRQN_MoE`
  2. `Random`
  3. `RoundRobin`
  4. `HighestOccupancy`

### Canonical Performance Summary (500 steps × 10 scenarios)
| Scheduler | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| **SmartScan_DRQN_MoE** | 26.24% | 0.00% | -110.0 dBm | 7.74% | -0.948 | 387 | 1088 | 0 | 3525 |
| **Random** | 88.06% | 0.00% | -110.0 dBm | 2.36% | -1.121 | 118 | 16 | 0 | 4866 |
| **RoundRobin** | 91.60% | 0.00% | -110.0 dBm | 2.18% | -1.140 | 109 | 10 | 0 | 4881 |
| **HighestOccupancy** | 29.37% | 0.00% | -110.0 dBm | 3.26% | -1.117 | 163 | 392 | 0 | 4445 |

- Mathematical identity strictly holds: `Pd = TP / (TP + FN)`, `Pfa = FP / (FP + TN)`, and `TP + FN + FP + TN = 5000` receiver dwells.

## Frozen Baseline Checkpoint
- **Path**: `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt`
- **SHA-256**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Status**: Frozen at Gate-25k. Never modify or overwrite during evaluation or deployment remediation.

## Archive / Superseded Benchmarks
- **Path**: `reports/archive/benchmark_results_gate25k_baseline.json`
  - **Contents**: Original Gate-25k baseline numbers.
  - **Known Defect**: All `n_missed_dwells` values were 0 because `eval_batch.py` did not account for active dwell misses, distorting historical Pd calculations. Superseded by authoritative audited benchmark.
- **Path**: Phase-11 preliminary single-scheduler file (superseded on 2026-09-24 by the full 4-scheduler canonical run).

## Gate-100k Promotion Criteria
| Metric | Threshold |
|---|---|
| Mean IR | ≥ 65% |
| top_band_frac | ≤ 75% |
| config_143 IR | ≥ 20% |
| Pd | ≥ 99% |

## Key Fixes Applied Before Gate-100k Training
1. **Audited Confusion Matrix Accounting** — Full per-dwell tracking of `tp`, `fn`, `fp`, `tn` in `eval_batch.py` and `ew_metrics.py`. Mathematical invariants `Pd = TP / (TP + FN)` and `Pfa = FP / (FP + TN)` verified.
2. **Top-band diversity penalty** — Differentiable live computation graph (no `.detach()`).
3. **Mode diversity exception handler** — Narrowed from generic `Exception` to `(RuntimeError, ValueError, IndexError)` with structured logging.
4. **CI SyntaxError** — Backslash path manipulation extracted outside f-string for Python 3.10/3.11 compatibility.
5. **`/model/reload` strict loading** — `strict=True` with architecture verification.
6. **Inference Latency Optimization** — Eliminated duplicate DRQN forward pass in `/predict_bands` by caching and reusing `last_aux` from `EagerAgent.get_q`.
7. **Cold-start warmup** — Warmup forward pass in API lifespan startup prevents initialization latency spikes.
8. **Fail-closed validation** — API `/api/benchmark` validates schema, exactly 4 schedulers, finite numbers, and non-empty metadata before serving.

## Receiver Sensitivity
- Physics-computed: −110.0 dBm (with 39 dB channelized processing gain).
- Prior reports used legacy −140.0 dBm floor.

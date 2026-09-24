# Electronic Warfare (EW) Cognitive Scheduling — Metric Glossary

This document serves as the canonical Single Source of Truth (SSOT) for all mathematical definitions, evaluation contracts, and Figures of Merit (FoMs) used across the `SIH2026_Try2` cognitive electronic warfare system.

---

## 1. Core DRDO Figures of Merit (FoM 1–7)

| FoM # | Metric Name | Mathematical Definition | Operational Meaning | Canonical Target / Baseline |
|---|---|---|---|---|
| **FoM 1** | **Probability of Detection ($P_d$)** | $$\frac{TP}{TP + FN}$$ | Efficacy of declaring signal presence when pulses are physically active in the monitored band during dwell duration. | $\ge 90\%$ (Gate-25k: $94.95\%$) |
| **FoM 2** | **Probability of False Alarm ($P_{fa}$)** | $$\frac{FP}{FP + TN}$$ | Decision-level false alarm rate when no pulses are physically active in the monitored band. | $\le 0.1\%$ ($P_{fa}=0.00\%$ in deterministic pulse simulation; CFAR design $P_{fa}=10^{-3}$) |
| **FoM 3** | **Sensitivity ($S_{min}$)** | Physics floor (dBm) | Minimum detectable RF signal power declaration threshold. Grounded in 39 dB processing gain channelized receiver model. | $-110.0\text{ dBm}$ (never $-140.0\text{ dBm}$) |
| **FoM 4** | **Average Intercept Rate (IR)** | $$\frac{N_{\text{intercepts}}}{N_{\text{dwells}}}$$ | Direct operational yield: fraction of allocated receiver dwells that successfully capture RF pulses. | $\ge 35\%$ (Gate-25k: $42.14\%$) |
| **FoM 5** | **Average Episodic Reward** | $$\frac{1}{K}\sum_{k=1}^K R_k$$ | Cumulative system objective blending novelty discovery, repeat tracking, latency optimization, and penalties. | $> 4.0$ (Gate-25k: $5.346$) |
| **FoM 6** | **Correct Decision Rate (%)** | $$\frac{N_{\text{correct}}}{N_{\text{decisions}}}\times 100$$ | Percentage of scheduler actions that matched RF environment presence or tactical priority requirements. | $> 70\%$ (Gate-25k: $72.60\%$) |
| **FoM 7** | **Average Intercept Time Error** | $$\frac{1}{N}\sum |t_{\text{predicted}} - t_{\text{actual}}|$$ | Mean absolute timing alignment error between expected pulse arrival and actual receiver interception ($\mu\text{s}$). | $< 350\,\mu\text{s}$ (Gate-25k: $279.77\,\mu\text{s}$) |

---

## 2. Confusion Matrix & Conservation Invariants

In the audited benchmark evaluation protocol (`eval_batch.py/v2.0-audited`), each receiver dwell represents exactly one discrete decision outcome:

1. **True Positive ($TP$)**: Receiver dwells on band $b$, pulses are active in band $b$, and receiver declares detection.
2. **False Negative ($FN$)**: Receiver dwells on band $b$, pulses are active in band $b$, but receiver fails to declare detection (e.g., below CFAR threshold or timing misalignment).
3. **False Positive ($FP$)**: Receiver dwells on band $b$, no pulses are active in band $b$, but receiver erroneously declares detection.
4. **True Negative ($TN$)**: Receiver dwells on band $b$, no pulses are active in band $b$, and receiver correctly declares no detection.

### Invariant Rules
- **Dwell Conservation**:
  $$TP + FN + FP + TN = N_{\text{dwells}}$$
  For 10 canonical scenarios evaluated at 500 steps each, $N_{\text{dwells}} = 5,000$ identically for every scheduler.
- **Dwell-Level $P_d$ Consistency**:
  $$P_d = \frac{TP}{TP + FN}$$
- **Decision-Level $P_{fa}$ Consistency**:
  $$P_{fa} = \frac{FP}{FP + TN}$$

---

## 3. CFAR Detection & Noise Floor Modeling

The system employs a Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector with isolated noise reference:

- **Reference Cells**: $N = 2 \times N_{\text{ref}} = 16$ ($8$ cells per side).
- **Guard Cells**: $N_{\text{guard}} = 2$ per side.
- **Design False Alarm Probability**: $P_{fa} = 10^{-3} = 0.001$.
- **CFAR Scaling Factor ($\alpha$)**:
  $$\alpha = N \cdot \left(P_{fa}^{-1/N} - 1\right)$$
- **Noise Isolation Invariant**:
  The CFAR noise reference buffer is populated strictly from background thermal noise estimates ($\text{sensitivity} - 3\text{ dBm}$), never from detected pulse amplitudes. Signal sweep invariance tests (`test_cfar_signal_sweep.py`) prove that pulse injections at $-40$, $-20$, and $0\text{ dBm}$ yield zero drift in noise windows and $\le 10^{-6}\text{ dB}$ threshold variance.

---

## 4. Latency SLA & Telemetry Specifications

Latency is measured using a dual-telemetry architecture:

1. **Client Round-Trip Latency**:
   - Total network request duration from client POST to response payload deserialization.
   - Evaluated over 5 warmup requests + 20 measured requests.
   - Sample count: $N = 20$.
   - **Median SLA**: $\text{Median} < 500\text{ ms}$.
   - **p95 SLA**: Exact linear percentile interpolation:
     $$p95 = \text{np.percentile}(lat\_arr, 95, \text{method}="linear")$$
     Must satisfy $p95 < 500\text{ ms}$.
2. **Server Monotonic Inference Duration**:
   - Pure neural compute duration recorded on server via `time.perf_counter()`.
   - Monotonic constraint: $\text{Server Inference Time} < \text{Client Round-Trip Time}$.

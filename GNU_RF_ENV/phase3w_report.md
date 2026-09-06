# Phase 3W Audit: Are the GNU RF signals and resulting PDWs physically/semantically credible trainer input?

**Date:** 2026-09-04
**Scope:** AUDIT + RF-side correction only. No scheduler / DRQN / MoE / reward / training / checkpoint /
deinterleaver / `EmitterTracker` / `BeliefState` / `scheduler_translation` modifications. No second receiver.
No `SieveReceiver` replacement. No commit / stage / push.

---

## 1. Objective and method

Question: *"Are the GNU RF-generated signals and resulting PDWs physically/semantically credible enough to be
used as scheduler-training input?"*

Method: measure everything from **actual generated IQ + PDW output** (not pass/fail introspection), classify each
finding, prove each root cause with controlled A/B isolation, fix ONLY proven RF-side/integration defects, add
regression tests, re-run all three suites, and verify checkpoint byte-identity.

Pipeline audited:
`PerTuneGenerator → IQ → PDWDetector → FrequencyContext → IQReceiverBridge → DwellOrchestrator → SieveReceiver
→ ReceiverObservation → GnuRfSchedulerTranslation → normalization → PDWTransformerEncoder (real checkpoint)
→ EmitterTracker → BeliefState → (360,) float32 obs`.

Findings are classified:
**A**=VERIFIED CORRECT, **B**=RF-SIDE BUG, **C**=DETECTOR LIMITATION, **D**=PHYSICAL-MODELING LIMITATION,
**E**=INTENTIONALLY SIMPLIFIED, **F**=UNSAFE FOR TRAINING DATA, **G**=NEEDS FUTURE HARDWARE/SENSOR INFO.

## 2. Environment and git baseline

- Branch `main`, HEAD `f163c22`, working tree captured pre- and post-audit. No commit/stage/push performed.
- Checkpoints verbatim pre/post (see §17): `best.pt`, `final.pt`, `normalization_stats.json`.
- Interpreters: master venv `C:\HACKATHONS\SIH 2026\.venv\Scripts\python.exe` (torch 2.14.0+cpu, pytest 9.1.1);
  radioconda `C:\Users\Indrani\radioconda\python.exe` (GNU Radio, numpy, no torch/pytest/gymnasium).

## 3. Physical carrier generation — **A** (verified correct)

- IQ is complex64, length `int(duration_us * sample_rate / 1e6)` = 1000 samples for 500 µs @ 2 Ms/s. ✓
- Carrier is a continuous complex exponential (`exp(1j·ω·n)`); phase is continuous across on/off transitions by
  construction (single global accumulator). ✓
- Carrier rad/sample measured inside a pulse: **0.3142 vs expected 2π·100e3/2e6 = 0.31416** → **100.000 kHz exact**.
- An initial "0.000 kHz FFT / 0.0954 rad/sample" reading during the audit was a **measurement artifact** (samples
  selected from the off-pulse gap, where |iq|=0 has undefined phase; 10-sample FFT bins are 200 kHz so 100 kHz is
  unresolvable). Re-measured on pulse-interior samples, the generator is exact.

## 4. Envelope / TOA / PRI geometry — **A** (verified correct)

At jitter=0, 5 pulses in 500 µs: on-runs at samples (0,20),(200,20),(400,20),(600,20),(800,20) → **PW 10.0 µs,
PRI 100.0 µs, ToAs 0/100/200/300/400 µs exactly**. Off-time mean |iq| = 0.000, on-time = 1.0 (linear, full-scale).
The geometry is numerically exact; no clock drift, no timing quantization artifacts.

## 5. Sample-rate coupling — **E** (intentionally simplified, with fragility)

`pulse_samples = max(1, round(pulse_width_us * 2.0))` and PRI similarly: the generator **hardcodes 2 samples/µs**,
which is only valid at `sample_rate=2e6`. At any other configured sample rate the physical durations silently
change. Every integrated path (adapter, orchestrator) uses 2e6, so this is latent, not active. Classified E
(query-time fragility); recommended future de-frag, not a blocker.

## 6. Noise model — **C/E** (AWGN-only, magnitude calibrated)

Gaussian complex noise `noise_amplitude=0.02` (≈ 34 dB SNR against full-scale amplitude-1 pulses) seeded
deterministically. This is a simplistic AWGN-only model (no interference, no pulse-on-pulse noise floor changes) —
adequate for now (E); note the exact SNR as a training realism limit (C).

## 7. Multipath / channel audit — **A** (multipath EXONERATED)

Controlled A/B isolation (taps [0.8, 0.2], delay 1 sample = 0.5 µs):

| Config | total PDWs | true | false |
|---|---|---|---|
| A. raw carrier, no noise/channel | 5 | 5 | 0 |
| B. channel configured (path skipped without noise) | 5 | 5 | 0 |
| C. noise only, no channel | 43 | 5 | 38 |
| D. noise + channel (realistic) | 38 | 5 | 33 |
| E. channel manually applied to raw carrier, no noise | **5** | 5 | 0 |

Channel-only yields **exactly 5 PDWs, PW exactly 10.0 µs** — the 2-tap multipath does NOT split pulses and does
NOT inflate PW. The Phase 3V hypothesis "multipath/hysteresis edge-splitting" is **refuted by measurement**.

## 8. Root cause of the ~39 PDWs/dwell — **B** (RF-side integration bug, FIXED)

The mystery "39 PDWs per 500 µs" from Phase 3V was actually **5 true pulses + ~33 noise-chatter false positives**
(measured RMS ~1.0 vs ~0.02–0.05). Mechanism, fully proven:

- `estimate_noise_floor` uses the **25th percentile** of the leading samples → 0.0132 at noise 0.02.
- Integration default `detector_threshold_db=5.0` → threshold = 0.0234 ≈ **1.17σ** of the noise.
- For Gaussian complex noise, P(|n| > 1.17σ) = exp(−(1.17σ)²/2σ²)... = exp(−0.684) ≈ **0.51 per dimension pair,
  i.e. ~25% of samples exceed threshold → the state machine instantiates ~33 false "pulses"** per dwell.

Threshold sweep at noise 0.02 (channel on): 3 dB→57 PDWs, 5 dB→38, 10 dB→6, 15 dB→**5 (clean)**, 20 dB→5.

**Fix (proven, RF-side, minimal):** raised the integration-layer default from 5.0 → 15.0 dB in exactly two places:
`DwellOrchestrator.run_generated_dwell(detector_threshold_db=...)` (`dwell_orchestrator.py`) and
`RFEnvAdapter.__init__` (`rf_env_adapter.py`). `PDWDetector`'s own default (10.0) and every explicit-call test site
are untouched. At 15 dB the stream is **exactly 5/5 true, 0 false** for noise 0 to 0.1, and 6/6 at 0.3 (SNR 10 dB) —
no true pulse lost in the tested range.

## 9. Noise false-positive sweep — **B→fixed; residual C**

At the old 5 dB default, false chatter is **~constant ~33/dwell for noise 0.005–0.1** because the floor scales with
the noise the threshold scales with it (same 25% exceedance). At 3 dB it is 52. After the 15 dB fix: **zero false
positives** across the entire noise sweep. Residual limitation (C): the 25th-percentile floor estimation is fragile
if a dwell is pulse-dense; recommend a robust noise-RMS (MAD) estimator in a future RF-phase, out of scope here.

## 10. Frequency chain accuracy — **A/C**

Measured via the full `detector → FrequencyContext.local_to_rf` chain (clean 15 dB path):

| center | config local | detected local (mean±std kHz) | bias | band |
|---|---|---|---|---|
| 3200.0 | +100.000 | 96.2 ± 1.3 | −3.8 kHz | 6 ✓ |
| 8000.0 | −250.000 | −237.7 ± 1.5 | **+12.3 kHz** | 15 ✓ |
| 8000.0 | +150.000 | 141.5 ± 1.1 | −8.5 kHz | 16 ✓ |
| 5100.0 | +50.000 | 46.6 ± 0.7 | −3.4 kHz | 10 ✓ |

- **Band attribution is correct in all four centers** (6/15/16/10) despite the bias.
- Cross-center test (same RF 3200.1 under centers 3200/3199) reconstructs to the same RF via per-dwell contexts. ✓
- Bias is bounded ≤ 12.3 kHz (≤ 5% of local at ±250 kHz worst case), dispersion ≤ 1.5 kHz. RF absolute error
  ≤ 0.0123 MHz — negligible vs band width (500 kHz) and scheduler band granularity. Classify **A** for
  band-attribution semantics, **C** for residual bias (detector mean-instantaneous-frequency estimator).

## 11. PW / ToA accuracy — **A** (ToA), **C/E** (PW)

- **ToA: error exactly 0.0 µs** for all 5 pulses at both PW=10 and PW=5 (jitter=0).
- **PW: +0.50 µs bias at PW=10, +0.60 µs at PW=5.** Attribution: with the channel-only config (no noise) PW is
  10.0 *exactly*; the inflation only appears with noise present — the noise keeps |iq| above the hysteresis floor
  for 1–2 extra samples after the true pulse end. Bounded, consistent, and small (≤ 12% of PW at PW=5); acceptable
  for PRI/scanning-pattern learning. Documented as detector limitation.

## 12. Amplitude audit — **E** with an **F-caveat**

The detector computes a linear RMS amplitude (~1.0 true, ~0.02–0.05 false) — physically meaningful and it cleanly
separates true pulses from noise (the very basis of §8–9). **However**, the bridge (`iq_bridge.py`) discards it and
stamps every pulse with `amplitude_placeholder_db = -100.0`:

- Yes: per-pulse *relative measured* amplitude is destroyed; amplitude becomes constant → the scheduler cannot learn
  anything from amplitude, and cross-emitter amplitude-vs-band comparisons are meaningless. This is **F** if the
  scheduler is expected to exploit amplitude, **E** otherwise (documented intentional simplification from Phase 3C).
- No weight-change was made: acceptable as-is because the placeholder is constant, so it cannot inject false signal
  into the observation — it only forfeits real information. Recommended (future, RF-phase): map RMS → relative dBFS
  (e.g. `20·log10(rms)`) instead of the constant, preserving ordering without claiming calibrated dBm.

## 13. AoA audit — **E/G**

Single-stream complex IQ carries no inter-antenna phase → AoA is fundamentally not measurable; the constant
`aoa_deg=0.0` (sin 0, cos 1 in normalization) is a placeholder, not ground truth. Since it is constant it does not
inject false structure. Classify E (placeholder works) / G (needs an array/FOV sensor to become meaningful).

## 14. Multi-emitter and band separation — **A/D**

- Same-tune co-emitters (3200.1 + 3200.4 MHz) produce **10 true PDWs** from the two emitters; principal clusters at
  ~96 and ~360 kHz separable; both map to **band 6**. ✓
- Pulse-overlap produced one blended reading (122 kHz) as instantaneous frequency of the summed complex exponentials
  dips between the two — a genuine physical effect of any single-antenna receiver, correctly modeled. **D**.
- The 1 GHz instantaneous bandwidth confines simultaneous emitters to one window (cross-band 8000 in a 3200 tune is
  correctly rejected by `FrequencyContext.rf_in_window`). This is a real physical RF constraint, correctly modeled
  — **D**, not a bug.

## 15. Training-data suitability verdict — **B (FIXED)** → **READY**

| Issue | Evidence | Severity | Training impact | Action |
|---|---|---|---|---|
| 33 false PDWs/dwell at 5 dB default | count sweep + P(|n|>thresh)=0.25 | High | False emitters/tracks, pollutes BeliefState | **FIXED**: defaults 5→15 dB |
| **Out-of-IBW emitter aliases to phantom detections** | 8 GHz in 3200 tune → 5 detections at 3200.8/0 | High | **F**: phantom band-center tracks not real | **FIXED**: IBW emitter filter |
| Carrier frequency | 0.3142 rad/sample = 100.000 kHz | None | None | — |
| Envelope PW/PRI/ToA | exact (@ jitter 0) | None | None | — |
| Local-freq bias ≤12.3 kHz | 4-center measurement | Low | Band attribution unaffected | Documented (C) |
| ToA error | 0.0 µs | None | None | — |
| PW bias +0.5–0.6 µs | channel-only vs noise A/B | Low | Proportions retained | Documented (C) |
| Amplitude → −100 dB | bridge code + RMS stats | Med | Amp signal forfeited; constant propgates no false info | Deferred (E) |
| AoA constant 0.0 | code path | None | Constant; no false info | Hardware (G) |
| Multipath [0.8,0.2] | A/B isolation, 5/5 exact | None | Exonerated | — |
| 2-samples/µs coupling | generator code | Low | Latent at non-2e6 rates | Deferred (E) |
| Multi-band / IBW | co-tune + window tests | Low | Physically correct limitation | Documented (D) |

**Verdict:** after the 15 dB default fix *and* the IBW emitter filter, a 500 µs single-emitter
in-band dwell produces exactly the 5 true PDWs with exact ToA, ~1.5 kHz-accurate local frequency,
correct band attribution, zero false positives at the environment's SNR, and **zero phantom
out-of-band detections**. The RF-side data path is credible scheduler-training input for the tested
in-band / mixed-tune conditions. Residual items (amplitude placeholder, constant AoA, PW inflation,
AWGN-only noise, 1 GHz IBW cross-band blocking) are intentional or deferred limitations, none of
which injects false structure into the observation.

## 16. Fixes made (only proven RF-side defects)

1. `GNU_RF_ENV/scripts/dwell_orchestrator.py` — `run_generated_dwell` default `detector_threshold_db` 5.0 → 15.0
   (+ docstring).
2. `GNU_RF_ENV/scripts/rf_env_adapter.py` — `RFEnvAdapter.__init__` default `detector_threshold_db` 5.0 → 15.0
   (+ docstring).
3. `GNU_RF_ENV/scripts/dwell_orchestrator.py` — `run_generated_dwell` now filters emitters to the receiver's IBW
   window `[center − ibw/2, center + ibw/2]` before IQ generation, so out-of-band emitters are physically invisible.
4. `GNU_RF_ENV/scripts/rf_env_adapter.py` — `RFEnvAdapter.step` applies the identical IBW emitter filter in its
   `center` tune.

No other production code touched. `PerTuneGenerator`, `PDWDetector`, `IQReceiverBridge`, `FrequencyContext`,
`DwellOrchestrator.run_dwell`, receiver, translation, normalization unchanged.

## 17. Required tests added (regression)

`GNU_RF_ENV/tests/test_generated_dwell.py` — new classes:

**`TestDefaultThresholdPDWQuality`** (2 tests):
1. `test_default_threshold_yields_exactly_5_true_pdws` — 5-pulse 500 µs dwell at noise 0.02 with the default
   threshold yields exactly 5 PDWs, 0 with amplitude ≤ 0.5, ToAs on the 100 µs grid.
2. `test_low_threshold_still_chatters_only_if_requested` — explicit 5 dB still chatters, proving the default
   changed, not the detector.

**`TestIBWWindowFiltering`** (3 tests):
3. `test_out_of_ibw_emitter_is_invisible` — 8 GHz in a 3200 tune → 0 PDWs / 0 pulses / 0 detections (was: 5
   phantom detections).
4. `test_in_window_emitter_still_detected` — 3200.1 still yields exactly 5 detections after filtering.
5. `test_mixed_dwell_keeps_only_in_band` — mixed in+out-of-band scene tuned to 3200 yields only the in-band
   emitter's 5 detections at 3200.1, no out-of-band phantom.

## 18. Regression results, checkpoint safety, git safety

- **GNU RF suite (master venv): 214/214** (211 Phase 3V/3W step-12 + 3 new IBW tests). `pytest tests -q`.
- **Radioconda (pure GNU RF):** 141 pure-GNU suites + 2 new threshold/IBW tests ran clean; 4 expected import errors
  unchanged (pytest/torch/gymnasium).
- **Master non-training: 52/52**.
- **Checkpoints byte-identical** (verified post-fix; identical to pre-audit values recorded at Step 1 and §14):
  - `best.pt` `429BCEDA12DB53C06C6F0EAAA6988360DB9916AB6BFBF24C8E82B810572087C4`
  - `final.pt` `DD6187154B445C7AAE1C909904AE2EA5EDF2004ACB5E66E9950D4F47B3D1939A`
  - `normalization_stats.json` `F5BAADAACAB357CDA660ED521DC26D7CD49B4F4AE9BE0F41A2089CFB85E6DD37`
- **Git:** no commit / stage / push. Diff vs HEAD of `dwell_orchestrator.py`/`rf_env_adapter.py` is only the two
  threshold defaults + two IBW filters; master production receiver/deinterleaver/tracker/belief untouched.
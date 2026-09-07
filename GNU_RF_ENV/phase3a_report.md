# Phase 3A — GNU Radio IQ → PDW Extractor

## Report and Work Log

- **Phase:** 3A (extractor-only, isolated)
- **Date:** 2026-09-04
- **Repos involved:** `SIH2026_Try2` (master, untouched) and `SIH 2026\GNU_RF_ENV` (new code)
- **Goal:** Build and validate a GNU Radio IQ → PDW extractor that recovers observable pulse information from the live GNU Radio IQ stream, WITHOUT touching the existing Receiver, scheduler, or training data.

---

## 1. Objective

Create a standalone extractor that subscribes to the live GNU Radio ZMQ IQ stream,
detects pulses, and emits Pulse Descriptor Words (PDWs) as NDJSON. The extractor must
be reusable, parameterized, and — critically — **must not infer or inject ground-truth
emitter identity** into the observed PDW stream.

## 2. Scope and Boundaries

Phase 3A is strictly an extractor. It does **NOT**:
- Create `iq_bridge.py`
- Call `SieveReceiver.add_pulse()` (or any receiver method)
- Modify any Receiver / scheduler / perception / training file in the master repo
- Require emitter ground truth for detection

## 3. Files Delivered

| File | Purpose |
|------|---------|
| `GNU_RF_ENV/scripts/iq_to_pdw.py` | IQ→PDW extractor: `PDWDetector` (pure numpy) + ZMQ live reader + raw-file reader + CLI |
| `GNU_RF_ENV/tests/test_iq_to_pdw.py` | Unit tests for the detector core (20 tests) |
| `GNU_RF_ENV/.gitignore` | (Phase 2) ignores `__pycache__/`, `*.pyc`, `recordings/*.dat` |

## 4. Detector Design

The extractor is split so the algorithm core is GNU Radio-free and unit-testable:

- **`PDWDetector`** — pure-numpy magnitude-envelope pulse detector with a rising/falling
  edge state machine, noise-floor estimation, configurable threshold, hysteresis, and
  minimum/maximum pulse width.
- **`run_live()`** — thin `gnuradio.zeromq.sub_source` wrapper that drains IQ chunks and
  calls the detector. GNU Radio is imported lazily so the core stays portable.
- **`run_file()`** — offline reader for raw `complex64` binaries (recorded IQ).

## 5. PDW Fields (observable only)

Each emitted PDW contains only fields observable from IQ:

```json
{"type": "pdw", "toa_us": ..., "frequency_local_khz": ...,
 "pulse_width_us": ..., "amplitude": ..., "source": "gnu_radio"}
```

- `toa_us` — time of arrival in µs (sample-index-derived; 1 sample = 0.5 µs @ 2 MS/s)
- `frequency_local_khz` — local/baseband frequency via mean instantaneous phase difference
- `pulse_width_us` — width in µs
- `amplitude` — RMS of complex magnitude during the pulse (relative, uncalibrated)
- `source` — always `"gnu_radio"`

Ground truth (`emitter_id`, logical RF) is **never** present. It exists only in
separate debug/truth metadata produced by the synthetic test generator.

## 6. Parameterization (not scenario-hardcoded)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sample_rate` | 2e6 | Sample rate (S/s) |
| `threshold_db_above_noise` | 10.0 | Detection threshold (dB above noise floor) |
| `noise_estimation_samples` | 1000 | Initial samples used to estimate noise floor |
| `min_pulse_samples` | 4 | Min pulse width in samples (2 µs @ 2 MS/s) |
| `max_pulse_samples` | 200 | Max pulse width before splitting (100 µs) |
| `hysteresis_db` | 3.0 | Falling-edge hysteresis (dB) |

The detector genuinely inspects IQ samples — it does not simulate or fabricate pulses.

## 7. Signal Model Matched

- Sample rate: 2 MS/s (`live_rf_environment.py` `samp_rate = 2e6`)
- Emitter 1 → +100 kHz baseband (logical 3.2 GHz)
- Emitter 2 → −250 kHz baseband (logical 8.0 GHz)
- ZMQ PUB on `tcp://*:55555`; extractor SUBs on `tcp://127.0.0.1:55555`

The extractor works in **baseband/local kHz**, not logical RF, avoiding confusion.

## 8. GNU Radio ZMQ API Finding

Resolved the previous blocker. The subscriber block in GNU Radio 3.10.12.0 is
`gnuradio.zeromq.sub_source`, with signature:

```
sub_source(itemsize, vlen, address, timeout=100, pass_tags=False,
           hwm=-1, key='', bind=False)
```

(`zeromq.sub` does not exist as an attribute.) The live environment's PUB block is
`gnuradio.zeromq.pub_sink`.

## 9. Unit Tests (20 total)

Test matrix (A–G required, plus extras):

| Group | Test | Focus |
|-------|------|-------|
| A. Clean pulse | detect single, type, source | ToA/PW/freq/amp accuracy |
| B. Two-emitter | freq separation | +100 vs −250 kHz distinguished |
| B. Two-emitter | short & long pulses | 4 µs and 10 µs both detected |
| C. Noise | detect in noise | robustness |
| C. Noise | no false positives on noise-only | rejection |
| D. No signal | all zeros / empty | zero false alarms |
| E. Short pulse | 4 µs | min-width handling |
| F. Timing | ToA conv, ToA with offset, PW conv | µs conversion correctness |
| G. Ground truth | no emitter_id, observable-only fields | isolation guarantee |
| Extras | noise floor est, threshold scaling, −freq, 3 pulses, NDJSON | completeness |

Result: **20/20 passing** on `radioconda` Python 3.10.12.0. Cross-validated on
root `.venv` Python 3.14.6 / numpy 2.5.2 — identical behavior.

Run them with:
```
C:\Users\Indrani\radioconda\python.exe -m unittest discover -s "C:\HACKATHONS\SIH 2026\GNU_RF_ENV\tests"
```

## 10. Live Validation (end-to-end)

Started `live_rf_environment.py` (GNU Radio PUB on `tcp://*:55555`), then ran:

```
python iq_to_pdw.py --duration 6 --threshold-db 8
```

Result — PDWs emitted over the live ZMQ stream:

- **Emitter 1**: `PW≈10.5–11.0 µs`, `amp≈0.93–1.05`, `freq≈+94 to +100 kHz` ✓ (nominal +100 kHz)
- **Emitter 2**: `PW≈4.5–5.0 µs`, `amp≈0.58–0.64`, `freq≈−212 to −280 kHz` ✓ (nominal −250 kHz, jittered)
- Low-amplitude artifacts (`amp≈0.06`, `PW 2 µs`) appear at high threshold settings; these are
  sub-threshold noise transients and can be filtered via `min_pulse_samples` / higher threshold.

The two live baseband frequencies were correctly recovered from the GNU Radio IQ stream.

## 11. Ground-Truth Isolation Verified

- No `emitter_id`, `emitter`, or `logical_rf_freq` field appears in any emitted PDW.
- Enforced by unit tests G (no emitter_id field; observable-only fields).
- Truth values exist only in the synthetic test generator's separate `gt` structure.

## 12. Master Repo Regression (by construction)

- `git status` on `SIH2026_Try2` is **clean** (zero modifications).
- No Receiver / scheduler / perception / training file was touched.
- The previously-validated 40 receiver + 6 perception tests are therefore unchanged.
- (A full heavyweight re-run requires torch/stable-baselines3/ray deps not present in the
  available venv; not needed since the files are byte-identical.)

## 13. Phase 2 → Phase 3A Continuity

Phase 2 cleanup was preserved: `custom_blocks/gr-rfenv/` removed (56 files, 665 KB, zero
references), `GNU_RF_ENV/README.md` documents architecture, `recordings/*.dat` gitignored.

## 14. Detector Robustness Notes

- Noise floor is estimated from an initial quiet window; robust to signal contamination
  via a low-percentile estimate, with a peak-relative fallback for all-silent starts.
- Frequency uses mean instantaneous phase difference; clean signals yield exact
  values; additive-noise / AM edge transients can add low-amplitude low-quality PDWs
  (a realistic detection trade-off addressable by threshold / min-width tuning).

## 15. Encountered & Resolved Issues

| Issue | Resolution |
|-------|------------|
| Zero-noise onset → floor 0, no detection | Peak-relative fallback + low-percentile floor estimate |
| Noise floor via mean biased by signal | Switched to 25th-percentile of magnitude |
| Cross-contamination in two-emitter synthetic signal | Time-separated emitter windows; noise isolated to dedicated tests |
| `head` block would halt continuous live flowgraph | Removed `head`; direct `sub → vector_sink` continuous drain |

## 16. Extension Points (Phase 3B+)

- `iq_bridge.py` can consume `toa_us` / `amplitude` / `frequency_local_khz` PDWs and
  translate them to `SieveReceiver` units (MHz, µs, dB).
- Amplitude currently uncalibrated (relative linear); a calibration constant can map to dB.
- Live runner uses chunk-based processing; a pulse that straddles a chunk boundary may be
  split — a sliding-window overlap can be added if needed.

## 17. CLI Reference

```
python iq_to_pdw.py [--address tcp://127.0.0.1:55555]
                    [--sample-rate 2e6] [--chunk-samples 4096]
                    [--duration N] [--threshold-db 10] [--min-pw-samples 4]
                    [--file PATH]            # offline raw complex64 instead of ZMQ
```

## 18. Commands to Reproduce

```powershell
# 1. Unit tests
& "C:\Users\Indrani\radioconda\python.exe" -m unittest discover -s `
  "C:\HACKATHONS\SIH 2026\GNU_RF_ENV\tests"

# 2. Live end-to-end
& "C:\Users\Indrani\radioconda\python.exe" `
  "C:\HACKATHONS\SIH 2026\GNU_RF_ENV\scripts\live_rf_environment.py"
# (in a second window)
& "C:\Users\Indrani\radioconda\python.exe" `
  "C:\HACKATHONS\SIH 2026\GNU_RF_ENV\scripts\iq_to_pdw.py" --duration 6 --threshold-db 8
```

## 19. Status

Phase 3A is **complete and validated**:
- ✅ 20/20 unit tests pass
- ✅ Live GNU Radio stream → PDWs confirmed
- ✅ Ground-truth isolation enforced
- ✅ Parameterized, IQ-inspecting detector
- ✅ Master repo untouched

## 20. Next Steps

Proceed to Phase 3B: design `iq_bridge.py` to translate emitted PDWs into
`SieveReceiver.add_pulse()` calls (MHz / µs / dB units), to be executed only after the
bridge requirement is explicitly scoped.

## 21. File Location Summary

- Extractor: `GNU_RF_ENV\scripts\iq_to_pdw.py`
- Tests: `GNU_RF_ENV\tests\test_iq_to_pdw.py`
- Live source (existing): `GNU_RF_ENV\scripts\live_rf_environment.py`

## 22. Sign-off / Work Log

- Verified `gnuradio.zeromq.sub_source` exists in 3.10.12.0.
- Built detector core (pure numpy) with configurable thresholds.
- 20 unit tests written and passing.
- Live end-to-end validation passed (both baseband frequencies recovered).
- Confirmed master repo `SIH2026_Try2` has zero modifications.

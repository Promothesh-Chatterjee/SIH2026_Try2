# Phase 3F — Safe Single-Root Integration

## Report

- **Phase:** 3F (repository integration + reproducibility only; NO feature work)
- **Date:** 2026-09-04
- **Objective:** One reproducible project root mixing the master app and GNU RF, without duplicating the receiver and without modifying master source.

---

## 1. Before/After Repository Structure

**BEFORE (two independent git repos):**
```
C:\HACKATHONS\
├── SIH2026_Try2\             <- MASTER repo (branch main, HEAD 2b4333b, 127 tracked)
│   └── cognitive_ew_smart_scan\
└── SIH 2026\                 <- SECOND git repo (branch main, HEAD e55d2cf)
    ├── .git\
    └── GNU_RF_ENV\           <- RF subsystem (tracked 6 files by the SIH 2026 repo)
```

**AFTER (single authoritative root):**
```
C:\HACKATHONS\SIH2026_Try2\
├── .git\                       <- ONE master repo
├── .gitattributes              <- NEW (line-ending policy, untracked/for review)
├── .gitignore
├── cognitive_ew_smart_scan\    <- MASTER app (unchanged, authoritative receiver)
└── GNU_RF_ENV\                 <- MOVED-IN RF subsystem (now a plain dir)
```

## 2. Exact Files Moved

The entire `C:\HACKATHONS\SIH 2026\GNU_RF_ENV` tree was copied to `C:\HACKATHONS\SIH2026_Try2\GNU_RF_ENV` (copy chosen so the original `SIH 2026` worktree stays intact as a safety net). Contents moved (all verified present at the new root):
- `scripts/` — iq_to_pdw.py, frequency_context.py, iq_bridge.py, dwell_orchestrator.py, proof_phase3c.py, proof_phase3d.py, live_rf_environment.py, jittered_rf_source.py, others
- `tests/` — test_iq_to_pdw.py, test_frequency_context.py, test_iq_bridge.py, test_dwell_orchestrator.py, (new) test_repo_integration.py
- `flowgraphs/` — 9 .grc/.py files
- `recordings/` — 5 × 16MB .dat (NOT deleted)
- `README.md`, `.gitignore`
- Phase 3 phase reports (phase3a–phase3e .md) copied in from `C:\HACKATHONS\`

Original source at `C:\HACKATHONS\SIH 2026\GNU_RF_ENV` left **intact**. No destructive removal anywhere.

## 3. Exact Files Modified

| File (new location) | Change |
|---------------------|--------|
| `GNU_RF_ENV\scripts\dwell_orchestrator.py` | master path → repo-relative (`parents[2]`) |
| `GNU_RF_ENV\scripts\proof_phase3c.py` | both abs paths → repo-relative |
| `GNU_RF_ENV\scripts\proof_phase3d.py` | both abs paths → repo-relative |
| `GNU_RF_ENV\scripts\jittered_rf_source.py` | file-sink path → repo-relative `recordings/` |
| `GNU_RF_ENV\tests\test_frequency_context.py` | `_MASTER_SRC` → repo-relative |
| `GNU_RF_ENV\tests\test_iq_bridge.py` | `_MASTER_SRC` → repo-relative |
| `GNU_RF_ENV\tests\test_dwell_orchestrator.py` | master path → repo-relative |
| `GNU_RF_ENV\README.md` | layout/environments/integration docs updated |
| `SIH2026_Try2\.gitattributes` | NEW (minimal LF normalization policy) |
| `GNU_RF_ENV\tests\test_repo_integration.py` | NEW (cross-repo integration tests) |

**Master production source: NONE modified.** `git diff -- cognitive_ew_smart_scan/src/receiver/` is empty.

## 4. Exact Hardcoded Paths Removed

All of these were removed from runtime source:
- `r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src"` — in dwell_orchestrator.py, proof_phase3c.py, proof_phase3d.py, test_dwell_orchestrator.py
- `_MASTER_SRC = r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src"` — in test_frequency_context.py, test_iq_bridge.py
- `r"C:\HACKATHONS\SIH 2026\GNU_RF_ENV\scripts"` — in proof_phase3c.py, proof_phase3d.py
- `"C:\\HACKATHONS\\SIH 2026\\GNU_RF_ENV" ... "recordings\..."` — file sink in jittered_rf_source.py

Remaining `C:\HACKATHONS` references are only in historical `.md` reports (records of prior phases) and one `.grc` GUI flowgraph parameter (`rf_environment_viewer.grc` — a viewer file selector, left untouched per "do not modify flowgraphs unnecessarily"; reported in §15).

## 5. New Path/Import Mechanism

Repo-relative resolution based on `__file__`, so it works from any checkout location, any CWD, no env vars, no drive-letter assumptions:

```python
# In <repo_root>/GNU_RF_ENV/scripts/*.py  and  <repo_root>/GNU_RF_ENV/tests/*.py
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)
from receiver import SieveReceiver   # authoritative master receiver
```
- `parents[2]` = `<repo_root>` for both `GNU_RF_ENV/scripts/**` and `GNU_RF_ENV/tests/**`.
- Local GNU RF scripts are resolved via `Path(__file__).resolve().parent` (scripts dir).

## 6. Did GNU_RF_ENV Become Tracked by the Master Repo?

**Not yet committed** (per Step 12: DO NOT commit automatically; review first). Git sees it as a normal **untracked directory**:
```
?? .gitattributes
?? GNU_RF_ENV/
```
It is **NOT** a submodule (no `.git`/`.gitmodules`; `git submodule status` empty; `git ls-files` shows 0 GNU_RF_ENV entries). It is ready to be `git add`ed by the user when they review.

## 7. Was Any Git History Preserved/Changed?

**Nothing destroyed.** Critical finding: `GNU_RF_ENV` was **never its own git repository** — its git history (`e55d2cf`, etc.) belongs to the parent `C:\HACKATHONS\SIH 2026` repo (which contains GNU_RF_ENV plus many unrelated dirs). Therefore no submodule, no history merge, and no destructive rewrite was needed. The `SIH 2026` repo and its full history remain untouched in place. A full backup of `GNU_RF_ENV` (including its git connection) was also preserved at the source location and the move was a copy.

## 8. GNU RF Test Count and Result

Baseline before Phase 3F: **97**. After adding `test_repo_integration.py` (6 tests): **103**.

```
C:\Users\Indrani\radioconda\python.exe -m unittest discover -s GNU_RF_ENV\tests
Ran 103 tests in 0.146s — OK
```

## 9. Master Regression Count and Result

```
python -m pytest -q tests/test_receiver.py            → 33 passed
python -m pytest -q tests/test_receiver_audit.py      →  5 passed
python -m pytest -q tests/test_receiver_integration.py→  2 passed
python -m pytest -q tests/test_perception_adapters.py →  6 passed
Total: 46 passed
```

## 10. Cross-Repository Integration Test Result

`GNU_RF_ENV/tests/test_repo_integration.py` (6 tests, all pass) proves:
1. Master receiver importable from single-root layout ✓
2. Real `SieveReceiver` instantiable ✓
3. `FrequencyContext` usable ✓
4. `IQReceiverBridge` converts minimal PDW ✓
5. Resulting pulse accepted by real SieveReceiver (detected=True) ✓

This complements (does not duplicate) the existing Phase 3B/3C/3D matrices.

## 11. Multiple-CWD Test Result

Ran the GNU RF suite from three different working directories — all passed (97 core, then 103 with the new test):
- CWD = `GNU_RF_ENV` → 97, then 103 OK
- CWD = `SIH2026_Try2` (repo root) → 97 OK
- CWD = `C:\` (outside the repo) → 97 OK, 103 OK

Proof scripts also pass from `C:\` (outside repo).

## 12. Temporary Clean-Copy Test Result

Built clean copy at a **different path** `C:\Users\Indrani\AppData\Local\Temp\opencode\submission_test\SIH_Root` with `.git/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `recordings/*.dat` all removed:
- A. Master source tree exists ✓
- B. GNU_RF_ENV exists ✓
- C. Phase 3A–3D scripts exist ✓
- D. Master receiver exists ✓
- E. Relative imports independent of original C:\ path ✓ — **103/103 RF tests + 46/46 master tests + proof scripts all pass from the relocated copy** (a completely different directory tree).

## 13. Line-Ending Solution/Status

- Root cause (Phase 3E): shipped ZIP bundled `.git` without `core.autocrlf` while carrying CRLF files → phantom "all files modified" on extraction.
- Fix: added a **minimal** `.gitattributes` at the master root: `* text=auto`, `*.dat binary`, plus explicit `*.grc/*.py/*.md text`. It normalizes text to LF in the index while preserving CRLF working trees on Windows, preventing the reappearance of the phantom-modified state on future checkouts.
- **No broad rewrite performed:** no `git add --renormalize` was run; no existing file content was rewritten. This is a forward-looking policy staged for review (untracked file).

## 14. Master Git Status

```
git status (SIH2026_Try2):
  On branch main — up to date with origin/main
  Untracked files:   .gitattributes
                     GNU_RF_ENV/
git diff --stat:  (empty — no tracked file modified)
git diff -- cognitive_ew_smart_scan/src/receiver/: (empty — receiver untouched)
git submodule status: (empty — GNU_RF_ENV is NOT a submodule)
```
Master source shows **no** unintended modification. Nothing was committed (Step 12 honored).

## 15. Remaining Reproducibility Problems

1. **`rf_environment_viewer.grc`** still contains a machine-specific absolute path to `recordings/two_emitter_rf_multipath.dat` (a GUI file-selector parameter). Left as-is because it is a flowgraph GUI param and the phase forbade unnecessary flowgraph edits. On another machine the operator must re-select the file in GNU Radio Companion. (Minor; not part of the unit-test path.)
2. **Master regression must be run from `cognitive_ew_smart_scan`** (its rootdir puts `src/` on the path). This is the existing, documented master convention — unchanged, not a defect.
3. **RF suite requires RadioConda** (`numpy` + GNU Radio). The master app runs in its own venv. Two interpreters are expected and documented in the README.
4. **GNU_RF_ENV is not yet committed** — it is staged only as untracked; a human review + `git add` is the next deliberate step.
5. Legacy `.md` reports still reference the old `C:\HACKATHONS\SIH 2026\...` paths (historical records — left intentionally).

## 16. Exact Recommendation for the Next Phase — **Phase 3G (commit + reproducibility hardening)**

Recommended next steps (NOT started — stop condition respected):
1. **Review and commit the Phase 3F integration** in the master repo: `git add GNU_RF_ENV/ .gitattributes` then a human-reviewed commit. This makes the single root a committed, trackable, reproducible unit and fixes the original ZIP omission.
2. **Add a top-level `README.md` / "How to run"** explaining the two environments (master venv vs RadioConda) using the existing GNU_RF_ENV README content.
3. **Optionally parameterize `rf_environment_viewer.grc`** file path (or document the manual re-select) when a flowgraph edit is explicitly approved.
4. **Verify a fresh clone → `git clone` → run** end-to-end (once committed) to lock in reproducibility from any machine.
5. **Then, on separate instruction**, begin RF feature work (per Phase 3E/3F out-of-scope list: per-tune generator, scheduler integration, calibration, AoA, etc.).

Out-of-scope items (per-tune generator, scheduler/observation integration, ML training, calibration, AoA, stochastic generation, deployment) were NOT implemented. Stop condition respected.

# PHASE 3G — COMMIT + FRESH-CLONE REPRODUCIBILITY REPORT

## Phase: 3G (release integrity — commit + reproducibility verification)
## Date: 2026-09-04

---

### 1. Pre-commit status
```
On branch main — up to date with origin/main
Untracked files:  .gitattributes
                   GNU_RF_ENV/
git diff --stat:                       (empty — no tracked file modified)
git diff:                              (empty — clean)
git diff -- cognitive_ew_smart_scan/src/receiver/:  (EMPTY — receiver untouched)
git ls-files .../src/receiver/:
  sieve_receiver.py, models.py, adapter.py, __init__.py   (all authoritative files tracked)
```
Only the two new root items (`.gitattributes`, `GNU_RF_ENV/`) were untracked. No tracked production file had changed.

### 2. Exact staged file summary
`git add GNU_RF_ENV .gitattributes` staged **32 new files** (all `A`):

| Category | Files |
|---|---|
| Line-ending policy | `.gitattributes` |
| Docs | `GNU_RF_ENV/README.md`, `phase3a–3f_report.md` |
| RF source (A) | `scripts/dwell_orchestrator.py`, `iq_bridge.py`, `iq_to_pdw.py`, `frequency_context.py`, `jittered_rf_source.py`, `live_rf_environment.py`, `proof_phase3c.py`, `proof_phase3d.py`, `epy_block_0.py` |
| RF tests (B) | `tests/test_iq_to_pdw.py`, `test_frequency_context.py`, `test_iq_bridge.py`, `test_dwell_orchestrator.py`, `test_repo_integration.py` |
| Flowgraphs | `flowgraphs/*.grc` + `*.py` (live, viewer, step02/03/04, step09/09b) |
| Gitignore | `GNU_RF_ENV/.gitignore` |

No receivers, no scheduler, no perception, no training files staged. No `.git`, no `recordings/*.dat`, no `__pycache__`, no `.pyc` staged (all confirmed ignored by `git check-ignore`). 9477 insertions, 0 deletions.

### 3. Receiver / scheduler / perception / training untouched
Confirmed by empty `git diff` and the staged `--name-status` list: **zero** files under `cognitive_ew_smart_scan/src/receiver/`, processor/scheduler, or perception were modified, added, or deleted.

### 4. Commit hash
`30b3c0b`

### 5. Commit message
`Integrate GNU RF environment` (single commit, 32 files, +9477)

### 6. Post-commit status
```
On branch main — ahead of 'origin/main' by 1 commit.
nothing to commit, working tree clean
```
Working tree **clean** after commit. (`ahead by 1` = commit not yet pushable, see §17.)

### 7. Number of GNU_RF_ENV tracked files
**31** tracked files (all Phase 3A–3D source + flowgraphs + tests + reports + docs + `.gitignore`).

### 8. GNU RF tests after commit
```
radioconda python -m unittest discover -s GNU_RF_ENV\tests
Ran 103 tests — OK
```

### 9. Master regression tests after commit
```
python -m pytest -q tests/test_receiver.py              → 33 passed
python -m pytest -q tests/test_receiver_audit.py        →  5 passed
python -m pytest -q tests/test_receiver_integration.py  →  2 passed
python -m pytest -q tests/test_perception_adapters.py   →  6 passed
Total: 46 passed
```

### 10. Fresh clone path
`C:\Users\Indrani\AppData\Local\Temp\SIH2026_freshclone`
- Source: `git clone C:\HACKATHONS\SIH2026_Try2 <clone>` (git's own object transfer of the actual committed repo — NOT manual file copies).
- Clone HEAD: `30b3c0b Integrate GNU RF environment`.
- ⚠️ Remote push was **denied (403)** — see §17 for why local-path clone was used.

### 11. Fresh clone GNU RF test result
```
radioconda python -m unittest discover -s <clone>\GNU_RF_ENV\tests
Ran 103 tests — OK
```

### 12. Fresh clone master test result
```
python -m pytest -q tests/test_receiver.py test_receiver_audit.py test_receiver_integration.py test_perception_adapters.py
46 passed
```

### 13. Fresh clone multi-CWD test result
Run from `C:\` (outside the clone), explicit path `-s <clone>\GNU_RF_ENV\tests`:
```
Ran 103 tests — OK
```
Proves **both** repository-location independence and CWD independence.

### 14. Phase 3C proof result
`proof_phase3c.py` → `ALL PROOFS PASSED.`

### 15. Phase 3D proof result
`proof_phase3d.py` → `ALL PROOFS PASSED.`

### 16. Absolute-path search result
```
findstr C:\HACKATHONS in GNU_RF_ENV\.py source → 0 hits
.md reports            → 30 hits (historical records — allowed)
flowgraphs/*.grc       → 1 hit (rf_environment_viewer.grc recording selector — known)
```
No machine-specific paths in committed Python runtime source.

### 17. Remaining reproducibility issue(s)
1. **Cannot push to remote.** Authenticated account `sihhackathon4` has read access to `https://github.com/Promothesh-Chatterjee/SIH2026_Try2.git` (ls-remote worked) but **push was denied (403)**. `origin/main` is still at `d3eca70a`; the Phase 3G commit `30b3c0b` exists only **locally**. Consequently a plain `git clone <remote>` cannot yet fetch it. This phase verified the committed repository by cloning the local repo (which contains the commit). To make the exact Step 9 remote-clone path valid, either grant push rights to this account or push under an authorized account.
2. **`rf_environment_viewer.grc`** still has a machine-specific recording selector path (`C:\HACKATHONS\SIH 2026\GNU_RF_ENV\recordings\two_emitter_rf_multipath.dat`). Not modified this phase (per scope). On another machine the operator must re-select the file in GRC.
3. **Master tests must run from `cognitive_ew_smart_scan`** (rootdir puts `src/` on path) — existing documented convention, unchanged.
4. **RF suite requires RadioConda/GNU Radio**; master runs in its own venv. Two interpreters, documented.
5. **Six `.pt` checkpoints remain force-tracked** (~19MB) despite `.gitignore` (`checkpoints/`, `**/checkpoints/`). Left exactly as-is per Step 3.

### 18. Exact recommendation for Phase 3H
Recommended next phase — **"Push + remote reproducibility sign-off"** (no new RF features):
1. **Resolve remote write access** (or push via an authorized credential) so `main` reaches `origin` at `30b3c0b`.
2. Re-run the canonical Step 9–13 remote-clone verification against `https://github.com/Promothesh-Chatterjee/SIH2026_Try2.git` (the true "fresh clone from actual remote") once the commit is pushed; confirm 103 GNU RF + 46 master from that clone.
3. **Decide on a data policy for the six committed `.pt` checkpoints** — either stop force-tracking them (earn the ~19MB back and honor `.gitignore`) or keep them deliberately; document the choice. This is the single largest non-source bloat in the repo.
4. **Parameterize or document `rf_environment_viewer.grc`** recording selector (remove the hardcoded path) when a GRC edit is explicitly approved.
5. Optionally add a top-level `README.md` "How to run (two interpreters)" using the existing GNU_RF_ENV README content.
6. Then, on **separate instruction**, begin the out-of-scope RF features (per-tune generator, scheduler/observation integration, amplitude calibration, AoA, stochastic generation, training, deployment).

**STOP — no new RF features implemented in this phase.**

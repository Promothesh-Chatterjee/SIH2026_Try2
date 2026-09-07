# Phase 3E — Master + GNU_RF_ENV Integration Audit

## Audit Report

- **Phase:** 3E (audit + safe integration decision; NO feature work)
- **Date:** 2026-09-04
- **Auditor scope:** repository boundaries, reproducibility, git/line-endings, submission structure. No source changes to either repo.

---

## 1. Current Directory / Repository Structure

**Two separate git repositories (unchanged):**

```
C:\HACKATHONS\
├── SIH2026_Try2\          <- MASTER repo (git, branch: main, HEAD 2b4333b, 127 tracked files)
│   ├── .git\
│   ├── .gitignore
│   └── cognitive_ew_smart_scan\      <- Application root
│       ├── src\            (62 tracked files)  receiver/ perception/ models/ training/ environment/ ...
│       ├── tests\          (21 tracked files)
│       ├── frontend\       (27 tracked files, React SPA + PDW components)
│       ├── scripts\        (7 tracked files)
│       ├── notebooks\      (3 tracked .ipynb)
│       ├── configs\        (2 tracked files)
│       ├── checkpoints\    (6 tracked .pt files — force-added despite .gitignore)
│       ├── pyproject.toml, requirements.txt, Dockerfile, *.md, .env.example
└── SIH 2026\               <- worktree holding the SECOND repo (GNU_RF_ENV)
    ├── .git\
    └── GNU_RF_ENV\         <- GNU Radio repo (branch: main, HEAD e55d2cf, 6 tracked files)
        ├── scripts\        (Phase 3A–3D, untracked)
        ├── tests\          (Phase 3A–3D, untracked)
        ├── flowgraphs\     (.grc/.py)
        ├── recordings\     (5 × 16MB .dat IQ, ignored)
        ├── metadata\       (0 files)
        ├── scenarios\      (0 files)
        └── README.md, .gitignore
```

Note: The GNU Radio repo lives at `C:\HACKATHONS\SIH 2026\GNU_RF_ENV` — i.e. it is nested *inside the workspace root folder* `C:\HACKATHONS\SIH 2026`, whose `.git` is at `C:\HACKATHONS\SIH 2026\.git`.

## 2. Exact Location of All Phase 3A–3D Files (verified present)

| File | Size (bytes) |
|------|-------------|
| `GNU_RF_ENV\scripts\iq_to_pdw.py` | 14492 |
| `GNU_RF_ENV\scripts\frequency_context.py` | 6025 |
| `GNU_RF_ENV\scripts\iq_bridge.py` | 5863 |
| `GNU_RF_ENV\scripts\dwell_orchestrator.py` | 7482 |
| `GNU_RF_ENV\tests\test_iq_to_pdw.py` | 19433 |
| `GNU_RF_ENV\tests\test_frequency_context.py` | 6619 |
| `GNU_RF_ENV\tests\test_iq_bridge.py` | 13385 |
| `GNU_RF_ENV\tests\test_dwell_orchestrator.py` | 16982 |
| `GNU_RF_ENV\scripts\proof_phase3c.py` | 1553 |
| `GNU_RF_ENV\scripts\proof_phase3d.py` | 2480 |

Authoritative master receiver verified at `SIH2026_Try2\cognitive_ew_smart_scan\src\receiver\`:
`sieve_receiver.py` (24800 B), `models.py` (1954 B), `adapter.py` (866 B), `__init__.py` (704 B). Not modified.

## 3. Cross-Repository Import Dependencies

| Script | Depends on master? | Mechanism |
|--------|--------------------|-----------|
| `iq_to_pdw.py` | **No** | `numpy` only (requires RadioConda). No master import. |
| `frequency_context.py` | **No** | Pure stdlib (`math`, `typing`). Zero deps. |
| `iq_bridge.py` | **No** (direct) | imports sibling `frequency_context`. |
| `dwell_orchestrator.py` | **YES** | `sys.path.insert(0, r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src")` then `from receiver import SieveReceiver`. **Hardcoded absolute Windows path.** |
| `test_frequency_context.py` | **YES** | hardcoded `_MASTER_SRC = r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src"` |
| `test_iq_bridge.py` | **YES** | same hardcoded `_MASTER_SRC` |
| `test_dwell_orchestrator.py` | **YES** | hardcoded `sys.path.insert(0, r"C:\HACKATHONS\SIH2026_Try2\...\src")` |

**Reproducibility assessment:**
- The **only** cross-repo dependency is on the authoritative `SieveReceiver`, reached exclusively via a **hardcoded absolute Windows path** (`C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src`). This is the packaging defect.
- Mechanism is CWD-independent (verified: tests + proofs run from both GNU_RF_ENV and SIH2026_Try2 CWDs, and with `PYTHONPATH` cleared).
- GNU RF suite requires **RadioConda** python because `iq_to_pdw.py`/`test_iq_to_pdw.py` import `numpy` and the RadioConda interpreter is the one with numpy (verified: radioconda `python.exe`).
- No reliance on environment variables, no `.env`, no relative path to master (path is absolute).
- **The absolute path breaks when the master repo is not at `C:\HACKATHONS\SIH2026_Try2\...`** (e.g., on another machine or after submission extraction). This is the key reproducibility risk.

## 4. GNU_RF_ENV Test Results (RadioConda)

```
C:\Users\Indrani\radioconda\python.exe -m unittest discover -s tests
Ran 97 tests in 0.121s — OK
```

97 tests passing — exactly as expected (20 Phase 3A + 20 Phase 3B + 25 Phase 3C + 32 Phase 3D), and this count was NOT altered to "match". Also re-run successfully from the master repo CWD (97 OK).

## 5. Master Receiver / Perception Regression Results

```
python -m pytest -q tests/test_receiver.py            → 33 passed
python -m pytest -q tests/test_receiver_audit.py      →  5 passed
python -m pytest -q tests/test_receiver_integration.py→  2 passed
python -m pytest -q tests/test_perception_adapters.py →  6 passed
Total: 46 passed
```

## 6. Git Status of Both Repositories

**Master (`SIH2026_Try2`):** `git status` — **CLEAN** (no changes; HEAD `2b4333b Minor fixes`; 127 tracked files; branch `main`).

**GNU_RF_ENV (`SIH 2026\GNU_RF_ENV`):**
```
 M README.md
 M flowgraphs/step02_single_tone.py          <- pre-existing edits (from Phase 0/2, not this phase)
?? .gitignore
?? flowgraphs/rf_environment_live.grc
?? flowgraphs/rf_environment_viewer.grc
?? flowgraphs/rf_environment_viewer.py
?? flowgraphs/step09b_jittered_emitter.grc
?? scripts/     (Phase 3A–3D)
?? tests/       (Phase 3A–3D)
```
No tracked file in GNU_RF_ENV was modified during this phase. All Phase 3 scripts/tests remain untracked (never committed).

## 7. Explanation of the ZIP Omission of GNU_RF_ENV

Two submission archives were found:
- `C:\HACKATHONS\SIH 2026.zip` (837MB, dated 2026-08-29) — contains only `SIH 2026/.git` + worktree; **no GNU_RF_ENV at all** (predates it).
- `C:\HACKATHONS\SIH2026_Try2.zip` (60MB, dated 2026-09-04) — contains `SIH2026_Try2/` with `cognitive_ew_smart_scan/` **and the `.git/` directory inside**, and **NO `GNU_RF_ENV/`**.

**Root cause:** GNU_RF_ENV is a *separate git repository* living in a different worktree (`C:\HACKATHONS\SIH 2026\GNU_RF_ENV`), which is a **sibling** of `SIH2026_Try2` at the filesystem level — not a child. The submission ZIP was created from `SIH2026_Try2` alone, so GNU_RF_ENV was never included. The only "PDW" matches in the ZIP are `frontend/src/components/PDWFeed.jsx` and `PDWScatter.jsx` (React UI), **not** the RF pipeline.

This is a genuine reproducibility gap: the submitted archive cannot run or demonstrate the Phase 3A–3D RF pipeline.

## 8. Explanation of the Apparent Line-Ending Git Modifications

**Findings:**
- Live master repo: `core.autocrlf = true`, `core.filemode = false`, **no `core.eol`**, no `.gitattributes`.
- Working tree files are CRLF (verified: sieve_receiver.py = 572 CRLF / 0 bare LF). Index/HEAD stores LF (`core.autocrlf=true` normalizes CRLF→LF).
- Current `git diff` is **empty**; working file hash-object equals HEAD blob (`b4c4f0...`). **The live repo is genuinely clean right now.**
- The shipped ZIP's embedded `.git/config` sets `filemode=false` but does **NOT** contain `core.autocrlf` (the live repo added it locally). The ZIP's working tree is CRLF (verified 572/0).

**Explanation:** The ZIP bundles the `.git` directory with a config lacking `core.autocrlf`, while shipping CRLF working-tree files. When such an archive is extracted and inspected on a system where git normalizes differently (or with autocrlf unset/false), git compares the CRLF bytes on disk against the LF blobs in the index and reports **almost every text file as modified** — even though bytes "appear substantively unchanged" and would re-clean under `core.autocrlf=true`. This is exactly the churn described in the audit. The mismatch is a **line-ending-normalization / config-state artifact**, not substantive source edits. No broad rewrite was performed (phase scope forbids it).

## 9. Recommended Repository Structure — **OPTION A**

```
SIH2026_Try2/                      <- single master repo
├── cognitive_ew_smart_scan/       <- existing application (unchanged, authoritative receiver)
└── GNU_RF_ENV/                    <- moved IN as a top-level directory
    ├── scripts/  tests/  flowgraphs/  recordings/  README.md
```

**Rationale (based on actual code, not aesthetics):**
- The extreme coupling: `dwell_orchestrator.py` hardcodes `C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src`. Under Option A, this becomes `relative` or a clean `PYTHONPATH` from a single repo root — eliminating the only cross-repo path dependency.
- Option A ships both the app **and** the RF pipeline in one reproducible unit, directly fixing the ZIP omission (Section 7) while preserving master git history (Option A is additive: new top-level dir; zero changes to the 127 tracked files).
- Option B puts GNU Radio *inside* `cognitive_ew_smart_scan\gnuradio\`, which blends two unrelated subsystems into one Python package namespace and complicates the app's packaging (`pyproject.toml`), for no benefit. Rejected.
- Option C (keep separate) leaves the hardcoded absolute path and the ZIP omission unsolved — fails the reproducibility goal. Rejected.

**Important caveat for Option A:** GNU_RF_ENV depends on **RadioConda** python (for numpy) and real GNU Radio for live flowgraphs, while the master app uses its own venv. Option A keeps GNU_RF_ENV as a self-contained directory that is still run with RadioConda — it does not force GNU Radio into the master venv. The master ML system can still be deployed/run **without** GNU Radio installed (GNU Radio is only needed for the RF-simulation half). Duplication is avoided: the receiver is never copied; GNU_RF_ENV imports it from the single root.

## 10. Recommended Submission Structure

Ship **one** root folder (Option A) with the GNU_RF_ENV merged in, zipped without the `.git` directories:
```
SIH2026_Try2_Submission/
├── README.md                        (explains master vs RF, how to run each)
├── cognitive_ew_smart_scan/         (MASTER: app; keep its .git OUT of submission or a single clean .git)
└── GNU_RF_ENV/                      (RF pipeline + flowgraphs)
```
- **A (required to run master ML):** `cognitive_ew_smart_scan/src`, `tests`, `configs`, `scripts`, `pyproject.toml`, `requirements.txt`, `README.md`, `Architecture.md`, `Design.md`, `PRD.md`, `QuickStart`, `.env.example`.
- **B (required to run RF simulation):** `GNU_RF_ENV/scripts` (Phase 3A–3D), `GNU_RF_ENV/tests`, `GNU_RF_ENV/flowgraphs`, `GNU_RF_ENV/README.md`, plus a note that RadioConda + GNU Radio is required.
- **C (optional dev material):** `frontend/` (source + README; `node_modules`/`dist` excluded), `notebooks/`, `configs/`.
- **D (should NOT ship):** all `.git/` directories, caches, checkpoints, recordings.

**Fix the hardcoded path before shipping** so extraction on any machine works (see Section 12).

## 11. Files That Should Be Excluded From Submission

| Category | Location | Classification |
|----------|----------|----------------|
| `.git/` directories | both repos (incl. inside shipped ZIP) | EXCLUDE |
| `__pycache__/` (10 dirs) + `.pyc` (51 files) | master src/tests, GNU_RF_ENV | EXCLUDE |
| `.pytest_cache/` | master | EXCLUDE |
| `checkpoints/*.pt` (6 files, ~19MB, tracked) | master | REVIEW (see below) |
| `recordings/*.dat` (5×16MB IQ) | GNU_RF_ENV | EXCLUDE (already gitignored) |
| `frontend/node_modules` | master/frontend | EXCLUDE (none present) |
| `frontend/dist`, `build` | master/frontend | EXCLUDE (none present) |
| `*.pyc`, `.venv/`, `.env` | both | EXCLUDE |
| Dockerfile | master | KEEP (deployment reference) |
| notebooks/ .ipynb | master | OPTIONAL |
| flowgraphs/*.grc */**.py | GNU_RF_ENV | KEEP (required for RF) |

**Checkpoints REVIEW:** These 6 `.pt` files are force-tracked (committed despite `.gitignore`). They are ~19MB of model weights — large and machine-specific. They should likely be **EXCLUDED** from a source submission (they'll be RE-derived by training), but they are currently within git history so they're in the ZIP. Decide explicitly: exclude for a clean source submission, or keep if demonstrating a trained model is desired. No deletion performed this phase.

## 12. Genuine Defects Found

1. **[Packaging] Hardcoded absolute imports to the master receiver.** `dwell_orchestrator.py` (L36) and the two bridge/dwell test files insert `r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src"` directly. This is the single reproducibility blocker — it only works on this machine at this path. It is NOT a master-repo defect; a fix belongs in the GNU_RF_ENV scripts/tests (make the path derived from a repo-relative or env-driven root). This is a GNU_RF_ENV-side packaging defect, safe to fix in the RF-side code — **no master source modification needed.**
2. **[Submission] GNU_RF_ENV omitted from the archive** (Section 7) — the RF pipeline is not demonstrable from the current ZIP.
3. **[Housekeeping] 6 model checkpoints force-committed** into the master repo (bloat + noise). Not a code defect; a submission-hygiene decision.
4. No genuine defect found that prevents the master system from executing — master runs 46/46 and is clean. No STOP-before-modify situation was hit for master source.

## 13. Were Any Source Changes Required?

**No.** The master repository required **zero** source changes (still clean, HEAD `2b4333b`, 46/46 tests). No GNU_RF_ENV tracked file was modified during this audit. Only new untracked Phase 3 files already existed; nothing was deleted, reset, or rewritten. Per instructions, no line-ending rewrite and no git reset were performed.

The only change recommended (per Section 12 defect #1) is a future RF-side fix to remove the hardcoded absolute path — intentionally deferred, as this phase is audit-only.

## 14. Exact Next Phase Recommendation — **Phase 3F (safe integration, in this order)**

Before any later RF features, do the minimal safe integration to make the pipeline reproducible:

1. **Single-root merge (Option A):** copy/move `C:\HACKATHONS\SIH 2026\GNU_RF_ENV` into `SIH2026_Try2\GNU_RF_ENV` so master and RF share one repo root.
2. **Remove the hardcoded absolute path** in the GNU RF scripts/tests: replace the `sys.path.insert(0, r"C:\...\SIH2026_Try2\cognitive_ew_smart_scan\src")` with a **repo-relative** resolution (e.g., `Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan"/ "src"`) so it works from any checkout location.
3. **Re-run both suites** after the move: GNU RF (RadioConda, 97) and master (venv, 46) to prove the integration is byte-for-byte behavior-preserving.
4. **Fix the submission:** produce a single clean ZIP (no `.git/`, no caches, no recordings; exclude or keep checkpoints per decision) that contains both `cognitive_ew_smart_scan` and `GNU_RF_ENV`, and add a README recording the RadioConda (RF) vs venv (master) interpreters.
5. Add `core.autocrlf`/`.gitattributes` guidance so future extractions don't reintroduce the phantom "all files modified" state.

**Explicitly out of scope for Phase 3F (do NOT implement without further instruction):** per-tune GNU Radio generator, dwell sweep enhancement, scheduler integration, ReceiverObservation→scheduler, ML training integration, reward changes, amplitude calibration, AoA estimation, stochastic scenario generation, deployment changes.

Phase 3F is NOT started. Stop condition respected.

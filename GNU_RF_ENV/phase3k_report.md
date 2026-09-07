# PHASE 3K -- AUDIT CHECKPOINT INTEGRITY & CLASSIFY THE SINGLE HDBSCAN TEST FAILURE

## Phase: 3K (release-integrity / checkpoint + test audit)
## Date: 2026-09-04
## Result: **ROOT CHECKPOINT WRITER IDENTIFIED AND CHECKPOINTS RESTORED VIA GIT (NO reset --hard); HDBSCAN FAILURE CLASSIFIED AS A DETERMINISTIC, TRACKED-IN-HEAD PRE-EXISTING REGRESSION (source=04d6485, test=b1f90a0). NO SOURCE CODE CHANGED.**

---

### 1. Step 1 -- Checkpoint hash capture (pre-restore)
Tracked root checkpoints were showing as **modified** in the working tree (same byte size, different content):

| File | Worktree hash | HEAD hash | byte size |
|------|---------------|-----------|-----------|
| `cognitive_ew_smart_scan/checkpoints/best.pt` | `1115a9095d7cfc50cc650fe7328070df3510ab62` | `ebc80c2651a590b35ba15c0d8e588828a2794b0e` | 3,259,969 |
| `cognitive_ew_smart_scan/checkpoints/final.pt` | `ea734bcf3fbe4dff970f4ac157f8bc7445cefed8` | `5195460241b00a45656cffce07df9150dc828d5e` | 3,259,391 |

Note: an initial attempt to extract the HEAD blob via `git cat-file blob > file` produced a 6,560,670-byte file -- a **PowerShell binary redirection artifact**. Authoritative size (`git cat-file -s HEAD:...` = 3,259,969) matches the worktree exactly, so the earlier "larger HEAD blob" observation was wrong. Same size, different content = modified.

### 2. Step 2 -- Root `checkpoints/best.pt` + `checkpoints/final.pt` writer identified
The root-git-ignored (`**/checkpoints/`) artifacts are overwritten by the **training smoke test**:
`tests/test_synthetic_training.py::test_safe_training_entrypoint_runs`
  -> `train_deinterleaver_safe()`  (train_deinterleaver.py L565-576)
  -> loads `configs/training_config.yaml` whose `output_dir: "checkpoints"` (**bare** root, L2)
  -> `save_state(model, output_dir / "best.pt")` (L550)   => writes **root checkpoints/best.pt**
  -> `torch.save(..., .../epochNNN.pt)` (L557-558, quick_smoke)
  -> `torch.save(..., output_dir / "final.pt")` (L561)    => writes **root checkpoints/final.pt**

The default `output_dir` in the function is `checkpoints/deinterleaver` (subdir), which is why the subdir training paths were originally ruled out; the **config override to bare `checkpoints`** is the root-cause writer. Every full `pytest tests/` run that executes `test_synthetic_training.py` therefore re-writes the tracked root checkpoints. This also explains the pre-tracked `checkpoints/epoch005/010/015/020.pt` in root and the newly-appeared untracked `epoch001.pt` + `normalization_stats.json`.

### 3. Steps 3-4 -- Restore ONLY the two tracked checkpoints via git (no reset --hard)
```
git restore --source=HEAD -- cognitive_ew_smart_scan/checkpoints/best.pt cognitive_ew_smart_scan/checkpoints/final.pt
```
Exit 0. Verified hashes now equal HEAD:
- best.pt  worktree=`ebc80c2651a...` = HEAD `ebc80c2651a...`  (match)
- final.pt worktree=`5195460241b...` = HEAD `5195460241b...`  (match)
`git status --short` shows **no checkpoint modifications** (only the 4 untracked phase reports remain).

### 4. Steps 5-6 -- HDBSCAN failure reproducibility (NO code change, 3 runs)
Target: `tests/test_windowed_deinterleave.py::WindowedClusterTests::test_clusters_synthetic`
```
run #1: pairwise_f1 = 0.5539135194307607  (FAIL, < 0.9)
run #2: pairwise_f1 = 0.5539135194307607  (FAIL)
run #3: pairwise_f1 = 0.5539135194307607  (FAIL)
```
Result is **bit-for-bit identical across all 3 runs** -> fully DETERMINISTIC (rules out seed/random sensitivity).

### 5. Steps 7-9 -- Test, algorithm, and environment facts
- Test (unchanged since `b1f90a0`): 3 clean separable clusters, 120 pulses each (360 total), centers `[-1,-1,-1,0,0,0]`,`[1,-1,1,0,0,0]`,`[-1,1,1,0,0,0]`, scale 0.05; fixed seeds (`torch.manual_seed(0)`, `np.random.default_rng(1)`); `min_cluster_size=8, min_samples=3`, `window_size=100, stride=50`; asserts `pairwise_f1 > 0.9`, `n_clusters >= 2`.
- Algorithm (`src/models/deinterleaver.py`, unchanged since `04d6485`): windowed embedding via random-init transformer -> per-window HDBSCAN (euclidean, eom) -> cross-window reconciliation by shared-pulse majority merge (`reconcile_overlap_frac=0.5`). Warning observed: "HDBSCAN assigned all pulses to noise" (from a window).
- Environment: Python 3.14.6, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, hdbscan 0.8.44, torch 2.14.0+cpu.

### 6. Step 8 -- Regression history (root cause)
- `tests/test_windowed_deinterleave.py`: unchanged since `b1f90a0` (blob identical at HEAD).
- `src/models/deinterleaver.py`: unchanged since `04d6485` (blob `5eb49c8...` identical at `04d6485` and HEAD).
- `04d6485` (after the test was written in `b1f90a0`) **changed `embed_pdws_windowed`** window-embedding accumulation from a simple per-owner-sum to an **averaged embedding** (`out = out / owner_count`), altering the separability the test's fixed-seed scenario depended on.
- NOT changed by `d3eca70` (its only "deinterleaver" path is `train_deinterleaver.py`, unrelated), NOT changed by the GNU RF integration `30b3c0b`, NOT changed by the merge `2012fa2`. This is a **committed-in-HEAD, deterministic state** existing before Phase 3G's work.

### 7. Step 10 -- Failure classification
**Category: DETERMINISTIC PRE-EXISTING REGRESSION, TRACKED IN HEAD.**
The failing state is reproducible bit-for-bit, originates from a tracked-in-HEAD algorithm change (`04d6485`) interacting with a tracked-in-HEAD fixed-seed test (`b1f90a0`), is **not** induced by this phase, not induced by the GNU RF merge, and not a random/threshold-only flake. The single surviving failure in the 148/1/5 full-suite result.

### 8. Step 16 -- Checkpoint state after single-test runs
Re-checked after all HDBSCAN runs: `best.pt`/`final.pt` still match HEAD (single HDBSCAN test does NOT run training and does not write checkpoints; only the full-suite training smoke test does). Working tree clean except 4 untracked phase reports.

### 9. Phase 3K out-of-scope (NOT done, by permission)
- Did NOT modify/weaken the HDBSCAN test or `deinterleaver.py`.
- Did NOT change any dependency version.
- Did NOT `git reset --hard`.
- Did NOT commit or push.
- Did NOT auto-fix the `test_clusters_synthetic` failure (classified only).
- Did NOT touch GNU RF env/suites (103/103 unchanged), master suites 46/46, or proofs 3C/3D (verified PASS in prior phases; unaffected by this phase's read-only checkpoint work).

import json, math, statistics, sys

TEL = r"C:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2\cognitive_ew_smart_scan\runs\20260906-121139-7ae97d\telemetry.jsonl"
lines = [json.loads(l) for l in open(TEL)]
ep = [l for l in lines if l["type"] == "episode"]
val = [l for l in lines if l["type"] == "val"]


def within(step, lo, hi):
    return lo <= step <= hi


bounds = [(0, 10000), (10000, 25000), (25000, 50000), (50000, 75000),
          (75000, 100000), (100000, 125000), (125000, 150000), (150000, 153000)]

print("=== RUN 01 FORENSIC ANALYSIS ===")
print("run_id=20260906-121139-7ae97d  episodes=%d  val_records=%d" % (len(ep), len(val)))
print("NOTE: telemetry logs band_priorities (MoE PRIOR), NOT selected actions.")
print("Only pd/pfa/hits/reward/epsilon are decision signals. No coverage/discovery/")
print("intercept-time/action-entropy are logged. band-selection metrics are UNAVAILABLE.\n")


def analyze(rows):
    if not rows:
        return None
    rw = [r["ep_reward"] for r in rows]
    pd = [r["pd"] for r in rows]
    hits = [r["ep_hits"] for r in rows]
    eps = [r["epsilon"] for r in rows]
    # band prior concentration (prior, not selection)
    top1 = []; top3 = []; top5 = []; hn = []
    for r in rows:
        bp = r["band_priorities"]
        s = sum(bp)
        pv = [x / s for x in bp] if s > 0 else [0] * len(bp)
        H = -sum(p * math.log(p) if p > 0 else 0 for p in pv)
        hn.append(H / math.log(len(bp)) if len(bp) > 1 else 0.0)
        order = sorted(range(len(bp)), key=lambda i: bp[i], reverse=True)
        top1.append(pv[order[0]])
        top3.append(sum(pv[order[i]] for i in range(3)))
        top5.append(sum(pv[order[i]] for i in range(5)))
    return dict(
        n=len(rows),
        rw_mu=statistics.mean(rw), rw_med=statistics.median(rw), rw_std=statistics.stdev(rw),
        pd_mu=statistics.mean(pd), pd_min=min(pd),
        hits_mu=statistics.mean(hits), hits_min=min(hits), hits_max=max(hits),
        eps0=eps[0], eps1=eps[-1],
        hn_mu=statistics.mean(hn), top1=statistics.mean(top1), top3=statistics.mean(top3),
        top5=statistics.mean(top5),
    )


print(f"{'window':<20}{'n':>4}{'rew_mu':>9}{'rew_med':>9}{'rew_std':>9}{'Pd_mu':>7}{'Pd_min':>7}{'hits':>7}{'Hnorm':>7}{'top1':>6}{'top3':>6}{'top5':>6}{'eps':>10}")
for lo, hi in bounds:
    rows = [r for r in ep if within(r["step"], lo, hi)]
    a = analyze(rows)
    if a is None:
        print(f"{lo}-{hi}: <no episodes>")
        continue
    print(f"{lo}-{hi:<14}{a['n']:>4}{a['rw_mu']:>9.1f}{a['rw_med']:>9.1f}{a['rw_std']:>9.1f}"
          f"{a['pd_mu']:>7.3f}{a['pd_min']:>7.2f}{a['hits_mu']:>7.1f}{a['hn_mu']:>7.3f}"
          f"{a['top1']:>6.2f}{a['top3']:>6.2f}{a['top5']:>6.2f}{a['eps0']:>5.2f}-{a['eps1']:.2f}")

print("\n=== VALIDATION REWARD (RUN-01, on TRAIN split - see isolation note) ===")
for lo, hi in bounds:
    rows = [r for r in val if within(r["step"], lo, hi)]
    if rows:
        v = [r["val_reward"] for r in rows]
        print(f"{lo}-{hi:<14} n={len(rows):>2} val_mu={statistics.mean(v):>9.1f} samples={[round(x) for x in v]}")

print("\n=== FULL-RUN EPISODE-LEVEL DIVERSITY ===")
# highest/lowest hit episodes
byhits = sorted(ep, key=lambda r: r["ep_hits"])
print("episodes with ep_hits == 0:", [r["step"] for r in byhits if r["ep_hits"] == 0])
print("episodes with pd < 1.0:", [(r["step"], round(r["pd"], 3)) for r in ep if r["pd"] < 1.0])
print("lowest hits:", [(r["step"], r["ep_hits"], round(r["pd"], 2)) for r in byhits[:5]])
print("highest hits:", [(r["step"], r["ep_hits"], round(r["pd"], 2)) for r in byhits[-5:]])

print("\n=== BAND PRIOR: GLOBAL CONCENTRATION (across all 153 episodes, prior only) ===")
all_bp = []
for r in ep:
    bp = r["band_priorities"]
    s = sum(bp)
    all_bp.append([x / s for x in bp] if s > 0 else [0] * len(bp))
mean_prior = [statistics.mean(p[i] for p in all_bp) for i in range(36)]
srt = sorted(range(36), key=lambda i: mean_prior[i], reverse=True)
print("top-5 bands by mean prior share (band_idx: mean_prior):",
      [(i, round(mean_prior[i], 4)) for i in srt[:5]])
print("bottom-5 bands:", [(i, round(mean_prior[i], 4)) for i in srt[-5:]])

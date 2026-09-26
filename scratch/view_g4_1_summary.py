import json

with open("reports/g4_1_recovery_characterization_manifest.json") as f:
    m = json.load(f)

for model_key in ["gate25_baseline", "gate50_candidate", "g4_b", "g4_c"]:
    print(f"\n=== {model_key.upper()} SCENARIO BREAKDOWN ===")
    for scen, sdata in m["models"][model_key]["scenarios"].items():
        print(f"{scen:<12} | IR={sdata['ir_decision']:5.1f}% | Pd={sdata['pd']:5.1f}% | Hits/ms={sdata['gross_hits_per_ms']:.3f} | MaxRun={sdata['max_consecutive_band_run']:3d} | Modes={sdata['mode_counts']}")

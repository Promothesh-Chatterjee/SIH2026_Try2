import json

with open("checkpoints/scheduler/gate_100000_repaired_canonical_report.json") as f:
    d_ar = json.load(f)["summary"]["full_moe"]
with open("results/canonical_gate100k_t1.json") as f:
    dt1 = json.load(f)["summary"]["t1_predictive_utility"]
with open("results/canonical_gate105k_t1.json") as f:
    d105 = json.load(f)["summary"]["t1_predictive_utility"]
with open("results/canonical_gate110k_t1.json") as f:
    d110 = json.load(f)["summary"]["t1_predictive_utility"]

print("=" * 115)
print("CANONICAL 10-SCENARIO GATE COMPARISON: Baseline (AR) vs T1 (100k) vs 105k vs 110k")
print("=" * 115)
print(f"{'Metric':<22} | {'Gate-100k-AR':<14} | {'T1 Frozen':<12} | {'105k':<12} | {'110k':<12} | {'110k vs Baseline':<18}")
print("-" * 115)
print(f"{'Interception Rate':<22} | {d_ar['mean_intercept_rate']*100:6.2f}%        | {dt1['mean_intercept_rate']*100:6.2f}%     | {d105['mean_intercept_rate']*100:6.2f}%     | {d110['mean_intercept_rate']*100:6.2f}%     | {d110['mean_intercept_rate']*100 - d_ar['mean_intercept_rate']*100:+6.2f}%")
print(f"{'Raw Pulse Hits':<22} | {d_ar['total_hits']:6d}          | {dt1['total_hits']:6d}       | {d105['total_hits']:6d}       | {d110['total_hits']:6d}       | {d110['total_hits'] - d_ar['total_hits']:+6d} hits")
med_ar = d_ar.get('median_latency_us', 204.9)
mean_ar = d_ar.get('mean_latency_us', d_ar.get('mean_intercept_time_us', 204.9))
p90_ar = d_ar.get('p90_latency_us', 536.0)

print(f"{'Median Latency':<22} | {med_ar:6.1f} us       | {dt1['median_latency_us']:6.1f} us    | {d105['median_latency_us']:6.1f} us    | {d110['median_latency_us']:6.1f} us    | {d110['median_latency_us'] - med_ar:+6.1f} us")
print(f"{'Mean Latency':<22} | {mean_ar:6.1f} us       | {dt1['mean_latency_us']:6.1f} us    | {d105['mean_latency_us']:6.1f} us    | {d110['mean_latency_us']:6.1f} us    | {d110['mean_latency_us'] - mean_ar:+6.1f} us")
print(f"{'P90 Latency':<22} | {p90_ar:6.1f} us       | {dt1['p90_latency_us']:6.1f} us    | {d105['p90_latency_us']:6.1f} us    | {d110['p90_latency_us']:6.1f} us    | {d110['p90_latency_us'] - p90_ar:+6.1f} us")

print(f"{'Discovery Rate':<22} | {d_ar['mean_discovery_rate']*100:6.1f}%        | {dt1['mean_discovery_rate']*100:6.1f}%     | {d105['mean_discovery_rate']*100:6.1f}%     | {d110['mean_discovery_rate']*100:6.1f}%     | {d110['mean_discovery_rate']*100 - d_ar['mean_discovery_rate']*100:+6.1f}%")
print(f"{'Empty Band Escape':<22} | {d_ar['mean_empty_escape_rate']*100:6.1f}%        | {dt1['mean_empty_escape_rate']*100:6.1f}%     | {d105['mean_empty_escape_rate']*100:6.1f}%     | {d110['mean_empty_escape_rate']*100:6.1f}%     | {d110['mean_empty_escape_rate']*100 - d_ar['mean_empty_escape_rate']*100:+6.1f}%")
print("-" * 115)
print("\nPer-Scenario Hit Breakdown (Baseline AR -> T1 -> 105k -> 110k):")
for s110, s105, st1, sar in zip(d110['scenarios'], d105['scenarios'], dt1['scenarios'], d_ar['scenarios']):
    sid = s110['scenario_id']
    h110 = s110['hits']
    h105 = s105['hits']
    ht1 = st1['hits']
    har = sar['hits']
    print(f"  {sid:<12}: AR={har:4d} -> T1={ht1:4d} -> 105k={h105:4d} -> 110k={h110:4d} (110k vs AR: {h110-har:+4d}) | 110k Med Lat = {s110['median_latency_us']:5.1f} us")

with open('results/agile_t1_final_500steps.json') as f:
    at1 = json.load(f)['t1_predictive_utility']
with open('results/agile_gate105k_t1.json') as f:
    a105 = json.load(f)['t1_predictive_utility']
with open('results/agile_gate110k_t1.json') as f:
    a110 = json.load(f)['t1_predictive_utility']

print('\n' + '=' * 95)
print('AGILE BENCHMARK (AG-01 to AG-10) COMPARISON: T1 vs 105k vs 110k')
print('=' * 95)
print(f"{'Scenario':<8} | {'T1 Pd':<10} | {'105k Pd':<10} | {'110k Pd':<10} | {'110k vs 105k':<14} | {'110k Med Lat':<14}")
print('-' * 95)
for sc in sorted(a110.keys()):
    p_t1 = at1[sc]['mean_ir']
    p_105 = a105[sc]['mean_ir']
    p_110 = a110[sc]['mean_ir']
    lat_110 = a110[sc]['median_latency']
    print(f"{sc:<8} | {p_t1:6.2f}%    | {p_105:6.2f}%    | {p_110:6.2f}%    | {p_110-p_105:+7.2f}% pp    | {lat_110:6.1f} us")
print('=' * 95)

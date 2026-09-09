from collections import Counter
import json
from pathlib import Path
from src.evaluation.miss_classifier import HierarchicalMissClassifier
from src.evaluation.miss_logger import DecisionRecord

def analyze_all_misses(telemetry_dir: str = 'results/post110k/telemetry') -> None:
    p_dir = Path(telemetry_dir)
    files = list(p_dir.glob('*.json'))
    if not files:
        print('No telemetry files found in', telemetry_dir)
        return

    primary_counts = Counter()
    secondary_counts = Counter()
    scenario_primary = {}
    total_misses = 0

    for f in files:
        with open(f, encoding='utf-8') as jf:
            d = json.load(jf)
        scen = d.get('scenario_id', f.stem)
        scenario_primary[scen] = Counter()
        for dec_dict in d.get('decisions', []):
            rec = DecisionRecord(**dec_dict)
            if not rec.hit:
                attr = HierarchicalMissClassifier.classify(rec)
                if attr:
                    total_misses += 1
                    primary_counts[attr.primary_cause] += 1
                    scenario_primary[scen][attr.primary_cause] += 1
                    for sc in attr.secondary_causes:
                        secondary_counts[sc] += 1

    actions_map = {
        'NEXT_BAND_PREDICTION_ERROR': 'Predictor transition matrix adaptation',
        'UNPREDICTED_AGILE_HOP': 'Agile predictor sensitivity boost',
        'ETA_ERROR': 'Temporal dwell duration adjustment',
        'DRQN_RANKING_ERROR': 'Action-conditioned utility weight tuning',
        'DENSE_CONTENTION_MISS': 'Spatial AoA filtering and multi-receiver arbitration',
        'COGNITIVE_EXPLORATION_MISS': 'Exploration entropy regulation',
        'DWELL_MODE_MISMATCH': 'Adaptive dwell mode scaling',
        'FORCED_ESCAPE_MISS': 'Consecutive empty threshold damping',
        'SPATIAL_PRIORITIZATION_MISS': 'AoA spatial tracking layer',
        'STALE_TRACK_ERROR': 'Track decay rate optimization',
    }

    md_lines = [
        '# Miss Root Cause Summary & Attribution Analysis',
        '',
        f'Total Misses Analyzed across Scenarios: **{total_misses}**',
        '',
        '| Root Cause (Primary) | Count | % of Misses | Main Scenarios | Corrective Action |',
        '| :--- | :--- | :--- | :--- | :--- |',
    ]

    for cause, count in primary_counts.most_common():
        pct = (count / max(1, total_misses)) * 100
        scens = [sc for sc, cnts in scenario_primary.items() if cnts.get(cause, 0) > 0]
        scen_str = ', '.join(scens[:3]) if scens else 'All'
        act = actions_map.get(cause, 'Investigate telemetry')
        md_lines.append(f'| {cause} | {count} | {pct:.1f}% | {scen_str} | {act} |')

    md_lines.extend([
        '',
        '## Secondary Contributing Causes',
        '',
        '| Secondary Factor | Occurrences | Frequency |',
        '| :--- | :--- | :--- |',
    ])
    for sc, count in secondary_counts.most_common():
        pct = (count / max(1, total_misses)) * 100
        md_lines.append(f'| {sc} | {count} | {pct:.1f}% |')

    out_md = Path('results/post110k/MISS_ROOT_CAUSE_SUMMARY.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Generated results/post110k/MISS_ROOT_CAUSE_SUMMARY.md successfully')

if __name__ == '__main__':
    analyze_all_misses()

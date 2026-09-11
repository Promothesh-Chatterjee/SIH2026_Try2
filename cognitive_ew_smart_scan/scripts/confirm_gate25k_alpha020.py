import copy
import json
import logging
from pathlib import Path
import numpy as np
import torch
import yaml

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.training.staged_gate_evaluator import StagedGateEvaluator
from cognitive_ew_smart_scan.src.training.val_set import FixedValidationSet
from cognitive_ew_smart_scan.src.data.tsrd_root import resolve_tsrd_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('confirm_gate25k_alpha020')

def run_confirmation():
    # 1. Paths & Configurations
    train_cfg_path = Path('cognitive_ew_smart_scan/configs/training_config.yaml')
    with open(train_cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)
    model_cfg_path = Path('cognitive_ew_smart_scan/configs/model_config.yaml')
    with open(model_cfg_path, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    data_dir = resolve_tsrd_root(None, train_cfg)
    frozen_ckpt_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt')
    assert frozen_ckpt_path.exists(), f'Frozen checkpoint not found at {frozen_ckpt_path}'

    env_cfg = copy.deepcopy(train_cfg.get('env', {}))
    reward_cfg = copy.deepcopy(model_cfg.get('reward', {}))
    full_env_cfg = {**env_cfg, **reward_cfg, 'n_bands': 36, 'n_modes': 5, 'n_actions': 180}
    full_env_cfg.setdefault('belief', {})
    full_env_cfg['belief']['ema_alpha_miss_confirmed'] = 0.20

    # 2. Load Frozen Weights
    ckpt = torch.load(frozen_ckpt_path, map_location='cpu', weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt['state_dict'])
    drqn.eval()

    # 3. Validation Set (Canonical 10 Scenarios)
    val_cfg = train_cfg.get('validation', {})
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get('subset', 'val')),
        mode='stare',
        n_files=int(val_cfg.get('n_files', 10)),
        seed=int(val_cfg.get('seed', 42)),
        allow_synthetic_fallback=False,
    )
    assert len(val_set.files_used) == 10, f'Expected 10 val files, got {len(val_set.files_used)}'

    # 4. Evaluator (Hierarchy of baselines + DRQN)
    evaluator = StagedGateEvaluator(
        output_dir=Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate'),
        gates=[],
        val_files=val_set.files_used,
        env_config=full_env_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        seed=42,
        device=torch.device('cpu'),
    )

    logger.info('='*80)
    logger.info('RUNNING INDEPENDENT CONFIRMATION EVALUATION FOR Gate-25k-R4.2-alpha020')
    logger.info('Policies evaluated: random, round_robin, highest_occupancy, drqn')
    logger.info('='*80)

    bench = evaluator.evaluate_baseline_hierarchy(
        drqn,
        moe=None,
        n_steps=1000,
        policies=['random', 'round_robin', 'highest_occupancy', 'drqn']
    )

    policies = bench['policies']
    drqn_res = policies['drqn']
    rand_res = policies['random']
    rr_res = policies['round_robin']
    ho_res = policies['highest_occupancy']

    scens = {s['scenario_id']: s for s in drqn_res['scenario_breakdown']}
    agile_scen_ids = ('config_119', 'config_241', 'config_29', 'config_195')
    sparse_scen_ids = ('config_143', 'config_119')
    agile_irs = [scens[k]['intercept_rate'] for k in scens if any(k.startswith(aid) for aid in agile_scen_ids)]
    sparse_irs = [scens[k]['intercept_rate'] for k in scens if any(k.startswith(sid) for sid in sparse_scen_ids)]

    final_report = {
        'designation': 'Gate-25k-R4.2-alpha020',
        'status': 'OFFICIAL_OPERATIONAL_CANDIDATE',
        'checkpoint_source': str(frozen_ckpt_path),
        'ema_alpha_miss_confirmed': 0.20,
        'ema_alpha': 0.30,
        'drqn_mean_ir': drqn_res['intercept_rate'] * 100,
        'drqn_median_ir': float(np.median([s['intercept_rate'] for s in scens.values()])) * 100,
        'drqn_worst_case_ir': float(np.min([s['intercept_rate'] for s in scens.values()])) * 100,
        'drqn_agile_ir': float(np.mean(agile_irs)) * 100,
        'drqn_sparse_ir': float(np.mean(sparse_irs)) * 100,
        'config_29_ir': scens['config_29']['intercept_rate'] * 100,
        'config_241_ir': scens['config_241']['intercept_rate'] * 100,
        'config_119_ir': scens['config_119']['intercept_rate'] * 100,
        'config_143_ir': scens['config_143']['intercept_rate'] * 100,
        'random_ir': rand_res['intercept_rate'] * 100,
        'round_robin_ir': rr_res['intercept_rate'] * 100,
        'highest_occupancy_ir': ho_res['intercept_rate'] * 100,
        'distinct_bands': drqn_res['distinct_bands'],
        'pd': drqn_res['decision_level_pd'] * 100,
        'pfa': drqn_res['pfa'],
        'scenario_breakdown': {k: v['intercept_rate'] * 100 for k, v in scens.items()},
    }

    report_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/gate_25k_r4_2_alpha020_report.json')
    report_path.write_text(json.dumps(final_report, indent=2))
    logger.info('Saved final operational candidate report to %s', report_path)

    # Print summary
    print('\n' + '='*80)
    print('GATE-25k-R4.2-ALPHA020 CONFIRMATION SUMMARY')
    print('='*80)
    print(f"Designation:           Gate-25k-R4.2-alpha020 (OFFICIAL OPERATIONAL CANDIDATE)")
    print(f"Mean Intercept Rate:   {final_report['drqn_mean_ir']:.2f}% (vs Random {final_report['random_ir']:.2f}%, RR {final_report['round_robin_ir']:.2f}%, HO {final_report['highest_occupancy_ir']:.2f}%)")
    print(f"Median Intercept Rate: {final_report['drqn_median_ir']:.2f}%")
    print(f"Worst-case IR:         {final_report['drqn_worst_case_ir']:.2f}%")
    print(f"Agile IR:              {final_report['drqn_agile_ir']:.2f}%")
    print(f"Sparse IR:             {final_report['drqn_sparse_ir']:.2f}%")
    print(f"Fast Hopper (cfg_29):  {final_report['config_29_ir']:.2f}%")
    print(f"Hopper (cfg_241):      {final_report['config_241_ir']:.2f}%")
    print(f"Sparse (cfg_143):      {final_report['config_143_ir']:.2f}%")
    print(f"Sparse (cfg_119):      {final_report['config_119_ir']:.2f}%")
    print(f"Decision Pd:           {final_report['pd']:.2f}%")
    print(f"Pfa:                   {final_report['pfa']:.4f}")
    print(f"Distinct Bands:        {final_report['distinct_bands']:.1f}/36")
    print('='*80 + '\n')

if __name__ == '__main__':
    run_confirmation()

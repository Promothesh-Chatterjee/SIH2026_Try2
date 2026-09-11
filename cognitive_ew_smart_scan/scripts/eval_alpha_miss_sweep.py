import argparse
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
logger = logging.getLogger('alpha_miss_sweep')

def run_sweep():
    # Load base config
    train_cfg_path = Path('cognitive_ew_smart_scan/configs/training_config.yaml')
    with open(train_cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)
    model_cfg_path = Path('cognitive_ew_smart_scan/configs/model_config.yaml')
    with open(model_cfg_path, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    data_dir = resolve_tsrd_root(None, train_cfg)
    ckpt_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt')
    assert ckpt_path.exists(), f'Frozen checkpoint not found at {ckpt_path}'

    env_cfg = copy.deepcopy(train_cfg.get('env', {}))
    reward_cfg = copy.deepcopy(model_cfg.get('reward', {}))
    full_env_cfg = {**env_cfg, **reward_cfg, 'n_bands': 36, 'n_modes': 5, 'n_actions': 180}

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt['state_dict'])
    drqn.eval()

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

    alpha_candidates = [
        ('A', 0.05),
        ('B', 0.08),
        ('C', 0.12),
        ('D', 0.20),
        ('Control', 0.30),
    ]

    results = []

    for label, alpha_val in alpha_candidates:
        logger.info('='*80)
        logger.info('EVALUATING CANDIDATE %s: alpha_miss_confirmed = %.2f', label, alpha_val)
        logger.info('='*80)

        cand_env_cfg = copy.deepcopy(full_env_cfg)
        cand_env_cfg.setdefault('belief', {})
        cand_env_cfg['belief']['ema_alpha_miss_confirmed'] = float(alpha_val)

        evaluator = StagedGateEvaluator(
            output_dir=Path('cognitive_ew_smart_scan/checkpoints/alpha_sweep_temp'),
            gates=[],
            val_files=val_set.files_used,
            env_config=cand_env_cfg,
            model_config=model_cfg,
            train_config=train_cfg,
            seed=42,
            device=torch.device('cpu'),
        )

        bench = evaluator.evaluate_baseline_hierarchy(drqn, moe=None, n_steps=1000, policies=['drqn'])
        drqn_res = bench['policies']['drqn']
        scens = {s['scenario_id']: s for s in drqn_res['scenario_breakdown']}

        agile_scen_ids = ('config_119', 'config_241', 'config_29', 'config_195')
        sparse_scen_ids = ('config_143', 'config_119')
        agile_irs = [scens[k]['intercept_rate'] for k in scens if any(k.startswith(aid) for aid in agile_scen_ids)]
        sparse_irs = [scens[k]['intercept_rate'] for k in scens if any(k.startswith(sid) for sid in sparse_scen_ids)]

        res_entry = {
            'label': label,
            'alpha_miss': alpha_val,
            'mean_ir': drqn_res['intercept_rate'] * 100,
            'median_ir': float(np.median([s['intercept_rate'] for s in scens.values()])) * 100,
            'worst_case_ir': float(np.min([s['intercept_rate'] for s in scens.values()])) * 100,
            'agile_ir': float(np.mean(agile_irs)) * 100,
            'sparse_ir': float(np.mean(sparse_irs)) * 100,
            'config_29_ir': scens['config_29']['intercept_rate'] * 100,
            'config_241_ir': scens['config_241']['intercept_rate'] * 100,
            'config_119_ir': scens['config_119']['intercept_rate'] * 100,
            'config_143_ir': scens['config_143']['intercept_rate'] * 100,
            'distinct_bands': drqn_res['distinct_bands'],
            'pd': drqn_res['decision_level_pd'] * 100,
            'pfa': drqn_res['pfa'],
        }
        results.append(res_entry)
        logger.info('Candidate %s (alpha=%.2f): Mean IR=%.2f%% | Agile IR=%.2f%% | Sparse IR=%.2f%% | Worst=%.2f%% | cfg_29=%.2f%%',
                    label, alpha_val, res_entry['mean_ir'], res_entry['agile_ir'], res_entry['sparse_ir'], res_entry['worst_case_ir'], res_entry['config_29_ir'])

    out_file = Path('cognitive_ew_smart_scan/checkpoints/alpha_miss_sweep_results.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    logger.info('Sweep results saved to %s', out_file)

if __name__ == '__main__':
    run_sweep()
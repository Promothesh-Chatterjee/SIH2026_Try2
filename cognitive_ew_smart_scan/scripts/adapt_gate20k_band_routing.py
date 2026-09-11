import logging
from pathlib import Path
import torch
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('adapt_checkpoint')

def adapt_checkpoint():
    src_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_gate20k/checkpoint_gate_20000.pt')
    dst_dir = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_rescue_r4')
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / 'checkpoint_gate_20000_band_routing.pt'

    logger.info('Loading immutable baseline from %s', src_path)
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)

    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    # load_state_dict automatically loads shared components and runs calibrate_band_routing
    model.load_state_dict(ckpt['state_dict'])

    new_ckpt = {
        'global_step': ckpt.get('global_step', 20000),
        'episode': ckpt.get('episode', 0),
        'epsilon': ckpt.get('epsilon', 1.0),
        'reward_baseline': ckpt.get('reward_baseline', -0.39),
        'state_dict': model.state_dict(),
        'target_state_dict': model.state_dict(),
        'metadata': {
            **ckpt.get('metadata', {}),
            'architecture': 'DRQNScheduler_band_routing_v1',
            'parent_checkpoint': str(src_path),
        }
    }

    torch.save(new_ckpt, dst_path)
    logger.info('Successfully saved adapted checkpoint to %s', dst_path)

if __name__ == '__main__':
    adapt_checkpoint()
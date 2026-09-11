import numpy as np
import torch
from pathlib import Path
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv, BeliefState
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer

def audit_r4c_q_landscape():
    print('='*70)
    print('R4-C: AUDITING GATE 20K Q-LANDSCAPE OVER ALL 36 BANDS')
    print('='*70)
    
    ckpt_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_gate20k/checkpoint_gate_20000.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt['state_dict'])
    drqn.eval()
    
    n_bands = 36
    # Cold Neutral Belief
    b = BeliefState(n_bands)
    obs = np.zeros(360, dtype=np.float32)
    for i in range(n_bands):
        obs[i*10:(i+1)*10] = b.band_features(i)
        
    t_obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
    h = drqn.init_hidden(1, 'cpu')
    with torch.no_grad():
        q, _, _ = drqn(t_obs, h)
        q_np = q[0, 0].numpy()
        
    band_max_q = [np.max(q_np[i*5:(i+1)*5]) for i in range(n_bands)]
    top_bands = np.argsort(band_max_q)[::-1]
    
    print('Top 10 Bands by Q-value in neutral state:')
    for rank, band_idx in enumerate(top_bands[:10]):
        best_mode = np.argmax(q_np[band_idx*5:(band_idx+1)*5])
        print(f'  Rank {rank+1:2d}: Band {band_idx:2d} (Mode {best_mode}) -> Q = {band_max_q[band_idx]:.4f}')
        
    print('\nBottom 5 Bands by Q-value in neutral state:')
    for rank, band_idx in enumerate(top_bands[-5:]):
        best_mode = np.argmax(q_np[band_idx*5:(band_idx+1)*5])
        print(f'  Rank {36-4+rank:2d}: Band {band_idx:2d} (Mode {best_mode}) -> Q = {band_max_q[band_idx]:.4f}')
        
    # Now: What if Band 30 (a bottom band) gets a massive confirmed detection?
    # Occupancy = 1.0, DetRate = 1.0, MissRate = 0.0, Uncertainty = 0.0, EmitterCount = 1.0, Agility = 1.0, Conf = 1.0
    obs_boost = obs.copy()
    obs_boost[30*10 : 30*10 + 10] = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    t_boost = torch.from_numpy(obs_boost).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_boost, _, _ = drqn(t_boost, h)
        q_boost_np = q_boost[0, 0].numpy()
        
    b30_q_before = np.max(q_np[30*5:31*5])
    b30_q_after = np.max(q_boost_np[30*5:31*5])
    overall_max_q_after = np.max(q_boost_np)
    overall_best_action_after = np.argmax(q_boost_np)
    
    print('\nTesting Response to Maximum Confirmed Detection on Bottom Band 30:')
    print(f'  Band 30 Q before: {b30_q_before:.4f}')
    print(f'  Band 30 Q after:  {b30_q_after:.4f} (delta = {b30_q_after - b30_q_before:+.4f})')
    print(f'  Overall Best Action after detection: Band {overall_best_action_after//5}, Mode {overall_best_action_after%5} (Q = {overall_max_q_after:.4f})')
    print(f'  Did Band 30 win the argmax? {overall_best_action_after//5 == 30}')

if __name__ == '__main__':
    audit_r4c_q_landscape()

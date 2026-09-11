import numpy as np
import torch
from pathlib import Path
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import BeliefState

def audit_dueling():
    ckpt_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_gate20k/checkpoint_gate_20000.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt['state_dict'])
    drqn.eval()

    n_bands = 36
    b = BeliefState(n_bands)
    obs = np.zeros(360, dtype=np.float32)
    for i in range(n_bands):
        obs[i*10:(i+1)*10] = b.band_features(i)

    obs_b30 = obs.copy()
    obs_b30[30*10 : 30*10 + 10] = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    h = drqn.init_hidden(1, 'cpu')
    with torch.no_grad():
        t1 = drqn.input_norm(torch.from_numpy(obs).unsqueeze(0).unsqueeze(0))
        t2 = drqn.input_norm(torch.from_numpy(obs_b30).unsqueeze(0).unsqueeze(0))
        
        out1, _ = drqn.lstm(t1, h)
        out2, _ = drqn.lstm(t2, h)
        
        v1 = drqn.value_stream(out1).item()
        v2 = drqn.value_stream(out2).item()
        
        a1 = drqn.advantage_stream(out1)[0, 0].numpy()
        a2 = drqn.advantage_stream(out2)[0, 0].numpy()
        
    print(f'V(neutral state): {v1:.4f}')
    print(f'V(Band 30 active): {v2:.4f} (delta = {v2 - v1:+.4f})')
    print(f'Band 2 Mode 4: A1 = {a1[2*5+4]:.4f}, A2 = {a2[2*5+4]:.4f} (delta = {a2[2*5+4] - a1[2*5+4]:+.4f})')
    print(f'Band 30 Mode 1: A1 = {a1[30*5+1]:.4f}, A2 = {a2[30*5+1]:.4f} (delta = {a2[30*5+1] - a1[30*5+1]:+.4f})')

if __name__ == '__main__':
    audit_dueling()

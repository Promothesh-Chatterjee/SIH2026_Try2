import numpy as np
import torch
from pathlib import Path

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import BeliefState
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler

def audit_r4a_belief_decay():
    print('='*70)
    print('R4-A: AUDITING BELIEF PERSISTENCE ACROSS REVISIT GAPS')
    print('='*70)
    
    n_bands = 36
    belief = BeliefState(n_bands)
    
    # Simulate a hit on band 5
    target_band = 5
    belief.record_visit(target_band, hit=True, detections=[])
    belief.advance_time()
    belief.touch(target_band)
    
    initial_features = belief.band_features(target_band)
    print(f'Initial features on Band {target_band} post-hit:')
    print(f'  Occupancy: {initial_features[0]:.4f}')
    print(f'  Det Rate:  {initial_features[1]:.4f}')
    print(f'  Miss Rate: {initial_features[2]:.4f}')
    print(f'  Uncertainty: {initial_features[3]:.4f}')
    print(f'  Revisit Age (norm): {initial_features[4]:.4f}')
    print(f'  Priority:  {initial_features[9]:.4f}')
    
    print('\nSimulating visits to OTHER bands (band 5 unvisited):')
    for gap in [1, 2, 3, 5, 10, 20, 35]:
        b_sim = BeliefState(n_bands)
        b_sim.record_visit(target_band, hit=True, detections=[])
        b_sim.advance_time()
        b_sim.touch(target_band)
        
        # Now visit other bands for 'gap' steps
        for step in range(gap):
            other_b = (target_band + 1 + (step % (n_bands - 1))) % n_bands
            b_sim.record_visit(other_b, hit=False, detections=[])
            b_sim.advance_time()
            b_sim.touch(other_b)
            
        feats = b_sim.band_features(target_band)
        unvisited_feats = b_sim.band_features(0) # clean unvisited band
        print(f'  Gap={gap:2d} steps | Band {target_band}: Occ={feats[0]:.3f}, Age={feats[4]:.2f}, DetRate={feats[1]:.3f}, Prio={feats[9]:.3f} vs Unvisited Band 0: Occ={unvisited_feats[0]:.3f}, Age={unvisited_feats[4]:.2f}, Prio={unvisited_feats[9]:.3f}')

    print('\nSimulating CONSECUTIVE MISSES on Band 5 (revisiting an agile/sparse emitter when it is silent):')
    b_miss = BeliefState(n_bands)
    b_miss.record_visit(target_band, hit=True, detections=[])
    b_miss.advance_time()
    b_miss.touch(target_band)
    for m in range(1, 6):
        b_miss.record_visit(target_band, hit=False, detections=[])
        b_miss.advance_time()
        b_miss.touch(target_band)
        feats = b_miss.band_features(target_band)
        print(f'  Miss #{m}: Occ={feats[0]:.4f}, DetRate={feats[1]:.4f}, Uncertainty={feats[3]:.4f}, Priority={feats[9]:.4f}')

def audit_r4b_gate20k_q_responsiveness():
    print('\n' + '='*70)
    print('R4-B: AUDITING GATE 20k DRQN Q-VALUE RESPONSIVENESS TO BELIEF')
    print('='*70)
    
    ckpt_path = Path('cognitive_ew_smart_scan/checkpoints/scheduler_v2_gate20k/checkpoint_gate_20000.pt')
    if not ckpt_path.exists():
        print(f'Checkpoint {ckpt_path} not found!')
        return
        
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(state_dict)
    drqn.eval()
    
    n_bands = 36
    
    # State 1: Clean Reset
    b1 = BeliefState(n_bands)
    obs1 = np.zeros(360, dtype=np.float32)
    for b in range(n_bands):
        obs1[b*10:(b+1)*10] = b1.band_features(b)
        
    # State 2: Fresh Hit on Band 5
    b2 = BeliefState(n_bands)
    b2.record_visit(5, hit=True, detections=[])
    b2.advance_time()
    b2.touch(5)
    obs2 = np.zeros(360, dtype=np.float32)
    for b in range(n_bands):
        obs2[b*10:(b+1)*10] = b2.band_features(b)
        
    # State 3: 2 Misses after Hit on Band 5
    b3 = BeliefState(n_bands)
    b3.record_visit(5, hit=True, detections=[])
    b3.record_visit(5, hit=False, detections=[])
    b3.record_visit(5, hit=False, detections=[])
    b3.advance_time()
    b3.touch(5)
    obs3 = np.zeros(360, dtype=np.float32)
    for b in range(n_bands):
        obs3[b*10:(b+1)*10] = b3.band_features(b)
        
    with torch.no_grad():
        h = drqn.init_hidden(1, 'cpu')
        t1 = torch.from_numpy(obs1).unsqueeze(0).unsqueeze(0)
        t2 = torch.from_numpy(obs2).unsqueeze(0).unsqueeze(0)
        t3 = torch.from_numpy(obs3).unsqueeze(0).unsqueeze(0)
        
        q1, _, _ = drqn(t1, h)
        q2, _, _ = drqn(t2, h)
        q3, _, _ = drqn(t3, h)
        
        q1_b5 = q1[0, 0, 5*5:(5+1)*5].numpy()
        q2_b5 = q2[0, 0, 5*5:(5+1)*5].numpy()
        q3_b5 = q3[0, 0, 5*5:(5+1)*5].numpy()
        
        q1_all = q1[0, 0].numpy()
        q2_all = q2[0, 0].numpy()
        q3_all = q3[0, 0].numpy()
        
        print(f'Band 5 Q-values across modes (SHORT, NORMAL, LONG, PREEMPTIVE, VERIFY):')
        print(f'  Cold Neutral State:      {np.round(q1_b5, 2)} | Max Q overall: {np.max(q1_all):.2f} (Action {np.argmax(q1_all)}, Band {np.argmax(q1_all)//5})')
        print(f'  Fresh Hit State (Occ=.65):{np.round(q2_b5, 2)} | Max Q overall: {np.max(q2_all):.2f} (Action {np.argmax(q2_all)}, Band {np.argmax(q2_all)//5})')
        print(f'  After 2 Misses (Occ=.31): {np.round(q3_b5, 2)} | Max Q overall: {np.max(q3_all):.2f} (Action {np.argmax(q3_all)}, Band {np.argmax(q3_all)//5})')
        print(f'\nAction selected by flat_argmax in Fresh Hit State: Band {np.argmax(q2_all)//5}, Mode {np.argmax(q2_all)%5}')

if __name__ == '__main__':
    audit_r4a_belief_decay()
    audit_r4b_gate20k_q_responsiveness()

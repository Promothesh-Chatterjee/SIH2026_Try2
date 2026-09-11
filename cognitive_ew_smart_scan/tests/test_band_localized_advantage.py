import numpy as np
import pytest
import torch
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler

def test_gate0_architecture_shapes():
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    obs = torch.randn(2, 4, 360)
    q_values, aux, (h, c) = model(obs)
    assert q_values.shape == (2, 4, 180)
    assert 'intercept_prob' in aux and aux['intercept_prob'].shape == (2, 4, 180)
    assert 'intercept_time_us' in aux and aux['intercept_time_us'].shape == (2, 4, 180)
    assert h.shape == (2, 2, 256)
    assert c.shape == (2, 2, 256)

def test_gate0_flat_argmax_pure_greedy():
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    model.calibrate_band_routing()
    model.eval()
    obs = torch.randn(1, 1, 360)
    with torch.no_grad():
        q_vals, _, _ = model(obs)
        expected_action = int(torch.argmax(q_vals[0, -1]).item())
    action, _ = model.act(obs[0, -1], mode_selection='flat_argmax')
    assert action == expected_action
    telem = getattr(model, 'last_decision_telemetry', {})
    assert not telem.get('action_was_overridden', True)
    assert telem.get('final_action') == expected_action

def test_gate1_architectural_locality_36_of_36():
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    model.calibrate_band_routing()
    model.eval()
    n_bands = 36
    obs_base = torch.zeros(1, 1, 360)
    for b in range(n_bands):
        obs_base[0, 0, b*10:(b+1)*10] = torch.tensor([0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
    wins = 0
    target_band_deltas = []
    cross_band_deltas = []
    for target_b in range(n_bands):
        obs_test = obs_base.clone()
        obs_test[0, 0, target_b*10:(target_b+1)*10] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        with torch.no_grad():
            q_base, _, _ = model(obs_base)
            q_test, _, _ = model(obs_test)
            qb_p = q_base[0, 0].numpy()
            qt_p = q_test[0, 0].numpy()
        best_act = int(np.argmax(qt_p))
        if best_act // 5 == target_b:
            wins += 1
        t_delta = float(np.mean(qt_p[target_b*5:(target_b+1)*5] - qb_p[target_b*5:(target_b+1)*5]))
        other_d = [float(np.mean(qt_p[ob*5:(ob+1)*5] - qb_p[ob*5:(ob+1)*5])) for ob in range(n_bands) if ob != target_b]
        target_band_deltas.append(t_delta)
        cross_band_deltas.append(float(np.mean(other_d)))
    assert wins == 36, f'Expected 36/36 band wins, got {wins}/36'
    mean_t = float(np.mean(target_band_deltas))
    mean_c = float(np.mean(cross_band_deltas))
    assert mean_t / max(1e-6, mean_c) >= 5.0

def test_gate1_neutral_input_balance():
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    model.calibrate_band_routing()
    model.eval()
    obs_base = torch.zeros(1, 1, 360)
    for b in range(36):
        obs_base[0, 0, b*10:(b+1)*10] = torch.tensor([0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
    with torch.no_grad():
        q, _, _ = model(obs_base)
        qp = q[0, 0].numpy()
    b_means = [float(np.mean(qp[b*5:(b+1)*5])) for b in range(36)]
    assert float(np.std(b_means)) < 1e-4
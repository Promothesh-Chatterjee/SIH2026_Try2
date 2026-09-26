"""Factorized DRQN Scheduler for Phase G5.

Decouples action selection into:
  1. Band Selector: b in [0..35]
  2. Dwell Mode Selector: m in [0..4]

Under the additively separable hypothesis:
  Q(s, b, m) = V(s) + A_b_tilde(s, b) + A_m_tilde(s, m)
where:
  A_b_tilde(s, b) = A_b(s, b) - mean_b(A_b)
  A_m_tilde(s, m) = A_m(s, m) - mean_m(A_m)

Inherits shared representation from Gate-25k frozen root.
New factorized heads initialized deterministically under the G5 initialization contract.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("factorized_drqn_scheduler")

CANONICAL_OBS_DIM = 360
CANONICAL_N_BANDS = 36
CANONICAL_N_MODES = 5
CANONICAL_N_ACTIONS = 180


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 hash of tensor data bytes."""
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class FactorizedDRQNScheduler(nn.Module):
    """Factorized Dueling DRQN Scheduler separating frequency band and dwell mode."""

    def __init__(
        self,
        obs_dim: int = CANONICAL_OBS_DIM,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.n_actions = n_bands * n_modes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.band_features = obs_dim // n_bands

        # --- Shared Representation (Inherited from Gate-25k) ---
        self.input_norm = nn.LayerNorm(obs_dim)
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.band_encoder = nn.Sequential(
            nn.Linear(self.band_features, 32),
            nn.ReLU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
        )

        # --- Factorized Heads (G5 Architecture) ---
        # 1. Band Advantage Head: maps [band_emb_b (32) + ctx (32)] -> 1 scalar advantage per band
        self.band_head = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 2. Mode Advantage Head: maps lstm_out (256) -> 5 dwell mode advantages
        self.mode_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_modes),
        )

        # Auxiliary prediction heads (for telemetry parity)
        self.intercept_prob_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Sigmoid(),
        )
        self.intercept_time_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Softplus(),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return h0, c0

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Returns:
            q_flat: Flat Q-values of shape (B, T, 180) where action a = b*5 + m.
            q_factorized: Dict with keys 'v', 'a_band', 'a_mode', 'q_joint', etc.
            hidden: Updated LSTM hidden state.
        """
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        x = self.input_norm(obs)
        lstm_out, next_hidden = self.lstm(x, hidden)

        # Value stream: V(s) shape (B, T, 1)
        v = self.value_stream(lstm_out)

        # Band stream: compute per-band advantage A_b(s, b)
        obs_bands = obs.view(B, T, self.n_bands, self.band_features)
        band_emb = self.band_encoder(obs_bands)  # (B, T, 36, 32)
        ctx = self.ctx_proj(lstm_out).unsqueeze(2).expand(-1, -1, self.n_bands, -1)  # (B, T, 36, 32)
        joint_band = torch.cat([band_emb, ctx], dim=-1)  # (B, T, 36, 64)
        a_band_raw = self.band_head(joint_band).squeeze(-1)  # (B, T, 36)
        a_band_tilde = a_band_raw - a_band_raw.mean(dim=-1, keepdim=True)  # (B, T, 36)

        # Mode stream: compute mode advantage A_m(s, m)
        a_mode_raw = self.mode_head(lstm_out)  # (B, T, 5)
        a_mode_tilde = a_mode_raw - a_mode_raw.mean(dim=-1, keepdim=True)  # (B, T, 5)

        # Additively separable joint Q: Q(s, b, m) = V(s) + A_b_tilde(s, b) + A_m_tilde(s, m)
        # Shape: (B, T, 36, 5)
        q_joint = v.unsqueeze(-1) + a_band_tilde.unsqueeze(-1) + a_mode_tilde.unsqueeze(-2)
        q_flat = q_joint.view(B, T, self.n_actions)  # (B, T, 180)

        aux_data = {
            "v": v,
            "a_band_raw": a_band_raw,
            "a_band_tilde": a_band_tilde,
            "a_mode_raw": a_mode_raw,
            "a_mode_tilde": a_mode_tilde,
            "q_joint": q_joint,
        }
        return q_flat, aux_data, next_hidden

    @classmethod
    def from_gate25_checkpoint(
        cls,
        ckpt_path: Path,
        initialization_seed: int = 42,
    ) -> Tuple[FactorizedDRQNScheduler, Dict[str, Any]]:
        """Instantiate FactorizedDRQNScheduler inheriting Gate-25k shared representation

        and initializing factorized heads under the G5 initialization contract.
        """
        model = cls()
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd = payload.get("state_dict", payload.get("online_drqn", payload))

        inherited_params = []
        newly_initialized_params = []
        mapping_records = []

        # 1. Inherit shared representation directly
        shared_prefixes = ["input_norm.", "lstm.", "value_stream.", "band_encoder.", "ctx_proj."]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in shared_prefixes):
                if name in src_sd:
                    with torch.no_grad():
                        param.copy_(src_sd[name])
                    inherited_params.append(name)
                    mapping_records.append({
                        "dest_name": name,
                        "src_name": name,
                        "shape": list(param.shape),
                        "transformation": "direct_copy",
                        "dest_sha256": compute_tensor_sha256(param),
                    })
                else:
                    raise KeyError(f"Expected shared parameter {name} missing in Gate-25 state dict!")

        # 2. Inherit auxiliary heads directly for telemetry parity
        for aux_prefix in ["intercept_prob_head.", "intercept_time_head."]:
            for name, param in model.named_parameters():
                if name.startswith(aux_prefix) and name in src_sd:
                    with torch.no_grad():
                        param.copy_(src_sd[name])
                    inherited_params.append(name)
                    mapping_records.append({
                        "dest_name": name,
                        "src_name": name,
                        "shape": list(param.shape),
                        "transformation": "direct_copy",
                        "dest_sha256": compute_tensor_sha256(param),
                    })

        # 3. Initialize band_head deterministically from Gate-25 band_advantage_head
        # Layer 0: direct copy from band_advantage_head.0
        with torch.no_grad():
            model.band_head[0].weight.copy_(src_sd["band_advantage_head.0.weight"])
            model.band_head[0].bias.copy_(src_sd["band_advantage_head.0.bias"])
        inherited_params.extend(["band_head.0.weight", "band_head.0.bias"])
        mapping_records.append({
            "dest_name": "band_head.0.weight",
            "src_name": "band_advantage_head.0.weight",
            "shape": list(model.band_head[0].weight.shape),
            "transformation": "direct_copy",
            "dest_sha256": compute_tensor_sha256(model.band_head[0].weight),
        })
        mapping_records.append({
            "dest_name": "band_head.0.bias",
            "src_name": "band_advantage_head.0.bias",
            "shape": list(model.band_head[0].bias.shape),
            "transformation": "direct_copy",
            "dest_sha256": compute_tensor_sha256(model.band_head[0].bias),
        })

        # Layer 2: mean across the 5 mode rows of band_advantage_head.2
        src_w2 = src_sd["band_advantage_head.2.weight"]  # [5, 32]
        src_b2 = src_sd["band_advantage_head.2.bias"]    # [5]
        dst_w2 = src_w2.mean(dim=0, keepdim=True)        # [1, 32]
        dst_b2 = src_b2.mean(dim=0, keepdim=True)        # [1]

        with torch.no_grad():
            model.band_head[2].weight.copy_(dst_w2)
            model.band_head[2].bias.copy_(dst_b2)
        newly_initialized_params.extend(["band_head.2.weight", "band_head.2.bias"])
        mapping_records.append({
            "dest_name": "band_head.2.weight",
            "src_name": "band_advantage_head.2.weight",
            "src_shape": list(src_w2.shape),
            "dest_shape": list(dst_w2.shape),
            "transformation": "mean_across_mode_dimension_dim0",
            "src_sha256": compute_tensor_sha256(src_w2),
            "dest_sha256": compute_tensor_sha256(dst_w2),
        })
        mapping_records.append({
            "dest_name": "band_head.2.bias",
            "src_name": "band_advantage_head.2.bias",
            "src_shape": list(src_b2.shape),
            "dest_shape": list(dst_b2.shape),
            "transformation": "mean_across_mode_dimension_dim0",
            "src_sha256": compute_tensor_sha256(src_b2),
            "dest_sha256": compute_tensor_sha256(dst_b2),
        })

        # 4. Initialize mode_head deterministically with fixed seed
        torch.manual_seed(initialization_seed)
        nn.init.xavier_uniform_(model.mode_head[0].weight)
        nn.init.zeros_(model.mode_head[0].bias)
        nn.init.xavier_uniform_(model.mode_head[2].weight, gain=0.1)
        nn.init.zeros_(model.mode_head[2].bias)

        newly_initialized_params.extend([
            "mode_head.0.weight", "mode_head.0.bias",
            "mode_head.2.weight", "mode_head.2.bias",
        ])
        mapping_records.append({
            "dest_name": "mode_head.0.weight",
            "shape": list(model.mode_head[0].weight.shape),
            "transformation": f"xavier_uniform_seed_{initialization_seed}",
            "dest_sha256": compute_tensor_sha256(model.mode_head[0].weight),
        })
        mapping_records.append({
            "dest_name": "mode_head.2.weight",
            "shape": list(model.mode_head[2].weight.shape),
            "transformation": f"xavier_uniform_gain_0.1_seed_{initialization_seed}",
            "dest_sha256": compute_tensor_sha256(model.mode_head[2].weight),
        })

        # Architecture configuration fingerprint
        arch_summary = f"FactorizedDRQN(obs={CANONICAL_OBS_DIM},bands={CANONICAL_N_BANDS},modes={CANONICAL_N_MODES},hidden={model.lstm_hidden},seed={initialization_seed})"
        arch_hash = hashlib.sha256(arch_summary.encode()).hexdigest()

        manifest = {
            "initialization_contract": "G5-Factorized-v1",
            "initialization_seed": initialization_seed,
            "architecture_hash": arch_hash,
            "parent_checkpoint": str(ckpt_path),
            "lineage_statement": "Gate-25 shared representation inherited; new factorized heads initialized under G5 initialization contract.",
            "inherited_parameters_count": len(inherited_params),
            "newly_initialized_parameters_count": len(newly_initialized_params),
            "inherited_parameters": inherited_params,
            "newly_initialized_parameters": newly_initialized_params,
            "mapping_records": mapping_records,
        }
        return model, manifest

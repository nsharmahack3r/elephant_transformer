import os
import yaml
import torch
import numpy as np
from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion


class ElephantTrajectoryGenerator:
    BEHAVIOR_MAP = {'STOP': 0, 'MOVE': 1}
    SEASON_MAP = {'dry': 0, 'wet': 1}
    TOD_MAP = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}

    def __init__(self, checkpoint_path=None, model=None,
                 scaler_gps=None, config_path=None, device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        self.scaler_gps = scaler_gps
        self.diffusion = DDIMDiffusion(T=200, S=40)
        self.model = model
        self.config_path = config_path

        if checkpoint_path is not None and self.model is None:
            self._load_model(checkpoint_path)

    def _load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)

        d_model = ckpt.get('d_model')
        num_layers = ckpt.get('num_layers')
        nhead = ckpt.get('nhead')
        max_seq_len = 200

        if d_model is None and self.config_path:
            with open(self.config_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
            d_model = yaml_cfg['model']['d_model']
            num_layers = yaml_cfg['model']['num_layers']
            nhead = yaml_cfg['model']['nhead']
            max_seq_len = yaml_cfg['model'].get('max_seq_len', 200)

        d_model = d_model or 128
        nhead = nhead or 8
        num_layers = num_layers or 4

        self.model = ElephantFineDiffusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.model.eval()

    def build_conditions(self, behavior, season, time_of_day,
                         lulc, human_settle, n):
        return {
            'behavior': torch.full((n,), self.BEHAVIOR_MAP[behavior],
                                   dtype=torch.long, device=self.device),
            'season': torch.full((n,), self.SEASON_MAP[season],
                                 dtype=torch.long, device=self.device),
            'time_of_day': torch.full((n,), self.TOD_MAP[time_of_day],
                                      dtype=torch.long, device=self.device),
            'lulc': torch.full((n,), lulc,
                               dtype=torch.long, device=self.device),
            'human_settle': torch.full((n,), int(human_settle),
                                       dtype=torch.long, device=self.device),
            'move_type': torch.zeros(n, dtype=torch.long, device=self.device),
        }

    def _default_env_context(self, n, seq_len, season):
        if season == 'dry':
            defaults = dict(ndvi=0.25, evi=0.18, lst=42.0,
                            elev=1100.0, slope=2.0, water=0.1)
        else:
            defaults = dict(ndvi=0.55, evi=0.40, lst=32.0,
                            elev=1100.0, slope=2.0, water=0.5)
        return {
            k: torch.full((n, seq_len), v,
                          dtype=torch.float32, device=self.device)
            for k, v in defaults.items()
        }

    def generate(self, n_trajectories=100,
                 behavior='MOVE', season='dry',
                 time_of_day='morning', lulc=10,
                 human_settle=False, env_context=None, seq_len=200,
                 micro_batch=8):
        all_trajectories = []

        for start in range(0, n_trajectories, micro_batch):
            n_batch = min(micro_batch, n_trajectories - start)

            conditions = self.build_conditions(
                behavior, season, time_of_day,
                lulc, human_settle, n_batch
            )

            if env_context is None:
                env_context_batch = self._default_env_context(
                    n_batch, seq_len, season
                )
            else:
                env_context_batch = {k: v[start:start + n_batch]
                                     for k, v in env_context.items()}
            conditions.update(env_context_batch)

            conditions['speed'] = torch.zeros(n_batch, seq_len,
                                              dtype=torch.float32, device=self.device)
            conditions['accel'] = torch.zeros(n_batch, seq_len,
                                              dtype=torch.float32, device=self.device)
            conditions['turning'] = torch.zeros(n_batch, seq_len,
                                                dtype=torch.float32, device=self.device)
            conditions['bearing'] = torch.zeros(n_batch, seq_len,
                                                dtype=torch.float32, device=self.device)
            conditions['persist'] = torch.zeros(n_batch, seq_len,
                                                dtype=torch.float32, device=self.device)
            conditions['step'] = torch.zeros(n_batch, seq_len,
                                             dtype=torch.float32, device=self.device)

            with torch.no_grad():
                norm_trajectories = self.diffusion.generate(
                    self.model, conditions, self.device, seq_len
                )

            all_trajectories.append(norm_trajectories.cpu())

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        trajectories_np = torch.cat(all_trajectories, dim=0).numpy()
        if self.scaler_gps is not None:
            trajectories_denorm = self.scaler_gps.inverse_transform(
                trajectories_np.reshape(-1, 2)
            ).reshape(n_trajectories, seq_len, 2)
        else:
            trajectories_denorm = trajectories_np

        return trajectories_denorm

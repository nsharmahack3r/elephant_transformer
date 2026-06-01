import torch
import numpy as np
from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion


class ElephantTrajectoryGenerator:
    BEHAVIOR_MAP = {'STOP': 0, 'MOVE': 1}
    SEASON_MAP = {'dry': 0, 'wet': 1}
    TOD_MAP = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}

    def __init__(self, checkpoint_path=None, model=None,
                 scaler_gps=None, device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        self.scaler_gps = scaler_gps
        self.diffusion = DDIMDiffusion(T=200, S=40)
        self.model = model

        if checkpoint_path is not None and self.model is None:
            self._load_model(checkpoint_path)

    def _load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt['config']
        self.model = ElephantFineDiffusionTransformer(
            d_model=cfg.get('d_model', 256),
            nhead=cfg.get('nhead', 8),
            num_layers=cfg.get('num_layers', 6)
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
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
                 human_settle=False, env_context=None, seq_len=200):
        conditions = self.build_conditions(
            behavior, season, time_of_day,
            lulc, human_settle, n_trajectories
        )

        if env_context is None:
            env_context = self._default_env_context(
                n_trajectories, seq_len, season
            )
        conditions.update(env_context)

        with torch.no_grad():
            norm_trajectories = self.diffusion.generate(
                self.model, conditions, self.device, seq_len
            )

        trajectories_np = norm_trajectories.cpu().numpy()
        if self.scaler_gps is not None:
            trajectories_denorm = self.scaler_gps.inverse_transform(
                trajectories_np.reshape(-1, 2)
            ).reshape(n_trajectories, seq_len, 2)
        else:
            trajectories_denorm = trajectories_np

        return trajectories_denorm

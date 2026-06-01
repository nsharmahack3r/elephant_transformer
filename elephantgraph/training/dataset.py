import torch
from torch.utils.data import Dataset
import numpy as np


class ElephantWindowDataset(Dataset):
    def __init__(self, windows, scaler_gps=None,
                 scaler_kin=None, scaler_env=None):
        self.windows = windows
        self.scaler_gps = scaler_gps
        self.scaler_kin = scaler_kin
        self.scaler_env = scaler_env

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]

        gps = np.stack([w['lon'], w['lat']], axis=-1)
        if self.scaler_gps:
            gps = self.scaler_gps.transform(gps)

        kin = np.stack([
            w['speed'], w['accel'], w['turning'],
            w['bearing'], w['persist'], w['step']
        ], axis=-1)
        if self.scaler_kin:
            kin = self.scaler_kin.transform(kin)

        env = np.stack([
            w['ndvi'], w['evi'], w['lst'],
            w['elev'], w['slope'], w['water']
        ], axis=-1)
        if self.scaler_env:
            env = self.scaler_env.transform(env)

        return {
            'gps': torch.tensor(gps, dtype=torch.float32),
            'speed': torch.tensor(kin[:, 0], dtype=torch.float32),
            'accel': torch.tensor(kin[:, 1], dtype=torch.float32),
            'turning': torch.tensor(kin[:, 2], dtype=torch.float32),
            'bearing': torch.tensor(kin[:, 3], dtype=torch.float32),
            'persist': torch.tensor(kin[:, 4], dtype=torch.float32),
            'step': torch.tensor(kin[:, 5], dtype=torch.float32),
            'ndvi': torch.tensor(env[:, 0], dtype=torch.float32),
            'evi': torch.tensor(env[:, 1], dtype=torch.float32),
            'lst': torch.tensor(env[:, 2], dtype=torch.float32),
            'elev': torch.tensor(env[:, 3], dtype=torch.float32),
            'slope': torch.tensor(env[:, 4], dtype=torch.float32),
            'water': torch.tensor(env[:, 5], dtype=torch.float32),
            'behavior': torch.tensor(w['behavior'], dtype=torch.long),
            'season': torch.tensor(w['season'], dtype=torch.long),
            'time_of_day': torch.tensor(w['time_of_day'], dtype=torch.long),
            'lulc': torch.tensor(w['lulc'], dtype=torch.long),
            'human_settle': torch.tensor(w['human_settle'], dtype=torch.long),
            'move_type': torch.tensor(w['move_type'], dtype=torch.long),
        }

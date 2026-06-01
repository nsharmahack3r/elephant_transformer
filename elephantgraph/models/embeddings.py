import torch
import torch.nn as nn


class LonLatKinematicEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        self.lon_embed = nn.Linear(1, d_model)
        self.lat_embed = nn.Linear(1, d_model)

        self.speed_embed = nn.Linear(1, d_model)
        self.accel_embed = nn.Linear(1, d_model)
        self.turning_embed = nn.Linear(1, d_model)
        self.bearing_embed = nn.Linear(1, d_model)
        self.persist_embed = nn.Linear(1, d_model)
        self.step_embed = nn.Linear(1, d_model)

        self.TOKEN_TYPES = {
            'lon': 0, 'lat': 1,
            'speed': 2, 'accel': 3,
            'turning': 4, 'bearing': 5,
            'persist': 6, 'step': 7
        }

    def forward(self, lon, lat, speed, accel, turning, bearing, persist, step):
        B, T = lon.shape
        tokens = torch.stack([
            self.lon_embed(lon.unsqueeze(-1)),
            self.lat_embed(lat.unsqueeze(-1)),
            self.speed_embed(speed.unsqueeze(-1)),
            self.accel_embed(accel.unsqueeze(-1)),
            self.turning_embed(turning.unsqueeze(-1)),
            self.bearing_embed(bearing.unsqueeze(-1)),
            self.persist_embed(persist.unsqueeze(-1)),
            self.step_embed(step.unsqueeze(-1)),
        ], dim=2)
        return tokens.view(B, 8 * T, self.d_model)


class EnvironmentalEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, ndvi, evi, lst, elev, slope, water):
        env = torch.stack([ndvi, evi, lst, elev, slope, water], dim=-1)
        return self.proj(env)

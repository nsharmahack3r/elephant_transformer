import torch
import torch.nn as nn
from elephantgraph.models.embeddings import LonLatKinematicEmbedding, EnvironmentalEmbedding
from elephantgraph.models.positional_encoding import ElephantPositionalEncoding
from elephantgraph.models.adaln_block import BehaviorConditionedAdaLN


class ElephantFineDiffusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dropout=0.1, max_seq_len=200):
        super().__init__()

        self.d_model = d_model

        self.gps_kin_embed = LonLatKinematicEmbedding(d_model)
        self.env_embed = EnvironmentalEmbedding(d_model)
        self.pos_enc = ElephantPositionalEncoding(d_model)

        self.timestep_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.transformer_blocks = nn.ModuleList([
            BehaviorConditionedAdaLN(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2)

    def timestep_sinusoidal(self, t, d_model):
        half = d_model // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) *
            (torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb

    def forward(self, x_noisy, t,
                speed, accel, turning, bearing, persist, step,
                ndvi, evi, lst, elev, slope, water,
                behavior, season, time_of_day,
                lulc, human_settle, move_type):
        B, T, _ = x_noisy.shape

        lon_noisy = x_noisy[:, :, 0]
        lat_noisy = x_noisy[:, :, 1]

        tokens = self.gps_kin_embed(
            lon_noisy, lat_noisy,
            speed, accel, turning, bearing, persist, step
        )

        env_ctx = self.env_embed(ndvi, evi, lst, elev, slope, water)
        env_ctx_expanded = env_ctx.unsqueeze(2).repeat(1, 1, 8, 1)
        env_ctx_flat = env_ctx_expanded.view(B, 8 * T, self.d_model)
        tokens = tokens + env_ctx_flat

        pe = self.pos_enc(T).unsqueeze(0).to(tokens.device)
        tokens = tokens + pe

        t_emb = self.timestep_sinusoidal(t, self.d_model)
        t_emb = self.timestep_embed(t_emb).unsqueeze(1)
        tokens = tokens + t_emb

        for block in self.transformer_blocks:
            tokens, _ = block(
                tokens,
                behavior, season, time_of_day,
                lulc, human_settle, move_type
            )

        tokens = self.out_norm(tokens)
        lon_tokens = tokens[:, 0::8, :]
        lat_tokens = tokens[:, 1::8, :]

        lon_noise_pred = self.out_proj(lon_tokens)[:, :, 0]
        lat_noise_pred = self.out_proj(lat_tokens)[:, :, 1]

        return torch.stack([lon_noise_pred, lat_noise_pred], dim=-1)

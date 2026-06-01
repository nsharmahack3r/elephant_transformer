import torch
import torch.nn as nn
import math


class H3TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(80, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.season_embed = nn.Embedding(2, d_model)
        self.behavior_embed = nn.Embedding(2, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, node_indices, node_embeddings_lookup,
                season, behavior, mask=None):
        B, seq_len = node_indices.shape
        embeds = node_embeddings_lookup[node_indices]
        x = self.node_embed(embeds)

        pe = self._sinusoidal_pe(seq_len, x.shape[-1]).unsqueeze(0).to(x.device)
        x = x + pe

        season_cond = self.season_embed(season).unsqueeze(1)
        behavior_cond = self.behavior_embed(behavior).unsqueeze(1)
        x = x + season_cond + behavior_cond

        x = self.encoder(x, src_key_padding_mask=mask)
        return self.out_proj(x)

    def _sinusoidal_pe(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class CoarseGenerator(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.1,
                 num_h3_nodes=500):
        super().__init__()
        self.encoder = H3TransformerEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout
        )
        self.node_predictor = nn.Linear(d_model, num_h3_nodes)
        self.occupancy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, node_indices, node_embeddings, season, behavior, mask=None):
        h = self.encoder(node_indices, node_embeddings, season, behavior, mask)
        next_node_logits = self.node_predictor(h)
        occupancy_duration = self.occupancy_head(h).squeeze(-1)
        return next_node_logits, occupancy_duration

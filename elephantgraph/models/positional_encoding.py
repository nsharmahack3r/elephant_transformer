import torch
import math


class ElephantPositionalEncoding(torch.nn.Module):
    TOKEN_TYPE_IDS = {
        'lon': 0, 'lat': 1,
        'speed': 2, 'accel': 3,
        'turning': 4, 'bearing': 5,
        'persist': 6, 'step': 7
    }

    def __init__(self, d_model=256, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        self.register_buffer('pe_table', self._build_pe_table(max_seq_len, half))

    def _build_pe_table(self, max_len, half_d):
        # Vectorised sinusoidal PE — no Python loops over positions/types
        def sinusoidal(positions, d):
            # positions: (N,) → output: (N, d)
            i = torch.arange(0, d, 2, dtype=torch.float32)
            div = 10000 ** (2 * i / d)               # (d//2,)
            args = positions.unsqueeze(1) / div       # (N, d//2)
            pe = torch.zeros(len(positions), d)
            pe[:, 0::2] = torch.sin(args)
            pe[:, 1::2] = torch.cos(args[:, :d // 2])
            return pe

        pos_ids  = torch.arange(max_len, dtype=torch.float32)   # (max_len,)
        type_ids = torch.arange(8,       dtype=torch.float32)   # (8,)

        pos_pe  = sinusoidal(pos_ids,  half_d)   # (max_len, half_d)
        type_pe = sinusoidal(type_ids, half_d)   # (8,       half_d)

        # Broadcast: table[pos, type] = [type_half | pos_half]
        table = torch.cat([
            type_pe.unsqueeze(0).expand(max_len, -1, -1),  # (max_len, 8, half_d)
            pos_pe.unsqueeze(1).expand(-1, 8, -1),         # (max_len, 8, half_d)
        ], dim=-1)  # (max_len, 8, d_model)
        return table

    def forward(self, seq_len):
        # pe_table: (max_len, 8, d_model) — reshape to (8*seq_len, d_model)
        # interleave: for each timestep t, 8 type tokens in order
        return self.pe_table[:seq_len].reshape(seq_len * 8, self.d_model)

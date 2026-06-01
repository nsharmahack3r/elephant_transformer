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

    def _sinusoidal_pe(self, position, d):
        pe = torch.zeros(d)
        for i in range(0, d, 2):
            div = 10000 ** (2 * i / d)
            pe[i] = math.sin(position / div)
            if i + 1 < d:
                pe[i + 1] = math.cos(position / div)
        return pe

    def _build_pe_table(self, max_len, half_d):
        table = torch.zeros(max_len, 8, self.d_model)
        for pos in range(max_len):
            for type_id in range(8):
                type_pe = self._sinusoidal_pe(type_id, half_d)
                pos_pe = self._sinusoidal_pe(pos, half_d)
                table[pos, type_id] = torch.cat([type_pe, pos_pe])
        return table

    def forward(self, seq_len):
        pe_list = []
        for t in range(seq_len):
            for type_id in range(8):
                pe_list.append(self.pe_table[t, type_id])
        return torch.stack(pe_list)

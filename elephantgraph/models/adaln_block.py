import torch
import torch.nn as nn


class BehaviorConditionedAdaLN(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.behavior_embed = nn.Embedding(2, d_model)
        self.season_embed = nn.Embedding(2, d_model // 4)
        self.tod_embed = nn.Embedding(4, d_model // 4)
        self.lulc_embed = nn.Embedding(20, d_model // 4)
        self.settle_embed = nn.Embedding(2, d_model // 4)
        self.move_type_embed = nn.Embedding(5, d_model // 4)

        cond_in = d_model + 5 * (d_model // 4)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.modulation = nn.Linear(d_model, 6 * d_model, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def encode_condition(self, behavior, season, time_of_day,
                         lulc, human_settle, move_type):
        beh = self.behavior_embed(behavior)
        sea = self.season_embed(season)
        tod = self.tod_embed(time_of_day)
        lulc_ = self.lulc_embed(lulc.long())
        set_ = self.settle_embed(human_settle.long())
        mt = self.move_type_embed(move_type)
        cond = torch.cat([beh, sea, tod, lulc_, set_, mt], dim=-1)
        return self.cond_proj(cond)

    def forward(self, x, behavior, season, time_of_day,
                lulc, human_settle, move_type):
        cond = self.encode_condition(
            behavior, season, time_of_day, lulc, human_settle, move_type
        )
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.modulation(cond).chunk(6, dim=-1)

        x_norm = (1 + gamma1.unsqueeze(1)) * self.norm1(x) + beta1.unsqueeze(1)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + alpha1.unsqueeze(1) * attn_out

        x_norm = (1 + gamma2.unsqueeze(1)) * self.norm2(x) + beta2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.ff(x_norm)

        return x, attn_weights

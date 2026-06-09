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


class SeasonalLatentDictionary(nn.Module):
    """
    A WildGraph-style learned latent dictionary, keyed by season.

    Each season owns a *codebook* of `num_entries` (K) latent prototype vectors
    of size d_model.  These prototypes are learned during training and come to
    represent recurring seasonal "movement modes" (e.g. dry-season corridor
    use, wet-season dispersal, water-point looping, ...).

    At every sequence position the decoder hidden state queries its season's
    codebook via scaled dot-product attention and retrieves a latent `z`, which
    is fused back into the hidden state.  This makes the next-cell prediction
    explicitly conditioned on a *discrete, reusable* seasonal latent rather than
    a single additive season embedding.

    Returned attention weights are used by the trainer for two regularizers:
      - diversity     : keep the K prototypes distinct (anti-collapse)
      - load-balance  : ensure all K prototypes get used (anti-mode-dropping)

    Inference controls:
      - latent_temperature : <1 sharpens selection toward a single mode,
                             >1 softens / blends modes
      - latent_mode        : force a specific dictionary entry index k
                             (hard one-hot selection) to deterministically
                             replay one seasonal movement mode
    """

    def __init__(self, d_model, num_seasons=2, num_entries=16, dropout=0.1):
        super().__init__()
        self.d_model     = d_model
        self.num_seasons = num_seasons
        self.num_entries = num_entries

        # The dictionary itself: (num_seasons, K, d_model)
        self.codebook = nn.Parameter(
            torch.randn(num_seasons, num_entries, d_model) * 0.02
        )

        self.query_proj = nn.Linear(d_model, d_model)
        self.out_proj   = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.scale      = d_model ** -0.5

    def forward(self, h, season, latent_temperature=1.0, latent_mode=None):
        """
        h       : (B, S, d_model) decoder hidden states
        season  : (B,) long, season index per sequence
        returns : h_out (B, S, d_model), attn (B, S, K)
        """
        B, S, D = h.shape
        book = self.codebook[season]                       # (B, K, D)

        q = self.query_proj(h)                             # (B, S, D)
        scores = torch.einsum('bsd,bkd->bsk', q, book) * self.scale   # (B, S, K)

        if latent_mode is not None:
            # Hard one-hot selection of a single dictionary entry
            attn = torch.zeros_like(scores)
            attn[..., int(latent_mode)] = 1.0
        else:
            attn = torch.softmax(scores / latent_temperature, dim=-1)  # (B, S, K)

        z = torch.einsum('bsk,bkd->bsd', attn, book)       # (B, S, D)
        z = self.dropout(self.out_proj(z))
        h_out = self.norm(h + z)                           # residual fuse
        return h_out, attn

    # ── Regularizers (used by the trainer) ──────────────────────────────────
    def diversity_loss(self):
        """Penalize redundancy within each season's codebook (off-diagonal
        cosine similarity).  Encourages K distinct prototypes."""
        cb = nn.functional.normalize(self.codebook, dim=-1)   # (Sn, K, D)
        gram = torch.matmul(cb, cb.transpose(-1, -2))         # (Sn, K, K)
        eye  = torch.eye(self.num_entries, device=cb.device).unsqueeze(0)
        off  = gram - gram * eye                              # zero the diagonal
        denom = self.num_entries * (self.num_entries - 1) + 1e-8
        return (off.pow(2).sum(dim=(-1, -2)) / denom).mean()

    @staticmethod
    def load_balance_loss(attn):
        """MoE-style importance loss: encourage all dictionary entries to be
        used across the batch (push mean usage toward uniform)."""
        # attn: (B, S, K) -> mean usage per entry
        usage = attn.reshape(-1, attn.shape[-1]).mean(dim=0)  # (K,)
        K = usage.shape[0]
        # Minimized (= 1.0) when usage is uniform; grows as it concentrates
        return K * usage.pow(2).sum()


class CoarseGenerator(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.1,
                 num_h3_nodes=500, num_latent_entries=16, num_seasons=2):
        super().__init__()
        self.encoder = H3TransformerEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout
        )
        self.latent_dict = SeasonalLatentDictionary(
            d_model=d_model, num_seasons=num_seasons,
            num_entries=num_latent_entries, dropout=dropout
        )
        self.node_predictor = nn.Linear(d_model, num_h3_nodes)
        self.occupancy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, node_indices, node_embeddings, season, behavior, mask=None,
                return_aux=False, latent_temperature=1.0, latent_mode=None):
        h = self.encoder(node_indices, node_embeddings, season, behavior, mask)

        # Seasonal latent dictionary lookup
        h, attn = self.latent_dict(
            h, season,
            latent_temperature=latent_temperature,
            latent_mode=latent_mode,
        )

        next_node_logits   = self.node_predictor(h)
        occupancy_duration = self.occupancy_head(h).squeeze(-1)

        if return_aux:
            aux = {
                'attn':             attn,
                'diversity_loss':   self.latent_dict.diversity_loss(),
                'load_balance_loss': SeasonalLatentDictionary.load_balance_loss(attn),
            }
            return next_node_logits, occupancy_duration, aux
        return next_node_logits, occupancy_duration

import torch
import torch.nn as nn
from elephantgraph.models.coarse_generator import CoarseGenerator
from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion


class ElephantGraph(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers_fine=6,
                 num_layers_coarse=4, dropout=0.1, max_seq_len=200,
                 num_h3_nodes=500):
        super().__init__()

        self.coarse_generator = CoarseGenerator(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_coarse,
            dropout=dropout,
            num_h3_nodes=num_h3_nodes,
        )

        self.fine_generator = ElephantFineDiffusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_fine,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.diffusion = DDIMDiffusion(T=200, S=40)

    def generate_regions(self, start_node, node_embeddings, season, behavior,
                         horizon=24):
        with torch.no_grad():
            current = start_node.unsqueeze(0)
            region_seq = [current]
            for _ in range(horizon - 1):
                logits, _ = self.coarse_generator(
                    torch.stack(region_seq, dim=1),
                    node_embeddings, season, behavior
                )
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                region_seq.append(next_token)
            return torch.cat(region_seq, dim=1)

    def generate_trajectory(self, conditions, device, seq_len=200):
        return self.diffusion.generate(
            self.fine_generator, conditions, device, seq_len
        )

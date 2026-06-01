import torch
import torch.nn as nn
import numpy as np


class DDIMDiffusion:
    def __init__(self, T=200, S=40, beta_start=0.0001, beta_end=0.02):
        self.T = T
        self.S = S

        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sample_steps = list(range(0, T, T // S))[::-1]

    def q_sample(self, x0, t):
        alpha_bar_t = self.alpha_bar.to(t.device)[t].view(-1, 1, 1).to(x0.device)
        noise = torch.randn_like(x0)
        x_t = (alpha_bar_t ** 0.5) * x0 + ((1 - alpha_bar_t) ** 0.5) * noise
        return x_t, noise

    def p_sample_ddim(self, x_t, t, t_prev, predicted_noise):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1).to(x_t.device)
        if t_prev >= 0:
            alpha_bar_t_prev = self.alpha_bar[t_prev].view(-1, 1, 1).to(x_t.device)
        else:
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

        pred_x0 = (x_t - (1 - alpha_bar_t) ** 0.5 * predicted_noise) / (alpha_bar_t ** 0.5 + 1e-8)
        pred_x0 = pred_x0.clamp(-3, 3)
        x_t_prev = alpha_bar_t_prev ** 0.5 * pred_x0 + (1 - alpha_bar_t_prev) ** 0.5 * predicted_noise

        return x_t_prev

    @torch.no_grad()
    def generate(self, model, conditions, device, seq_len=200):
        B = conditions['behavior'].shape[0]
        x = torch.randn(B, seq_len, 2).to(device)

        for i, t in enumerate(self.sample_steps):
            t_prev = self.sample_steps[i + 1] if i + 1 < len(self.sample_steps) else -1
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            predicted_noise = model(x, t_tensor, **conditions)
            x = self.p_sample_ddim(x, t, t_prev, predicted_noise)

        return x

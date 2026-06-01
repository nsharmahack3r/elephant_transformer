import torch
import torch.nn.functional as F
import math


def haversine_loss(pred_lon, pred_lat, true_lon, true_lat):
    R = 6371.0
    d_lat = torch.deg2rad(true_lat - pred_lat)
    d_lon = torch.deg2rad(true_lon - pred_lon)
    a = torch.sin(d_lat / 2) ** 2 + \
        torch.cos(torch.deg2rad(pred_lat)) * \
        torch.cos(torch.deg2rad(true_lat)) * \
        torch.sin(d_lon / 2) ** 2
    dist = 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
    return dist.mean()


def kinematic_consistency_loss(pred_gps, true_speed, true_turning):
    diffs = pred_gps[:, 1:, :] - pred_gps[:, :-1, :]
    impl_speed = torch.norm(diffs, dim=-1)

    v1 = diffs[:, :-1, :]
    v2 = diffs[:, 1:, :]
    cos_sim = F.cosine_similarity(v1, v2, dim=-1)
    impl_turn = torch.acos(torch.clamp(cos_sim, -1, 1))

    speed_loss = F.mse_loss(impl_speed.mean(dim=1), true_speed.mean(dim=1))
    turning_loss = F.mse_loss(impl_turn.mean(dim=1),
                               true_turning[:, 1:].abs().mean(dim=1))

    return speed_loss + 0.5 * turning_loss


def ecological_validity_loss(pred_gps, water_map, behavior, season):
    dry_mask = (season == 0)
    if dry_mask.sum() == 0:
        return torch.tensor(0.0)

    end_lon = pred_gps[dry_mask, -1, 0]
    end_lat = pred_gps[dry_mask, -1, 1]
    water_dist = water_map.query_distance(end_lon, end_lat)
    return water_dist.mean()


def total_diffusion_loss(predicted_noise, true_noise,
                         pred_x0=None, true_x0=None,
                         true_speed=None, true_turning=None,
                         behavior=None, season=None,
                         l_kin=0.1, l_eco=0.05):
    noise_loss = F.mse_loss(predicted_noise, true_noise)
    total = noise_loss

    if pred_x0 is not None and true_speed is not None:
        kin_loss = kinematic_consistency_loss(pred_x0, true_speed, true_turning)
        total = total + l_kin * kin_loss

    return total

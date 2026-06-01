import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def speed_distribution_match(real_speeds, gen_speeds):
    ks_stat, ks_pval = ks_2samp(real_speeds, gen_speeds)
    w_dist = wasserstein_distance(real_speeds, gen_speeds)
    return {
        'speed_ks_statistic': float(ks_stat),
        'speed_ks_pvalue': float(ks_pval),
        'speed_wasserstein': float(w_dist),
    }


def turning_angle_distribution_match(real_turns, gen_turns):
    ks_stat, ks_pval = ks_2samp(real_turns, gen_turns)
    w_dist = wasserstein_distance(real_turns, gen_turns)
    return {
        'turning_ks_statistic': float(ks_stat),
        'turning_ks_pvalue': float(ks_pval),
        'turning_wasserstein': float(w_dist),
    }


def step_length_distribution_match(real_steps, gen_steps):
    ks_stat, ks_pval = ks_2samp(real_steps, gen_steps)
    w_dist = wasserstein_distance(real_steps, gen_steps)
    return {
        'step_ks_statistic': float(ks_stat),
        'step_ks_pvalue': float(ks_pval),
        'step_wasserstein': float(w_dist),
    }


def density_pattern_score(real_trajs, gen_trajs, grid_size=100):
    all_lons = np.concatenate([t[:, 0].ravel() for t in real_trajs + gen_trajs])
    all_lats = np.concatenate([t[:, 1].ravel() for t in real_trajs + gen_trajs])
    lon_bins = np.linspace(all_lons.min(), all_lons.max(), grid_size)
    lat_bins = np.linspace(all_lats.min(), all_lats.max(), grid_size)

    def compute_density(trajs):
        lons = np.concatenate([t[:, 0].ravel() for t in trajs])
        lats = np.concatenate([t[:, 1].ravel() for t in trajs])
        hist, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
        return hist.flatten() / hist.sum()

    real_density = compute_density(real_trajs)
    gen_density = compute_density(gen_trajs)
    js_div = _jensen_shannon_divergence(real_density + 1e-10,
                                         gen_density + 1e-10)
    return float(js_div)


def _jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def r_coefficient(real_trajs, gen_trajs):
    def compute_r(trajs):
        all_points = np.concatenate([t for t in trajs], axis=0)
        centroid = all_points.mean(axis=0)
        dists = np.sqrt(((all_points - centroid) ** 2).sum(axis=1))
        return float(dists.std() / (dists.mean() + 1e-8))

    r_real = compute_r(real_trajs)
    r_gen = compute_r(gen_trajs)
    return {
        'r_real': r_real,
        'r_generated': r_gen,
        'r_ratio': r_gen / (r_real + 1e-8),
    }

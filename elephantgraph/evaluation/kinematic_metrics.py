import numpy as np


def compute_kinematics_from_traj(traj):
    diffs = np.diff(traj, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    step_lengths = np.linalg.norm(diffs, axis=1)

    if len(diffs) >= 2:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        cos_sim = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        )
        turning_angles = np.arccos(np.clip(cos_sim, -1, 1))
    else:
        turning_angles = np.zeros(0)

    return speeds, turning_angles, step_lengths


def speed_mean_match(real_trajs, gen_trajs):
    real_speeds = [compute_kinematics_from_traj(t)[0].mean() for t in real_trajs]
    gen_speeds = [compute_kinematics_from_traj(t)[0].mean() for t in gen_trajs]
    real_mean = np.mean(real_speeds)
    gen_mean = np.mean(gen_speeds)
    return {
        'real_mean_speed': float(real_mean),
        'gen_mean_speed': float(gen_mean),
        'speed_ratio': float(gen_mean / (real_mean + 1e-8)),
    }


def acceleration_distribution(real_trajs, gen_trajs):
    from scipy.stats import ks_2samp

    all_real_accel = []
    all_gen_accel = []
    for t in real_trajs:
        speeds = compute_kinematics_from_traj(t)[0]
        accels = np.diff(speeds)
        all_real_accel.extend(accels)
    for t in gen_trajs:
        speeds = compute_kinematics_from_traj(t)[0]
        accels = np.diff(speeds)
        all_gen_accel.extend(accels)

    ks_stat, ks_pval = ks_2samp(all_real_accel, all_gen_accel)
    return {
        'accel_ks_statistic': float(ks_stat),
        'accel_ks_pvalue': float(ks_pval),
    }


def movement_consistency_score(trajs):
    scores = []
    for traj in trajs:
        bearings = np.arctan2(
            np.diff(traj[:, 1]), np.diff(traj[:, 0])
        )
        if len(bearings) >= 2:
            bearing_diffs = np.abs(np.diff(bearings))
            bearing_diffs = np.minimum(bearing_diffs, 2 * np.pi - bearing_diffs)
            scores.append(float(np.mean(bearing_diffs)))
    return {
        'mean_bearing_change': float(np.mean(scores)) if scores else 0.0,
        'std_bearing_change': float(np.std(scores)) if scores else 0.0,
    }

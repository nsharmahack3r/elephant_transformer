import numpy as np
from scipy.signal import savgol_filter


def smooth_trajectory(traj, window_length=5, polyorder=2):
    if len(traj) < window_length:
        return traj
    smoothed = np.zeros_like(traj)
    smoothed[:, 0] = savgol_filter(traj[:, 0], window_length, polyorder)
    smoothed[:, 1] = savgol_filter(traj[:, 1], window_length, polyorder)
    return smoothed


def remove_stationary_points(traj, min_step_m=1.0):
    keep = np.ones(len(traj), dtype=bool)
    for i in range(1, len(traj)):
        dist = np.sqrt((traj[i, 0] - traj[i - 1, 0]) ** 2 +
                       (traj[i, 1] - traj[i - 1, 1]) ** 2)
        keep[i] = dist >= min_step_m
    return traj[keep]


def enforce_bounds(traj, lat_bounds, lon_bounds):
    lat = np.clip(traj[:, 1], lat_bounds[0], lat_bounds[1])
    lon = np.clip(traj[:, 0], lon_bounds[0], lon_bounds[1])
    return np.stack([lon, lat], axis=-1)


def compute_kinematics(traj):
    diffs = np.diff(traj, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)

    if len(diffs) >= 2:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        cos_sim = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        )
        turning = np.arccos(np.clip(cos_sim, -1, 1))
    else:
        turning = np.zeros(0)

    bearings = np.arctan2(diffs[:, 1], diffs[:, 0])

    return {
        'speeds': speeds,
        'turning_angles': np.degrees(turning),
        'bearings': np.degrees(bearings),
    }


def postprocess_trajectory(traj, smooth=True, remove_stationary=True,
                           lat_bounds=None, lon_bounds=None):
    if smooth:
        traj = smooth_trajectory(traj)

    if remove_stationary:
        traj = remove_stationary_points(traj)

    if lat_bounds is not None and lon_bounds is not None:
        traj = enforce_bounds(traj, lat_bounds, lon_bounds)

    return traj

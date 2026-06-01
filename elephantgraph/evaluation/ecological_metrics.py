import numpy as np
from scipy.spatial import KDTree


def ndvi_path_score(trajectories, ndvi_raster):
    scores = []
    for traj in trajectories:
        path_ndvi = []
        for lon, lat in traj:
            try:
                val = ndvi_raster.sample([(lon, lat)]).__next__()[0]
            except Exception:
                val = 0.0
            path_ndvi.append(val)
        scores.append(np.mean(path_ndvi))
    return np.array(scores)


def water_proximity_score(trajectories, water_points, season_code):
    tree = KDTree(water_points)
    scores = []
    for traj in trajectories:
        dists, _ = tree.query(traj)
        scores.append(dists.mean())
    return np.array(scores)


def behavior_fidelity(generated_speeds, generated_turnings,
                      condition_behavior):
    move_mask = np.array(condition_behavior) == 1
    stop_mask = np.array(condition_behavior) == 0

    move_speed = (np.mean(generated_speeds[move_mask]) if move_mask.sum() > 0
                  else 0.0)
    stop_speed = (np.mean(generated_speeds[stop_mask]) if stop_mask.sum() > 0
                  else 1e-6)

    speed_ratio = move_speed / (stop_speed + 1e-6)

    return {
        'behavior_respected': move_speed > stop_speed,
        'move_mean_speed': float(move_speed),
        'stop_mean_speed': float(stop_speed),
        'speed_ratio': float(speed_ratio),
    }


def elevation_consistency(trajectories, dem_raster):
    scores = []
    for traj in trajectories:
        elevs = []
        for lon, lat in traj:
            try:
                val = dem_raster.sample([(lon, lat)]).__next__()[0]
            except Exception:
                val = 0.0
            elevs.append(val)
        scores.append(float(np.std(elevs) / (np.mean(elevs) + 1e-8)))
    return np.array(scores)

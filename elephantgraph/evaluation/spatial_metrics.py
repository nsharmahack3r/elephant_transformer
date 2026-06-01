import numpy as np
from scipy.spatial.distance import directed_hausdorff
from fastdtw import fastdtw


def hausdorff_distance(traj_a, traj_b):
    d1 = directed_hausdorff(traj_a, traj_b)[0]
    d2 = directed_hausdorff(traj_b, traj_a)[0]
    return max(d1, d2)


def dtw_distance(traj_a, traj_b):
    dist, _ = fastdtw(traj_a, traj_b)
    return dist


def coverage(real_trajs, gen_trajs, metric_fn=hausdorff_distance):
    covered = set()
    for gen in gen_trajs:
        best_real = min(
            range(len(real_trajs)),
            key=lambda i: metric_fn(gen, real_trajs[i])
        )
        covered.add(best_real)
    return len(covered) / len(real_trajs)


def mean_distance_to_closest(gen_trajs, real_trajs, metric_fn=dtw_distance):
    dists = []
    for gen in gen_trajs:
        min_d = min(metric_fn(gen, real) for real in real_trajs)
        dists.append(min_d)
    return float(np.mean(dists))


def average_displacement_error(pred_trajs, true_trajs):
    errors = []
    for pred, true in zip(pred_trajs, true_trajs):
        min_len = min(len(pred), len(true))
        diff = pred[:min_len] - true[:min_len]
        ade = np.sqrt((diff ** 2).sum(axis=1)).mean()
        errors.append(ade)
    return float(np.mean(errors))


def final_displacement_error(pred_trajs, true_trajs):
    errors = []
    for pred, true in zip(pred_trajs, true_trajs):
        min_len = min(len(pred), len(true))
        fde = np.sqrt(((pred[min_len - 1] - true[min_len - 1]) ** 2).sum())
        errors.append(fde)
    return float(np.mean(errors))

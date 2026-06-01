import argparse
import json
import os
import numpy as np

from elephantgraph.evaluation.spatial_metrics import (
    hausdorff_distance,
    average_displacement_error,
    final_displacement_error,
    coverage,
)
from elephantgraph.evaluation.distributional_metrics import (
    speed_distribution_match,
    turning_angle_distribution_match,
    density_pattern_score,
    r_coefficient,
)
from elephantgraph.evaluation.kinematic_metrics import (
    compute_kinematics_from_traj,
    speed_mean_match,
    movement_consistency_score,
)

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated trajectories")
    parser.add_argument("--generated", type=str,
                        default=os.path.join(script_dir, "data", "processed", "generated_trajectories.npy"),
                        help="Path to generated trajectories .npy")
    parser.add_argument("--real", type=str,
                        default=os.path.join(script_dir, "data", "processed", "windows", "test_windows.npy"),
                        help="Path to real test windows .npy")
    parser.add_argument("--output", type=str,
                        default=os.path.join(script_dir, "data", "processed", "evaluation_report.json"),
                        help="Output JSON report path")
    args = parser.parse_args()

    gen = np.load(args.generated, allow_pickle=True)
    real_windows = np.load(args.real, allow_pickle=True)

    gen_trajs = [g for g in gen]
    real_trajs = [w['gps'] if isinstance(w, dict) else np.stack([w['lon'], w['lat']], axis=-1)
                  for w in real_windows]

    results = {}

    results['average_displacement_error'] = average_displacement_error(
        gen_trajs[:100], real_trajs[:100]
    )
    results['final_displacement_error'] = final_displacement_error(
        gen_trajs[:100], real_trajs[:100]
    )

    results['coverage'] = coverage(real_trajs[:50], gen_trajs[:50])

    gen_kin = [compute_kinematics_from_traj(t) for t in gen_trajs[:50]]
    real_kin = [compute_kinematics_from_traj(t) for t in real_trajs[:50]]

    all_gen_speeds = np.concatenate([k[0] for k in gen_kin])
    all_real_speeds = np.concatenate([k[0] for k in real_kin])

    speed_match = speed_distribution_match(all_real_speeds, all_gen_speeds)
    results['speed_distribution'] = speed_match

    all_gen_turns = np.concatenate([np.degrees(k[1]) for k in gen_kin if len(k[1]) > 0])
    all_real_turns = np.concatenate([np.degrees(k[1]) for k in real_kin if len(k[1]) > 0])

    if len(all_gen_turns) > 0 and len(all_real_turns) > 0:
        turn_match = turning_angle_distribution_match(all_real_turns, all_gen_turns)
        results['turning_angle_distribution'] = turn_match

    try:
        results['density_pattern_score'] = density_pattern_score(
            real_trajs[:50], gen_trajs[:50]
        )
    except Exception:
        results['density_pattern_score'] = None

    try:
        results['r_coefficient'] = r_coefficient(real_trajs[:50], gen_trajs[:50])
    except Exception:
        results['r_coefficient'] = None

    speed_mean = speed_mean_match(real_trajs[:50], gen_trajs[:50])
    results['speed_mean_match'] = speed_mean

    movement_consistency = movement_consistency_score(gen_trajs[:50])
    results['movement_consistency'] = movement_consistency

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"Evaluation report saved to {args.output}")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

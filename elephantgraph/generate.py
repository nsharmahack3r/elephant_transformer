import argparse
import numpy as np
import os
import pickle

from elephantgraph.inference.generator import ElephantTrajectoryGenerator

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories with ElephantGraph")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(script_dir, "checkpoints", "fine", "best_model.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of trajectories to generate")
    parser.add_argument("--behavior", type=str, default="MOVE",
                        choices=["MOVE", "STOP"])
    parser.add_argument("--season", type=str, default="dry",
                        choices=["dry", "wet"])
    parser.add_argument("--time-of-day", type=str, default="morning",
                        choices=["night", "morning", "afternoon", "evening"])
    parser.add_argument("--lulc", type=int, default=10)
    parser.add_argument("--human-settle", action="store_true")
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--scaler-dir", type=str,
                        default=os.path.join(script_dir, "data", "scalers"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(script_dir, "data", "processed", "generated_trajectories.npy"),
                        help="Output file path for generated trajectories")
    parser.add_argument("--config", type=str,
                        default=os.path.join(script_dir, "configs", "fine_config.yaml"),
                        help="Path to config YAML (for model architecture)")
    parser.add_argument("--micro-batch", type=int, default=8,
                        help="Batch size for generation (reduce if OOM)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gps_scaler = None
    scaler_path = os.path.join(args.scaler_dir, "gps_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            gps_scaler = pickle.load(f)

    generator = ElephantTrajectoryGenerator(
        checkpoint_path=args.checkpoint,
        scaler_gps=gps_scaler,
        config_path=args.config,
        device=args.device
    )

    trajectories = generator.generate(
        n_trajectories=args.n,
        behavior=args.behavior,
        season=args.season,
        time_of_day=args.time_of_day,
        lulc=args.lulc,
        human_settle=args.human_settle,
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, trajectories)
    print(f"Generated {args.n} trajectories saved to {args.output}")


if __name__ == "__main__":
    main()

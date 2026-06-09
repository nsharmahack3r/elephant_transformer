import argparse
import os
import sys

import numpy as np
import yaml

from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion
from elephantgraph.models.coarse_generator import CoarseGenerator
from elephantgraph.training.dataset import ElephantWindowDataset
from elephantgraph.training.dataset_coarse import CoarseWindowDataset
from elephantgraph.training.trainer_fine import train_fine_model
from elephantgraph.training.trainer_coarse import train_coarse_model
from elephantgraph.preprocessing.scalers import load_scalers

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Train ElephantGraph")
    parser.add_argument("--config", type=str,
                        default=os.path.join(script_dir, "configs", "fine_config.yaml"),
                        help="Path to config YAML file")
    parser.add_argument("--level", type=str, default="fine",
                        choices=["fine", "coarse"],
                        help="Training level")
    parser.add_argument("--train-data", type=str,
                        default=os.path.join(script_dir, "data", "processed", "windows", "train_windows.npy"))
    parser.add_argument("--val-data", type=str,
                        default=os.path.join(script_dir, "data", "processed", "windows", "val_windows.npy"))
    parser.add_argument("--scaler-dir", type=str,
                        default=os.path.join(script_dir, "data", "scalers"))
    parser.add_argument("--hourly-csv", type=str,
                        default=os.path.join(script_dir, "data", "processed", "hourly", "hourly.csv"))
    parser.add_argument("--h3-graph-dir", type=str,
                        default=os.path.join(script_dir, "data", "processed", "h3_graph"))
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/fine/best_model.pt)")
    args = parser.parse_args()

    config_path = args.config
    if args.level == "coarse" and config_path == os.path.join(script_dir, "configs", "fine_config.yaml"):
        config_path = os.path.join(script_dir, "configs", "coarse_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.level == "fine":
        train_windows = list(np.load(args.train_data, allow_pickle=True))
        val_windows = list(np.load(args.val_data, allow_pickle=True))

        scalers = load_scalers(args.scaler_dir) if os.path.exists(args.scaler_dir) else (None, None, None)
        gps_scaler, kin_scaler, env_scaler = scalers

        train_dataset = ElephantWindowDataset(
            train_windows, gps_scaler, kin_scaler, env_scaler
        )
        val_dataset = ElephantWindowDataset(
            val_windows, gps_scaler, kin_scaler, env_scaler
        )

        model = ElephantFineDiffusionTransformer(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dropout=config['model'].get('dropout', 0.1),
            max_seq_len=config['model'].get('max_seq_len', 200),
        )

        diffusion = DDIMDiffusion(
            T=config['diffusion']['T'],
            S=config['diffusion']['S'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
        )

        train_fine_model(
            model, diffusion, train_dataset, val_dataset,
            config['training'],
            resume_from=args.resume,
        )

    elif args.level == "coarse":
        train_dataset = CoarseWindowDataset(
            args.hourly_csv, args.h3_graph_dir,
            window_size=config['data'].get('window_size', 24),
            stride=config['data'].get('stride', 12),
            split="train",
            train_frac=config['data'].get('train_frac', 0.8),
            val_frac=config['data'].get('val_frac', 0.1),
        )
        val_dataset = CoarseWindowDataset(
            args.hourly_csv, args.h3_graph_dir,
            window_size=config['data'].get('window_size', 24),
            stride=config['data'].get('stride', 12),
            split="val",
            train_frac=config['data'].get('train_frac', 0.8),
            val_frac=config['data'].get('val_frac', 0.1),
        )

        model = CoarseGenerator(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers_coarse'],
            dropout=config['model'].get('dropout', 0.1),
            num_h3_nodes=train_dataset.num_h3_nodes,
            num_latent_entries=config['model'].get('num_latent_entries', 16),
            num_seasons=config['model'].get('num_seasons', 2),
        )

        train_coarse_model(
            model, train_dataset, val_dataset, config['training']
        )


if __name__ == "__main__":
    main()

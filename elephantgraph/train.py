import argparse
import yaml
import os
import sys

from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion
from elephantgraph.training.dataset import ElephantWindowDataset
from elephantgraph.training.trainer_fine import train_fine_model
from elephantgraph.preprocessing.scalers import load_scalers


def main():
    parser = argparse.ArgumentParser(description="Train ElephantGraph")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--level", type=str, default="fine",
                        choices=["fine", "coarse"],
                        help="Training level")
    parser.add_argument("--train-data", type=str,
                        default="data/processed/windows/train_windows.npy")
    parser.add_argument("--val-data", type=str,
                        default="data/processed/windows/val_windows.npy")
    parser.add_argument("--scaler-dir", type=str,
                        default="data/scalers/")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
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
            config['training']
        )


if __name__ == "__main__":
    import numpy as np
    main()

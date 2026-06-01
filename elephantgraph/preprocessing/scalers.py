import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fit_scalers(train_windows):
    all_gps = []
    all_kin = []
    all_env = []

    for w in train_windows:
        gps = np.stack([w['lon'], w['lat']], axis=-1)
        all_gps.append(gps)

        kin = np.stack([
            w['speed'], w['accel'], w['turning'],
            w['bearing'], w['persist'], w['step']
        ], axis=-1)
        all_kin.append(kin)

        env = np.stack([
            w['ndvi'], w['evi'], w['lst'],
            w['elev'], w['slope'], w['water']
        ], axis=-1)
        all_env.append(env)

    all_gps = np.concatenate(all_gps, axis=0)
    all_kin = np.concatenate(all_kin, axis=0)
    all_env = np.concatenate(all_env, axis=0)

    gps_scaler = MinMaxScaler(feature_range=(-1, 1))
    gps_scaler.fit(all_gps)

    kin_scaler = StandardScaler()
    kin_scaler.fit(all_kin)

    env_scaler = StandardScaler()
    env_scaler.fit(all_env)

    return gps_scaler, kin_scaler, env_scaler


def save_scalers(gps_scaler, kin_scaler, env_scaler, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'gps_scaler.pkl'), 'wb') as f:
        pickle.dump(gps_scaler, f)
    with open(os.path.join(output_dir, 'kinematic_scaler.pkl'), 'wb') as f:
        pickle.dump(kin_scaler, f)
    with open(os.path.join(output_dir, 'env_scaler.pkl'), 'wb') as f:
        pickle.dump(env_scaler, f)


def load_scalers(input_dir):
    import os
    with open(os.path.join(input_dir, 'gps_scaler.pkl'), 'rb') as f:
        gps_scaler = pickle.load(f)
    with open(os.path.join(input_dir, 'kinematic_scaler.pkl'), 'rb') as f:
        kin_scaler = pickle.load(f)
    with open(os.path.join(input_dir, 'env_scaler.pkl'), 'rb') as f:
        env_scaler = pickle.load(f)
    return gps_scaler, kin_scaler, env_scaler


def main():
    import argparse
    import os
    import pandas as pd
    from elephantgraph.preprocessing.windowing import create_windows

    parser = argparse.ArgumentParser(description="Fit and save scalers from training windows")
    parser.add_argument("--clean-data", type=str, required=True,
                        help="Path to cleaned CSV file from elephantgraph.preprocessing.clean")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save scaler .pkl files")
    args = parser.parse_args()

    df = pd.read_csv(args.clean_data, parse_dates=['timestamp'])
    windows = create_windows(df)
    print(f"Loaded {len(windows)} windows for scaler fitting")

    gps_scaler, kin_scaler, env_scaler = fit_scalers(windows)
    save_scalers(gps_scaler, kin_scaler, env_scaler, args.output)
    print(f"Scalers saved to {args.output}/")


if __name__ == "__main__":
    main()

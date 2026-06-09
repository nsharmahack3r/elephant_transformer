import numpy as np
import pandas as pd

WINDOW_SIZE = 200
STRIDE = 100


def create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    all_windows = []

    for elephant_id, group in df.groupby('elephant_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        n = len(group)

        for start in range(0, n - window_size, stride):
            window = group.iloc[start: start + window_size]

            if window.get('has_gap_before') is not None and window['has_gap_before'].any():
                continue

            kin_feats = {}
            kin_map = {
                'speed_kmh': 'speed', 'acceleration': 'accel',
                'turning_angle': 'turning', 'bearing': 'bearing',
                'dir_persistence': 'persist', 'step_dist_m': 'step'
            }
            for col, key in kin_map.items():
                if col in window.columns:
                    kin_feats[key] = window[col].values.astype(np.float32)
                else:
                    kin_feats[key] = np.zeros(window_size, dtype=np.float32)

            env_feats = {}
            env_map = {
                'NDVI': 'ndvi', 'EVI': 'evi', 'LST_celsius': 'lst',
                'elevation_m': 'elev', 'slope_deg': 'slope',
                'water_occ_1km': 'water'
            }
            for col, key in env_map.items():
                if col in window.columns:
                    env_feats[key] = window[col].values.astype(np.float32)
                else:
                    env_feats[key] = np.zeros(window_size, dtype=np.float32)

            sample = {
                'lon': window['longitude'].values.astype(np.float32),
                'lat': window['latitude'].values.astype(np.float32),
                **kin_feats,
                **env_feats,
                'behavior': int(window['behavior_code'].mode()[0]) if 'behavior_code' in window.columns else 0,
                'season': int(window['season_code'].iloc[0]) if 'season_code' in window.columns else 0,
                'time_of_day': int(window['time_of_day_encoded'].mode()[0]) if 'time_of_day_encoded' in window.columns else 0,
                'lulc': int(window['LULC_class'].mode()[0]) if 'LULC_class' in window.columns else 0,
                'human_settle': int(window['human_settle'].mode()[0]) if 'human_settle' in window.columns else 0,
                'move_type': int(window['movement_type_code'].iloc[0]) if 'movement_type_code' in window.columns else 0,
                'nsd': float(window['NSD'].iloc[-1]) if 'NSD' in window.columns else 0.0,
                'dist_origin': float(window['dist_from_origin_m'].iloc[-1]) if 'dist_from_origin_m' in window.columns else 0.0,
                'elephant_id': elephant_id,
                'start_time': str(window['timestamp'].iloc[0]),
            }
            all_windows.append(sample)

    return all_windows


def split_temporal(windows, train_frac=0.8, val_frac=0.1):
    """
    80/10/10 temporal split within each elephant.

    Windows are already ordered by start_time within each elephant
    (create_windows iterates in sorted order). We split the *window list*
    per elephant at the 80% and 90% marks so no window ever crosses a
    boundary — the positional split is a clean approximation of a time
    boundary without re-parsing timestamps.
    """
    from collections import defaultdict
    by_elephant = defaultdict(list)
    for w in windows:
        by_elephant[w['elephant_id']].append(w)

    train, val, test = [], [], []
    for eid, ws in by_elephant.items():
        n = len(ws)
        i_val  = int(n * train_frac)
        i_test = int(n * (train_frac + val_frac))
        train.extend(ws[:i_val])
        val.extend(ws[i_val:i_test])
        test.extend(ws[i_test:])
        print(f"  {eid}: {n} windows -> "
              f"train={i_val}  val={i_test-i_val}  test={n-i_test}")

    return train, val, test


def save_windows(windows, path):
    np.save(path, np.array(windows, dtype=object), allow_pickle=True)


def main():
    import argparse
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="Create sliding windows from cleaned GPS data")
    parser.add_argument("--input", type=str,
                        default=os.path.join(project_dir, "data", "processed", "clean.csv"),
                        help="Path to cleaned CSV file")
    parser.add_argument("--output", type=str,
                        default=os.path.join(project_dir, "data", "processed", "windows"),
                        help="Directory to write train/val/test .npy files")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"Window size in timesteps (default: {WINDOW_SIZE})")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Stride between windows (default: {STRIDE})")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of each elephant's windows used for training (default: 0.8)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction used for validation (default: 0.1); remainder = test")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['timestamp'])
    windows = create_windows(df, window_size=args.window_size, stride=args.stride)
    print(f"Created {len(windows)} total windows")

    print("Temporal 80/10/10 split per elephant:")
    train, val, test = split_temporal(windows,
                                      train_frac=args.train_frac,
                                      val_frac=args.val_frac)
    print(f"Total: {len(train)} train  {len(val)} val  {len(test)} test")

    os.makedirs(args.output, exist_ok=True)
    save_windows(train, os.path.join(args.output, "train_windows.npy"))
    save_windows(val,   os.path.join(args.output, "val_windows.npy"))
    save_windows(test,  os.path.join(args.output, "test_windows.npy"))
    print(f"Windows saved to {args.output}/")


if __name__ == "__main__":
    main()

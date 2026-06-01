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


def split_by_elephant(windows, val_elephant='AG005', test_elephant=None):
    train = [w for w in windows if w['elephant_id'] not in [val_elephant, test_elephant]]
    val = [w for w in windows if w['elephant_id'] == val_elephant]
    test = [w for w in windows if w['elephant_id'] == test_elephant]
    return train, val, test


def save_windows(windows, path):
    np.save(path, np.array(windows, dtype=object), allow_pickle=True)

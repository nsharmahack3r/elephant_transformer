import pandas as pd
import numpy as np


def remove_gps_outliers(df, max_speed_kmh=70):
    df = df.sort_values(['elephant_id', 'timestamp'])
    df = df[df['speed_kmh'] < max_speed_kmh]
    return df.reset_index(drop=True)


def flag_large_gaps(df, gap_threshold_sec=300):
    df = df.sort_values(['elephant_id', 'timestamp'])
    df['time_diff_sec'] = df.groupby('elephant_id')['timestamp'].diff().dt.total_seconds()
    df['has_gap_before'] = (
        df.groupby('elephant_id')['time_diff_sec']
          .transform(lambda x: x > gap_threshold_sec)
    )
    return df


def impute_missing_kinematics(df):
    kinematic_cols = [
        'speed_kmh', 'acceleration', 'turning_angle',
        'bearing', 'dir_persistence', 'step_dist_m'
    ]
    df = df.sort_values(['elephant_id', 'timestamp'])
    df[kinematic_cols] = (
        df.groupby('elephant_id')[kinematic_cols]
          .transform(lambda x: x.ffill().fillna(0))
    )
    return df


def encode_categoricals(df):
    from elephantgraph.preprocessing.schema import ENCODING_MAPS
    for col, mapping in ENCODING_MAPS.items():
        if col in df.columns:
            original = df[col].copy()
            for string_val, idx in mapping.items():
                original = original.str.upper().str.strip()
            for string_val, idx in mapping.items():
                df.loc[df[col].str.upper().str.strip() == string_val.upper(), col + '_code'] = idx
    if 'behavior_code' not in df.columns and 'behavior' in df.columns:
        df['behavior_code'] = df['behavior'].map({'STOP': 0, 'MOVE': 1}).fillna(0).astype(int)
    if 'season_code' not in df.columns and 'season' in df.columns:
        df['season_code'] = df['season'].map({'dry': 0, 'wet': 1}).fillna(0).astype(int)
    if 'time_of_day_encoded' not in df.columns and 'time_of_day' in df.columns:
        tod_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
        df['time_of_day_encoded'] = df['time_of_day'].map(tod_map).fillna(0).astype(int)
    if 'movement_type_code' not in df.columns and 'movement_type' in df.columns:
        df['movement_type_code'] = pd.factorize(df['movement_type'])[0]
    return df


def run_cleaning_pipeline(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    df = remove_gps_outliers(df)
    df = flag_large_gaps(df)
    df = impute_missing_kinematics(df)
    df = encode_categoricals(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data: {len(df)} rows -> {output_path}")
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean raw elephant GPS data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write cleaned CSV")
    args = parser.parse_args()
    run_cleaning_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()

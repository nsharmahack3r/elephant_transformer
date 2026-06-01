import os
import pandas as pd


def hourly_resample(df, output_path=None):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    agg_dict = {
        'longitude': 'mean',
        'latitude': 'mean',
        'speed_kmh': 'mean',
        'acceleration': 'mean',
        'turning_angle': 'mean',
        'bearing': 'mean',
        'dir_persistence': 'mean',
        'step_dist_m': 'sum',
        'NDVI': 'mean',
        'EVI': 'mean',
        'LST_celsius': 'mean',
        'elevation_m': 'mean',
        'slope_deg': 'mean',
        'water_occ_1km': 'mean',
        'NSD': 'last',
        'dist_from_origin_m': 'last',
        'hr_95_km2': 'last',
        'hr_50_km2': 'last',
    }

    categorical_cols = [
        'elephant_id', 'behavior_code', 'season_code',
        'time_of_day_encoded', 'LULC_class', 'movement_type_code'
    ]
    for col in categorical_cols:
        if col in df.columns:
            agg_dict[col] = 'first'

    hourly = (
        df.groupby('elephant_id')
          .resample('1h')
          .agg(agg_dict)
          .dropna(subset=['longitude', 'latitude'])
          .reset_index()
    )

    hourly['hour'] = pd.to_datetime(hourly['timestamp']).dt.hour
    hourly['month'] = pd.to_datetime(hourly['timestamp']).dt.month

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        hourly.to_csv(output_path, index=False)
        print(f"Hourly resampled: {len(hourly)} rows -> {output_path}")

    return hourly


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Hourly resample cleaned GPS data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to cleaned CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write hourly-resampled CSV")
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=['timestamp'])
    hourly_resample(df, output_path=args.output)


if __name__ == "__main__":
    import os
    main()

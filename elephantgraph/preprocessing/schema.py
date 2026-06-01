FEATURE_ROLES = {
    'gps_output': [
        'longitude',
        'latitude',
    ],
    'kinematic_tokens': [
        'speed_kmh',
        'acceleration',
        'turning_angle',
        'bearing',
        'dir_persistence',
        'step_dist_m',
    ],
    'env_tokens': [
        'NDVI',
        'EVI',
        'LST_celsius',
        'elevation_m',
        'slope_deg',
        'water_occ_1km',
    ],
    'adaln_conditions': [
        'behavior_code',
        'season_code',
        'time_of_day',
        'LULC_class',
        'human_settle',
        'movement_type_code',
    ],
    'range_context': [
        'NSD',
        'dist_from_origin_m',
        'hr_95_km2',
        'hr_50_km2',
    ],
    'temporal': [
        'hour',
        'month',
        'days_elapsed',
    ],
    'drop': [
        'event_id',
        'elephant_id',
        'timestamp',
        'season',
        'behavior',
        'movement_type',
        'rolling_speed_mean',
        'rolling_turn_std',
        'rolling_step_mean',
        'aspect_deg',
    ],
}

ENCODING_MAPS = {
    'time_of_day': {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3},
    'season':      {'dry': 0, 'wet': 1},
    'behavior':    {'STOP': 0, 'MOVE': 1},
}

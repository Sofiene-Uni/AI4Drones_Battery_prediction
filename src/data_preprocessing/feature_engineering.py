import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in meters between two GPS points.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c



def calculate_temporal_features(df):
    timestamp_col = 'timestamp'
    gps_col = 'sensor_gps_time_utc_usec'
    
    # Ensure columns exist
    if timestamp_col not in df.columns or gps_col not in df.columns:
        raise ValueError(f"Columns '{timestamp_col}' or '{gps_col}' not found in DataFrame")
    
    # Reorder columns: boot timestamp first, GPS UTC second, then the rest
    cols = [timestamp_col, gps_col] + [c for c in df.columns if c not in [timestamp_col, gps_col]]
    df = df[cols]

    # Convert GPS UTC to datetime
    gps_time = pd.to_datetime(df[gps_col], unit='us', errors='coerce')
    
    # Insert temporal features next to GPS UTC
    idx = df.columns.get_loc(gps_col)
    df.insert(idx + 1, 'date', gps_time.dt.date)
    df.insert(idx + 2, 'hour', gps_time.dt.hour)
    df.insert(idx + 3, 'minute', gps_time.dt.minute)
    df.insert(idx + 4, 'second', gps_time.dt.second)
    
    # Calculate time difference between consecutive points (in seconds)
    df.insert(idx + 5, 'delta_time_s', gps_time.diff().dt.total_seconds().fillna(0))
    
    # Calculate cumulative flight time since the first point (in seconds)
    df.insert(idx + 6, 'cumulative_flight_time_s', gps_time.sub(gps_time.iloc[0]).dt.total_seconds())
    
    return df



def calculate_gps_distance(df):
    """
    Calculate 3D segment distance based on GPS coordinates and altitude, then fill the DataFrame columns.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    lat_col = 'vehicle_gps_position_lat'
    lon_col = 'vehicle_gps_position_lon'
    alt_col = 'vehicle_gps_position_alt'  # make sure this exists

    # Convert units
    latitudes = df[lat_col] / 1e7
    longitudes = df[lon_col] / 1e7
    altitudes = df[alt_col] / 1e3  # mm -> meters

    # Initialize columns
    df['segment_distance_gsp'] = 0.0
    df['total_distance_gps'] = 0.0

    # Iterate row by row
    for i in range(1, len(df)):
        horizontal_distance = haversine(
            latitudes.iloc[i-1], longitudes.iloc[i-1],
            latitudes.iloc[i], longitudes.iloc[i]
        )
        vertical_distance = altitudes.iloc[i] - altitudes.iloc[i-1]
        distance_3d = np.sqrt(horizontal_distance**2 + vertical_distance**2)

        df.at[i, 'segment_distance_gps'] = distance_3d
        df.at[i, 'total_distance_gps'] = df.at[i-1, 'total_distance_gps'] + distance_3d

    return df


def calculate_xyz_distance(df):
    """
    Calculate 3D segment distance based on local x, y, z coordinates
    and fill the DataFrame columns as 'segment_distance_xyz' and 'total_distance_xyz'.
    """
    df = df.copy()

    # Column names
    x_col = 'vehicle_local_position_x'
    y_col = 'vehicle_local_position_y'
    z_col = 'vehicle_local_position_z'

    # Initialize columns
    df['segment_distance_xyz'] = 0.0
    df['total_distance_xyz'] = 0.0

    # Iterate row by row
    for i in range(1, len(df)):
        dx = df.at[i, x_col] - df.at[i-1, x_col]
        dy = df.at[i, y_col] - df.at[i-1, y_col]
        dz = df.at[i, z_col] - df.at[i-1, z_col]

        distance_xyz = np.sqrt(dx**2 + dy**2 + dz**2)
        df.at[i, 'segment_distance_xyz'] = distance_xyz
        df.at[i, 'total_distance_xyz'] = df.at[i-1, 'total_distance_xyz'] + distance_xyz

    return df


def check_distance_discrepancy(df):
    """
    Compare total distances from GPS vs local XYZ and return a nicely formatted report.
    
    Parameters:
        df : DataFrame
            Must contain 'total_distance_gps' and 'total_distance_xyz' columns.
            
    Returns:
        dict : discrepancy report with floats and percentage string
    """
    total_gps = float(df['total_distance_gps'].iloc[-1])
    total_xyz = float(df['total_distance_xyz'].iloc[-1])
    
    discrepancy = abs(total_gps - total_xyz)
    percentage_diff = (discrepancy / total_gps * 100) if total_gps != 0 else float('nan')
    
    report = {
        'total_distance_gps_m': round(total_gps, 2),
        'total_distance_xyz_m': round(total_xyz, 2),
        'discrepancy_m': round(discrepancy, 2),
        'percentage_difference': f"{round(percentage_diff, 2)}%" if not np.isnan(percentage_diff) else "NaN"
    }
    
    return report


def create_features(df):
    """
    Master function to generate all necessary features.
    """
    # Temporal features
    df = calculate_temporal_features(df)
    df = calculate_gps_distance(df)
    df = calculate_xyz_distance(df)
    
    # report = check_distance_discrepancy(df)
    # print(report)
    

    return df

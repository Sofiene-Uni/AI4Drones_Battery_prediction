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
    df = df.copy()
    
    lat_col = 'vehicle_gps_position_lat'
    lon_col = 'vehicle_gps_position_lon'
    alt_col = 'vehicle_gps_position_alt'

    # Fill missing GPS/altitude values with 0
    df[[lat_col, lon_col, alt_col]] = df[[lat_col, lon_col, alt_col]].fillna(0)

    # Convert units
    latitudes = df[lat_col] / 1e7
    longitudes = df[lon_col] / 1e7
    altitudes = df[alt_col] / 1e3  # mm -> meters

    # Initialize columns
    df['segment_distance_gps'] = 0.0
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




def latlonalt_to_xyz(lat, lon, alt):
    """
    Convert lat, lon (degrees) and altitude (m) to Earth-Centered, Earth-Fixed (ECEF) XYZ.
    Uses WGS84 ellipsoid approximation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis (m)
    e2 = 6.69437999014e-3  # eccentricity squared

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return X, Y, Z



def calculate_xyz_speeds(df):
    """
    Add columns for x, y, z distances (m) and speeds (m/s)
    based on GPS coordinates and time deltas.
    Requires: delta_time_s column and GPS columns.
    """
    df = df.copy()

    lat = df['vehicle_gps_position_lat'] / 1e7
    lon = df['vehicle_gps_position_lon'] / 1e7
    alt = df['vehicle_gps_position_alt'] / 1e3  # mm -> meters

    # Convert to Cartesian coordinates (ECEF approx)
    coords = np.array([latlonalt_to_xyz(la, lo, al) for la, lo, al in zip(lat, lon, alt)])
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Differences (per segment)
    dX = np.diff(X, prepend=X[0])
    dY = np.diff(Y, prepend=Y[0])
    dZ = np.diff(Z, prepend=Z[0])

    # Time deltas
    dt = df['delta_time_s'].replace(0, np.nan)  # avoid div by zero

    # Store distances
    df['distance_x'] = dX
    df['distance_y'] = dY
    df['distance_z'] = dZ

    # Speeds (m/s)
    df['speed_x'] = dX / dt
    df['speed_y'] = dY / dt
    df['speed_z'] = dZ / dt

    # Replace NaN/inf with 0
    df[['distance_x','distance_y','distance_z','speed_x','speed_y','speed_z']] = (
        df[['distance_x','distance_y','distance_z','speed_x','speed_y','speed_z']]
        .replace([np.nan, np.inf, -np.inf], 0)
    )

    return df


def calculate_cumulative_current(df):
    """
    Add a column that accumulates the 'current' over rows.
    """
    current_col = 'battery_status_current_a'  # replace with the actual column name
    if current_col not in df.columns:
        raise ValueError(f"Column '{current_col}' not found in DataFrame")

    df['cumulative_current'] = df[current_col].cumsum()
    return df


def create_features(df):
    """
    Master function to generate all necessary features.
    """
    # Temporal features
    df = calculate_temporal_features(df)
    df = calculate_gps_distance(df)
    df = calculate_xyz_speeds(df)

    return df

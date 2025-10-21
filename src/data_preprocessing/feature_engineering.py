import numpy as np
import pandas as pd
from src.utils.weather import get_weather_data




# ----------------------------
# Basic Utilities
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two GPS points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def latlonalt_to_xyz(lat, lon, alt):
    """Convert lat/lon/alt to ECEF XYZ coordinates (meters)."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return X, Y, Z



def sample_lines(df, n_samples=10):
    """
    Sample n_samples rows evenly from a DataFrame.
    Always includes the first and last row.
    """
    total_rows = len(df)
    print("sampling file ..")

    if total_rows <= n_samples:
        return df.copy()

    # Generate evenly spaced indices including first and last
    indices = np.linspace(0, total_rows - 1, n_samples, dtype=int)

    # Ensure first and last indices are included (just in case rounding issues)
    indices[0] = 0
    indices[-1] = total_rows - 1

    return df.iloc[indices].copy()


# ----------------------------
# Alight Unit
# ----------------------------

def align_units(df):
    """
    Align GPS-related units:
    - Convert latitude and longitude from 1e7 to degrees
    - Convert altitude from millimeters to meters
    """
    if "vehicle_gps_position_lat" in df.columns:
        df["vehicle_gps_position_lat"] = df["vehicle_gps_position_lat"] / 1e7

    if "vehicle_gps_position_lon" in df.columns:
        df["vehicle_gps_position_lon"] = df["vehicle_gps_position_lon"] / 1e7

    if "vehicle_gps_position_alt" in df.columns:
        df["vehicle_gps_position_alt"] = df["vehicle_gps_position_alt"] / 1000.0

    return df


# ----------------------------
# Temporal Features
# ----------------------------
def calculate_temporal_features(df):
    """Generate timestamp-based features."""
    gps_col = 'sensor_gps_time_utc_usec'
    if gps_col not in df.columns:
        raise ValueError(f"Column '{gps_col}' not found")

    gps_time = pd.to_datetime(df[gps_col], unit='us', errors='coerce')
    df['date'] = gps_time.dt.date
    df['hour'] = gps_time.dt.hour
    df['minute'] = gps_time.dt.minute
    df['second'] = gps_time.dt.second + gps_time.dt.microsecond / 1e6
    df['delta_time_s'] = gps_time.diff().dt.total_seconds().fillna(0.0)
    df['cumulative_flight_time_s'] = (gps_time - gps_time.iloc[0]).dt.total_seconds()
    return df



# ----------------------------
# GPS / Displacement Features
# ----------------------------
def calculate_gps_distance(df):
    """Compute 3D segment and cumulative distances."""
    lat, lon,alt = df['vehicle_gps_position_lat'], df['vehicle_gps_position_lon'], df['vehicle_gps_position_alt']
     
    df['segment_distance_gps'] = np.sqrt(
        haversine(lat.shift(1, fill_value=lat.iloc[0]), lon.shift(1, fill_value=lon.iloc[0]), lat, lon)**2 +
        (alt - alt.shift(1, fill_value=alt.iloc[0]))**2
    )
    df['total_distance_gps'] = df['segment_distance_gps'].cumsum()
    return df


def calculate_xyz_displacements(df):
    """Compute ECEF XYZ displacements and cumulative distances."""
    lat, lon, alt = df['vehicle_gps_position_lat'], df['vehicle_gps_position_lon'], df['vehicle_gps_position_alt']
    coords = np.array([latlonalt_to_xyz(la, lo, al) for la, lo, al in zip(lat, lon, alt)])
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    df['distance_x'] = abs ( np.diff(X, prepend=X[0]))
    df['distance_y'] = abs ( np.diff(Y, prepend=Y[0]))
    df['distance_z'] = abs (np.diff(Z, prepend=Z[0]))
    df['cumulative_x'] = np.cumsum(df['distance_x'])
    df['cumulative_y'] = np.cumsum(df['distance_y'])
    df['cumulative_z'] = np.cumsum(df['distance_z'])
    return df


def calculate_xyz_speeds(df):
    """Compute speeds along XYZ axes."""
    dt = df['delta_time_s'].replace(0, np.nan)
    for axis in ['x','y','z']:
        df[f'speed_{axis}'] = df[f'distance_{axis}'] / dt
    df[['speed_x','speed_y','speed_z']] = df[['speed_x','speed_y','speed_z']].fillna(0)
    return df

def calculate_segments_speeds(df):
    """
    Compute cumulative average speed up to each waypoint.
    avg_speed = total_distance_gps / cumulative_flight_time_s
    """
    # Avoid division by zero
    dt_cum = df['cumulative_flight_time_s'].replace(0, np.nan)
    df['average_speed'] = df['total_distance_gps'] / dt_cum
    df['average_speed'] = df['average_speed'].fillna(0)
    return df



# ----------------------------
# Kinematic Features
# ----------------------------
def calculate_kinematic_features(df):
    """Add total speed, horizontal/vertical speed, acceleration, heading, turn rate, climb rate."""
    df['speed_total'] = np.sqrt(df['speed_x']**2 + df['speed_y']**2 + df['speed_z']**2)
    df['speed_horizontal'] = np.sqrt(df['speed_x']**2 + df['speed_y']**2)
    df['speed_vertical'] = df['speed_z']

    # Acceleration
    dt = df['delta_time_s'].replace(0, np.nan)
    for axis in ['x','y','z']:
        df[f'accel_{axis}'] = df[f'speed_{axis}'].diff() / dt
    df['accel_total'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df[['accel_x','accel_y','accel_z','accel_total']] = df[['accel_x','accel_y','accel_z','accel_total']].fillna(0)

    # Heading & turn rate
    df['heading'] = np.degrees(np.arctan2(df['distance_y'], df['distance_x']))
    df['turn_rate'] = df['heading'].diff() / dt
    df['turn_rate'] = df['turn_rate'].fillna(0)

    # Climb rate
    df['climb_rate'] = df['distance_z'] / dt
    df['climb_rate'] = df['climb_rate'].fillna(0)
    return df


# ----------------------------
# Environmental Features
# ----------------------------
def add_environmental_features(df):
    """Add wind, temperature, humidity, air density and effective wind along path."""
    weather_data = df.apply(lambda row: get_weather_data(
        row['vehicle_gps_position_lat'],
        row['vehicle_gps_position_lon'],
        row['vehicle_gps_position_alt'],
        row['date'], row['hour'], row['minute'], row['second']
    ), axis=1)
    
    df[['wind_magnitude','wind_direction_deg','temperature_c','humidity_percent','air_density_kg_m3']] = pd.DataFrame(weather_data.tolist(), index=df.index)

    # Compute effective wind along path
    theta_rad = np.radians((df['wind_direction_deg'] + 180) % 360)
    wind_vec = np.column_stack((np.cos(theta_rad), np.sin(theta_rad)))
    path_vec = df[['distance_x','distance_y']].to_numpy()
    path_len = np.linalg.norm(path_vec, axis=1, keepdims=True)
    path_unit = np.divide(path_vec, path_len, out=np.zeros_like(path_vec), where=path_len!=0)
    df['effective_wind'] = np.einsum('ij,ij->i', wind_vec, path_unit) * df['wind_magnitude']
    return df


def remaining_battery(df, col='battery_status_remaining'):
    """
    Adds columns:
    - 'battery_variation': difference in battery_status_remaining between consecutive rows
    - 'previous_soc': SOC value of the previous row
      (first row is filled with second row to match first two rows)
    """
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    # Ensure we're working on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Row-to-row variation (difference)
    df['battery_variation'] = df[col].diff().fillna(0)

    # Previous SOC
    df['previous_soc'] = df[col].shift(1)

    # Fill first NaN with second row value
    if len(df) > 1:
        df.at[0, 'previous_soc'] = df[col].iloc[1]

    return df


def add_charge_column(df):
    """
    Add a column 'charge_As' representing the charge consumed per row (A·s).

    Parameters
    ----------
    """
    current_col='battery_status_current_a'
    time_col='delta_time_s'
    df['charge_As'] = df[current_col] * df[time_col]
    return df




# ----------------------------
# Master Feature Functions
# ----------------------------
def basic_features(df):
    """Generate temporal, GPS distance, and battery features."""
    df = align_units(df)
    df = calculate_temporal_features(df)
    df = calculate_gps_distance(df)
    df= add_charge_column(df)
    return df


def additional_features(df):
    """Generate kinematic and environmental features."""
    df = calculate_xyz_displacements(df)
    df= calculate_segments_speeds(df)
    
    #df = calculate_xyz_speeds(df)
    #df = calculate_kinematic_features(df)
    #df = add_environmental_features(df)
    return df

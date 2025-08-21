# src/data_curation/filters.py
from src.utils.config import load_config

def filter_empty_file(df):
    """Placeholder for empty file filter (currently does nothing)."""
    return True

def filter_voltage(df):
    """Placeholder for zero voltage filter (currently does nothing)."""
    return True

def filter_current(df):
    """Placeholder for no current filter (currently does nothing)."""
    return True

def filter_distance(df):
    """
    Returns True if the file passes the short distance filter.
    Compares the last entry of the 'total_distance' column with the threshold in config.
    """

    cfg = load_config()
    threshold_min = cfg.get("thresholds", {}).get("min_distance_m", 100)
    threshold_max = cfg.get("thresholds", {}).get("max_distance_m", 50000)
    last_distance = df['total_distance_gps'].iloc[-1]
    
    return (last_distance >= threshold_min and   last_distance <= threshold_max)

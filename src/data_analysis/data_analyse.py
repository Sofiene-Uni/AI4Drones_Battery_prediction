from src.utils.config import get_value
from pathlib import Path
import pandas as pd
from src.utils.reports import log_correlation_report
from src.utils.plotting import visualize_report
import numpy as np


def compute_correlation(all_dfs: list[pd.DataFrame]) -> None:
    """Helper function to compute and visualize correlation across multiple DataFrames."""
    if not all_dfs:
        return

    global_df = pd.concat(all_dfs, ignore_index=True)
    label_col = get_value("data.label_col")

    log_correlation_report(global_df, label_column=label_col)
    visualize_report("correlation")
    
    

def compute_flight_summary(all_dfs: list[pd.DataFrame], report_name="flight_summary") -> pd.DataFrame:
    """
    Compute key metrics per flight and overall mean/median across all flights:
    - total distance
    - total flight time
    - avg speed
    - avg instantaneous current
    - cumulative charge (A·s)
    - avg charge per meter (A·s/m)
    - avg charge per second (A·s/s)
    Save results to Excel.
    """
    results = []
    
    for idx, df in enumerate(all_dfs, start=1):
        # Ensure delta_time_s exists
        if 'delta_time_s' not in df.columns:
            df['delta_time_s'] = df['cumulative_flight_time_s'].diff().fillna(0)
        
        # Compute cumulative charge (A·s)
        if 'battery_status_current_a' in df.columns:
            df['segment_charge_As'] = df['battery_status_current_a'] * df['delta_time_s']
            df['cumulative_charge_As'] = df['segment_charge_As'].cumsum()
        else:
            df['segment_charge_As'] = 0
            df['cumulative_charge_As'] = 0

        total_distance = df["total_distance_gps"].iloc[-1] if "total_distance_gps" in df else np.nan
        total_flight_time = df["cumulative_flight_time_s"].iloc[-1] if "cumulative_flight_time_s" in df else np.nan
        avg_speed = total_distance / total_flight_time if total_distance and total_flight_time else np.nan
        avg_current_instant = df["battery_status_current_a"].mean() if "battery_status_current_a" in df else np.nan

        cumulative_charge = df["cumulative_charge_As"].iloc[-1]
        charge_per_meter = cumulative_charge / total_distance if total_distance and cumulative_charge else np.nan
        charge_per_second = cumulative_charge / total_flight_time if total_flight_time and cumulative_charge else np.nan

        results.append({
            "flight_id": idx,
            "total_distance_m": total_distance,
            "total_flight_time_s": total_flight_time,
            "avg_speed_mps": avg_speed,
            "avg_current_instant_A": avg_current_instant,
            "cumulative_charge_As": cumulative_charge,
            "charge_As_per_meter": charge_per_meter,
            "charge_As_per_second": charge_per_second,
        })
    
    results_df = pd.DataFrame(results)
    
    # Compute overall mean and median
    overall_mean = {"flight_id": "ALL_MEAN"}
    overall_median = {"flight_id": "ALL_MEDIAN"}
    metrics = ["total_distance_m", "total_flight_time_s", "avg_speed_mps",
               "avg_current_instant_A", "cumulative_charge_As", "charge_As_per_meter", "charge_As_per_second"]
    
    for m in metrics:
        overall_mean[m] = results_df[m].mean()
        overall_median[m] = results_df[m].median()
    
    # Append mean and median rows
    results_df = pd.concat([results_df, pd.DataFrame([overall_mean, overall_median])], ignore_index=True)
    
    # Save to Excel
    reports_dir = Path(get_value("paths.reports_dir"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    excel_path = reports_dir / f"{report_name}.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"✅ Flight summary saved to {excel_path}")
    
    return results_df




def run():
    processed_dir, logbook_path = get_value(
        ["paths.processed_dir", "paths.logbook_path"]
    )

    processed_dir = Path(processed_dir)
    logbook = pd.read_csv(logbook_path)

    all_dfs = []

    for i, row in enumerate(logbook.itertuples(), start=1):
        if not getattr(row, "valid", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping invalid file ID {row.id}")
            continue
        
        file_path = processed_dir / f"{row.id}.csv"
        df = pd.read_csv(file_path)
        all_dfs.append(df)

    compute_correlation(all_dfs)
    compute_flight_summary(all_dfs)
   

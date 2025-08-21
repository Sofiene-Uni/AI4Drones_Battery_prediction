import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import get_value  # your standalone get_value function

def log_training(epoch, avg_loss):
    """Log training results to CSV using config paths."""
    model_name = get_value("model.model_name")
    reports_dir = Path(get_value("paths.reports_dir"))
    log_path = reports_dir / f"{model_name}_training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")

    df = pd.DataFrame([[epoch, avg_loss]], columns=["epoch", "loss"])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

    print(f"✅ Epoch {epoch} loss logged to {log_path}")


def log_flight_report(flight_report):
    """Helper to save flight-level report to CSV."""
    model_name = get_value("model.model_name")
    reports_dir = Path(get_value("paths.reports_dir"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{model_name}_pred_vs_target.csv"

    flight_report.to_csv(report_path, index=False)
    print(f"✅ Flight report saved to {report_path}")
    return report_path


def log_evaluation(preds, targets, avg_loss):
    """Log evaluation results and save predictions vs targets to CSV."""
    model_name = get_value("model.model_name")
    reports_dir = Path(get_value("paths.reports_dir"))
    report_path = reports_dir / f"{model_name}_pred_vs_target.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation Loss (MSE): {avg_loss:.4f}")

    df_report = pd.DataFrame(
        np.hstack([targets, preds]),
        columns=["target" for _ in range(targets.shape[1])] +
                ["prediction" for _ in range(preds.shape[1])]
    )
    df_report.to_csv(report_path, index=False)

    print(f"✅ Predictions saved to {report_path}")


def postprocess_evaluation(all_preds, all_targets, file_ids, avg_loss, mode):
    if mode == "sample":
        log_evaluation(all_preds, all_targets, avg_loss)

    elif mode == "flight":
        df = pd.DataFrame({
            "file_id": file_ids,
            "prediction": all_preds.flatten(),
            "target": all_targets.flatten()
        })

        flight_report = (
            df.groupby("file_id", as_index=False)
              .agg(
                  prediction=("prediction", "sum"),
                  target=("target", "sum"),
                  num_samples=("prediction", "count")
              )
        )

        # Absolute error
        flight_report["abs_error"] = (flight_report["prediction"] - flight_report["target"]).abs()

        # Fractional percentage error
        flight_report["pct_error"] = np.where(
            flight_report["target"] != 0,
            flight_report["abs_error"] / flight_report["target"],
            np.nan
        )

        log_flight_report(flight_report)

    else:
        raise ValueError("mode must be 'sample' or 'flight'")


def log_correlation_report(df, label_column="label", report_name="correlation_report"):
    """Compute correlations with the label and save as CSV."""
    reports_dir = Path(get_value("paths.reports_dir"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")

    if label_column not in numeric_df.columns:
        print(f"⚠️ Label column '{label_column}' not found. Skipping correlation analysis.")
        return

    # Exclude time-related columns
    exclude = ["hour", "minute", "second", "timestamp", "sensor_gps_time_utc_usec", "battery_status_current_a"]
    numeric_df = numeric_df[[c for c in numeric_df.columns if not any(k in c.lower() for k in exclude)]]

    if label_column not in numeric_df.columns:
        numeric_df[label_column] = df[label_column]

    corr_series = numeric_df.corr(method="pearson")[label_column].sort_values(ascending=False)

    csv_path = reports_dir / f"{report_name}.csv"
    corr_series.to_csv(csv_path, header=True)
    print(f"✅ Correlation report saved to {csv_path}")

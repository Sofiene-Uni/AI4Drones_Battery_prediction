import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import get_value


def plot_report(df: pd.DataFrame, report_type: str, figsize=(10, 6)) -> plt.Figure:
    """
    Create a matplotlib figure from a DataFrame based on report type.
    Handles 'evaluation', 'training', and 'correlation' reports.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if report_type == "evaluation":
        ax.plot(df["target"], label="Target", alpha=0.7)
        ax.plot(df["prediction"], label="Prediction", alpha=0.7)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.set_title("Prediction vs Target")
        ax.legend()
    
    elif report_type == "training":
        ax.plot(df["epoch"], df["loss"], marker='o', label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss over Epochs")
        ax.legend()
    
    elif report_type == "correlation":
        label_column = get_value("plots.correlation_label", default="battery_status_current_a")
        df_sorted = df.sort_values(by=label_column)
        colors = df_sorted[label_column].apply(lambda x: "red" if x < 0 else "green")
        ax.barh(df_sorted.index, df_sorted[label_column], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted.index)
        ax.set_xlabel("Pearson correlation")
        ax.set_title(f"Correlation with '{label_column}'")
        ax.set_xlim(-1, 1)
    
    else:
        raise ValueError(f"Unknown report_type: {report_type}")
    
    plt.tight_layout()
    return fig


def visualize_report(report_type: str = "evaluation"):
    """
    Visualize a CSV report using config paths and parameters.
    Supports 'evaluation', 'training', and 'correlation'.
    """
    model_name = get_value("model.model_name")
    reports_dir = Path(get_value("paths.reports_dir", "").strip())

    # Determine the report file
    report_files = {
        "evaluation": f"{model_name}_pred_vs_target.csv",
        "training": f"{model_name}_training_log.csv",
        "correlation": "correlation_report.csv"
    }
    
    report_file = report_files.get(report_type)
    if not report_file:
        print(f"Unknown report_type: {report_type}")
        return
    
    report_path = reports_dir / report_file
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return

    df = pd.read_csv(report_path, index_col=0 if report_type == "correlation" else None)
    
    # Get plotting parameters from config
    figsize = tuple(get_value("plots.figsize", [10, 6]))
    
    # Create figure using the unified plotting function
    fig = plot_report(df, report_type, figsize=figsize)

    # Save figure
    fig_path = reports_dir / f"{model_name}_{report_type}_plot.png"
    fig.savefig(fig_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"✅ Visualization saved to {fig_path}")

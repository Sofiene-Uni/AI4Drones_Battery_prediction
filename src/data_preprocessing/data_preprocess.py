from pathlib import Path
import pandas as pd
from src.utils.config import  get_value
from src.utils.ulog_io import  extract_from_ulog
from src.data_preprocessing import  data_manip
from src.data_preprocessing import feature_engineering
from src.utils.reports import log_correlation_report
from src.utils.plotting import visualize_report

def run():
    """Run the data curation step on an existing logbook DataFrame and update status."""
    
    processed_dir = get_value("paths.processed_dir")
    logbook_path = get_value("paths.logbook_path")
    datasets_to_extract = get_value("datasets", {})

    logbook = pd.read_csv(logbook_path)
    processed_ids = []
    all_dfs = []  # collect all processed dfs for correlation

    for i, row in enumerate(logbook.itertuples(), start=1):
        if getattr(row, "processed", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping already processed file ID {row.id}")
            continue

        try:
            file_path = row.path
            file_id = row.id

            # Extract datasets from ulog
            datasets = extract_from_ulog(file_path, datasets_to_extract)
            fused_df = data_manip.fuse_datasets(datasets)
            
            # Feature engineering
            enhanced_df = feature_engineering.create_features(fused_df)
            
            # Save processed file
            output_path = Path(processed_dir) / f"{file_id}.csv"
            enhanced_df.to_csv(output_path, index=False)

            processed_ids.append(file_id)
            all_dfs.append(enhanced_df)
            print(f"[{i}/{len(logbook)}] ✅ Processed file ID {file_id}")

        except Exception as e:
            print(f"[{i}/{len(logbook)}] ⚠️ Skipping file ID {row.id}: {e}")

    # Compute correlation across all files
    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        log_correlation_report(global_df, label_column=get_value("data.label_col"))

    visualize_report("correlation")

    # Update status in logbook
    if processed_ids:
        logbook.loc[logbook["id"].isin(processed_ids), "processed"] = True
        logbook.to_csv(logbook_path, index=False)
        print(f"📓 Logbook updated: {len(processed_ids)} files marked as processed.")
    else:
        print("ℹ️ No files were successfully processed.")

 
        
        
    





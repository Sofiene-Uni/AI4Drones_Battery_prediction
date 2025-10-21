import pandas as pd
from pathlib import Path
from src.utils.config import get_value
from src.data_preprocessing import feature_engineering


def run():
    """Run preprocessing: feature engineering on curated & valid files only."""
    
    processed_dir, curated_dir, logbook_path = get_value(
        ["paths.processed_dir", "paths.curated_dir", "paths.logbook_path"]
    )
    
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)  
    curated_dir = Path(curated_dir)
    
    logbook = pd.read_csv(logbook_path)
    processed_ids = []
    all_dfs = []

    for i, row in enumerate(logbook.itertuples(), start=1):
        if not getattr(row, "valid", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping invalid file ID {row.id}")
            continue
        if getattr(row, "processed", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping already processed file ID {row.id}")
            continue

        input_file_path = curated_dir / f"{row.id}.csv"
        output_file_path = processed_dir / f"{row.id}.csv"  # Save in processed_dir

        try:
            df = pd.read_csv(input_file_path)
            # Feature engineering
            df = feature_engineering.additional_features(df)
            df.to_csv(output_file_path, index=False) 

            processed_ids.append(row.id)
            all_dfs.append(df)
            logbook.at[row.Index, "processed"] = True

            print(f"[{i}/{len(logbook)}] ✅ Processed file ID {row.id} -> saved to {output_file_path}")

        except Exception as e:
            print(f"[{i}/{len(logbook)}] ⚠️ Failed to process file ID {row.id}: {e}")
              
    # Save updated logbook
    if processed_ids:
        logbook.to_csv(logbook_path, index=False)
        print(f"📓 Logbook updated: {len(processed_ids)} files marked as processed.")
    else:
        print("ℹ️ No files were successfully processed.")   

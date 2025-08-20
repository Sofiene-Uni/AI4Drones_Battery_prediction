import pandas as pd
from src.utils.config import  load_config
from src.utils.ulog_io import  extract_from_ulog
from src.data_preprocessing import  data_manip
from src.data_preprocessing import feature_engineering
from src.utils.logbook import update_logbook_status  




def run():
    """Run the data curation step on an existing logbook DataFrame and update status."""
    
    cfg = load_config()
    processed_dir = cfg["paths"]["processed_dir"]
    logbook_path = cfg["paths"]["logbook_path"]
    logbook = pd.read_csv(logbook_path)
    
    datasets_to_extract = cfg.get("datasets", {})
    
    processed_ids = []

    for i, row in enumerate(logbook.itertuples(), start=1):
        if row.processed:
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping already processed file ID {row.id}")
            continue

        try:
            file_path = row.path
            file_id = row.id
    
            datasets = extract_from_ulog(file_path, datasets_to_extract) 
            fused_df = data_manip.fuse_datasets(datasets)
            enhanced_df = feature_engineering.create_features(fused_df)
            
            output_path = f"{processed_dir}/{file_id}.csv"
            enhanced_df.to_csv(output_path, index=False)
    
            processed_ids.append(file_id)
            print(f"[{i}/{len(logbook)}] ✅ Processed file ID {file_id}")

        except Exception as e:
            print(f"[{i}/{len(logbook)}] ⚠️ Skipping file ID {row.id}: {e}")

    # Update status in logbook
    if processed_ids:
        logbook.loc[logbook["id"].isin(processed_ids), "processed"] = True
        logbook.to_csv(logbook_path, index=False)
        print(f"📓 Logbook updated: {len(processed_ids)} files marked as processed.")
    else:
        print("ℹ️ No files were successfully processed.")




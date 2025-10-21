import pandas as pd
from pathlib import Path
from src.utils.config import get_value
from src.utils.ulog_io import extract_from_ulog
from src.data_preprocessing import feature_engineering
from src.data_preprocessing import data_manip
from src.data_curation.filters import (
    filter_empty_file,
    filter_voltage,
    filter_distance,
    filter_current
)

FILTER_MAP = {
    "filter_empty_files": filter_empty_file,
    "filter_voltage": filter_voltage,
    "filter_distance": filter_distance,
    "filter_current": filter_current,
}

def run():
    """Run data curation: extract, fuse, filter, and flag files as curated/valid."""
    
    # --- Paths and datasets ---
    curated_dir, logbook_path, datasets_to_extract = get_value(
        ["paths.curated_dir", "paths.logbook_path", "datasets"]
    )
    curated_dir = Path(curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)  

    # --- Curation config and enabled filters ---
    curation_cfg = get_value("curation", default={}) or {}
    enabled_filters = {
        FILTER_MAP[name]: name
        for name, enabled in curation_cfg.items()
        if enabled and name in FILTER_MAP
    }
    print("Enabled filters:", list(enabled_filters.values()))

    # --- Load logbook ---
    logbook = pd.read_csv(logbook_path)

    # --- Loop through files ---
    for i, row in enumerate(logbook.itertuples(), start=1):
        if getattr(row, "curated", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping already curated file ID {row.id}")
            continue

        file_path = Path(row.path)

        # --- Skip missing files ---
        if not file_path.exists():
            print(f"[{i}/{len(logbook)}] ⚠️ File not found: {file_path}")
            logbook.at[row.Index, "curated"] = True
            logbook.at[row.Index, "valid"] = False
            continue

        try:
            # --- Extract, fuse, feature engineering ---
            datasets = extract_from_ulog(str(file_path), datasets_to_extract)  # Convert Path -> str
            fused_df = data_manip.fuse_datasets(datasets)
            fused_df = feature_engineering.basic_features(fused_df)

            # --- Apply enabled filters ---
            is_valid = True
            failed_filter_name = None
            for filter_fn, name in enabled_filters.items():
                if not filter_fn(fused_df):
                    is_valid = False
                    failed_filter_name = name
                    break

            # --- Save CSV only if valid ---
            if is_valid:
                output_path = curated_dir / f"{row.id}.csv"
                fused_df.to_csv(output_path, index=False)

            # --- Delete original if invalid ---
            if not is_valid:
                file_path.unlink()
                print(f"🗑️ Deleted invalid file: {file_path.name}")

            # --- Update logbook ---
            logbook.at[row.Index, "curated"] = True
            logbook.at[row.Index, "valid"] = is_valid
            status = "✅ valid" if is_valid else f"❌ failed ({failed_filter_name})"
            print(f"[{i}/{len(logbook)}] {status} File ID {row.id}")

        except Exception as e:
            print(f"[{i}/{len(logbook)}] ⚠️ Error processing file ID {row.id}: {e}")
            logbook.at[row.Index, "curated"] = True
            logbook.at[row.Index, "valid"] = False
            # Do NOT delete original here; only delete if processed and invalid

    # --- Save updated logbook ---
    logbook.to_csv(logbook_path, index=False)
    print(f"📓 Logbook updated: {len(logbook)} entries curated & validated")

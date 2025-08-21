import pandas as pd
from src.utils.config import get_value
from src.data_curation.filters import (
    filter_empty_file,
    filter_voltage,
    filter_distance,
    filter_current
)

# Map of filter names to functions
FILTER_MAP = {
    "filter_empty_files": filter_empty_file,
    "filter_voltage": filter_voltage,
    "filter_distance": filter_distance,
    "filter_current": filter_current,
}


def run():
    """Run data curation: open processed files, apply filters enabled in config, flag curated/valid in logbook."""

    processed_dir, logbook_path = get_value(["paths.processed_dir", "paths.logbook_path"])

    # Determine which filters are enabled in config
    curation_cfg = get_value("curation", default={})
    enabled_filters = {
        FILTER_MAP[name]: name
        for name, enabled in curation_cfg.items()
        if enabled and name in FILTER_MAP
    }

    logbook = pd.read_csv(logbook_path)

    # Add curated and valid columns if they don't exist
    logbook["curated"] = logbook.get("curated", False)
    logbook["valid"] = logbook.get("valid", False)

    for i, row in enumerate(logbook.itertuples(), start=1):
        if not getattr(row, "processed", False):
            print(f"[{i}/{len(logbook)}] ⏭️ Skipping unprocessed file ID {row.id}")
            continue

        try:
            file_path = f"{processed_dir}/{row.id}.csv"
            df = pd.read_csv(file_path)

            # File was processed, so it's curated
            logbook.at[row.Index, "curated"] = True

            # Check filters
            is_valid = True
            failed_filter_name = None
            for filter_fn, name in enabled_filters.items():
                if not filter_fn(df):
                    is_valid = False
                    failed_filter_name = name
                    break

            logbook.at[row.Index, "valid"] = is_valid

            if is_valid:
                print(f"[{i}/{len(logbook)}] ✅ File ID {row.id} is valid")
            else:
                print(f"[{i}/{len(logbook)}] ❌ File ID {row.id} flagged as invalid (failed {failed_filter_name})")

        except Exception as e:
            print(f"[{i}/{len(logbook)}] ⚠️ Error with file ID {row.id}: {e}")
            logbook.at[row.Index, "curated"] = True
            logbook.at[row.Index, "valid"] = False

    # Save updated logbook without dropping any rows
    logbook.to_csv(logbook_path, index=False)
    print(f"📓 Logbook updated: {len(logbook)} entries, curated and validity flags applied")

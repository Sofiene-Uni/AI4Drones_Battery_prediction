import pandas as pd
from pathlib import Path
import os
import uuid
from datetime import datetime
from src.utils.config import  load_config


def create_logbook(logbook_path: Path) -> pd.DataFrame:
    """Create a new empty logbook CSV with updated curation and validity columns."""
    
    log_columns = [
        "filename", 
        "path",
        "size_bytes", 
        "created_at",
        "id",
        "curated",    
        "valid",     
        "processed",  
        "use_train",
        "use_test",
        "use_validate"
    ]
    
    logbook = pd.DataFrame(columns=log_columns)
    logbook.to_csv(logbook_path, index=False)
    
    print(f"📓 Created new logbook at {logbook_path} with columns: {log_columns}")
    return logbook


def load_logbook(logbook_path: Path) -> pd.DataFrame:
    """Load an existing logbook CSV."""
    cfg = load_config()
    logbook_path = Path(cfg["paths"]["logbook_path"])
    logbook = pd.read_csv(logbook_path)
    #print(f"✅ Loaded existing logbook ({len(logbook)} entries).")
    return logbook


def get_logbook() -> pd.DataFrame:
    """
    Load the logbook if it exists, otherwise create an empty one.
    """
    cfg = load_config()
    logbook_path = Path(cfg["paths"]["logbook_path"])
    
    if logbook_path.exists():
        return load_logbook(logbook_path)
    else:
        return create_logbook(logbook_path)
    
    
def save_logbook(logbook: pd.DataFrame) -> None:
    """
    Save the logbook DataFrame to CSV.
    
    Parameters:
        logbook (pd.DataFrame): The logbook to save.
        logbook_path (Path): Destination CSV path.
    """
    
    cfg = load_config()
    logbook_path = Path(cfg["paths"]["logbook_path"])
    logbook_path.parent.mkdir(parents=True, exist_ok=True)
    logbook.to_csv(logbook_path, index=False)
    #print(f"💾 Logbook saved at {logbook_path} ({len(logbook)} entries).")



def scan_files(logbook: pd.DataFrame) -> pd.DataFrame:
    """
    Scan raw_dir for .ulg files and update the logbook with new entries.
    
    Parameters:
        raw_dir (str): Directory containing raw flight logs.
        logbook (pd.DataFrame): Current logbook.
        logbook_path (str): Path to the logbook CSV.
    
    Returns:
        pd.DataFrame: Updated logbook
    """
    
    
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    
    logbook = get_logbook()
    
    log_columns = [
        "filename", "path", "size_bytes", "created_at", "id",
        "processed", "curated", "use_train", "use_test", "use_validate"
    ]

    new_entries = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".ulg"):
            fpath = os.path.join(raw_dir, fname)
            if fpath not in logbook["path"].values:  # check if already in logbook
                stats = os.stat(fpath)
                new_entries.append({
                    "filename": fname,
                    "path": fpath,
                    "size_bytes": stats.st_size,
                    "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    "id": str(uuid.uuid4()),
                    "processed": False,
                    "curated": False,
                    "use_train": False,
                    "use_test": False,
                    "use_validate": False
                })
                print(f"🆕 Added new file to logbook: {fname}")

    if new_entries:
        logbook = pd.concat([logbook, pd.DataFrame(new_entries, columns=log_columns)], ignore_index=True)


    save_logbook(logbook)

    return logbook



def update_logbook_status(step: str, ids: list[str]) -> pd.DataFrame:
    """
    Update the logbook step status (e.g., 'processed', 'curated') for given IDs.

    Parameters:
        logbook (pd.DataFrame): Current logbook dataframe.
        logbook_path (str): Path to the logbook CSV file.
        step (str): Column name to update (e.g., 'processed', 'curated').
        ids (list[str]): List of logbook entry IDs to update.

    Returns:
        pd.DataFrame: Updated logbook.
    """
    logbook=get_logbook()
    
    # Safety check
    if step not in logbook.columns:
        raise ValueError(f"Step '{step}' not found in logbook columns: {list(logbook.columns)}")
    
    # Update rows where ID matches
    before = logbook[step].sum()
    logbook.loc[logbook["id"].isin(ids), step] = True
    after = logbook[step].sum()

    save_logbook(logbook)
    print(f"🔄 Updated step '{step}' for {len(ids)} entries. ({before} → {after} True)")
    return logbook

import os
import pandas as pd
from datetime import datetime
from pyulog import ULog



def scan_ulog_folder(folder: str, logbook_path: str = None) -> pd.DataFrame:
    """
    Scan a folder for .ulg files and create a logbook DataFrame.

    Args:
        folder (str): Path to the folder to scan.
        logbook_path (str, optional): Where to save the logbook (CSV).
                                      If None, does not save.

    Returns:
        pd.DataFrame: Logbook with columns [filename, path, size_bytes, created_at].
    """
    files = []
    for fname in os.listdir(folder):
        if fname.endswith(".ulg"):
            fpath = os.path.join(folder, fname)
            stats = os.stat(fpath)
            files.append({
                "filename": fname,
                "path": fpath,
                "size_bytes": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat()
            })

    logbook = pd.DataFrame(files)

    if logbook_path:
        os.makedirs(os.path.dirname(logbook_path), exist_ok=True)
        logbook.to_csv(logbook_path, index=False)

    return logbook


def load_ulog(file_path: str):
    """
    Load a ULog (.ulg) file.

    Args:
        file_path (str): Path to the .ulg file.

    Returns:
        ULog: pyulog ULog object containing datasets.
    """
    if ULog is None:
        raise ImportError("pyulog is not installed. Install it with `pip install pyulog`.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    try:
        ulog = ULog(file_path)
        return ulog
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")
        
        

def extract_from_ulog(ulg_file_path, datasets_to_extract):
    extracted_data = {}
    
    try:
        ulog = ULog(ulg_file_path)  
        print(f"ULog file loaded successfully: {ulg_file_path}\n")
    
        for dataset_name, fields in datasets_to_extract.items():
            try:
                dataset = ulog.get_dataset(dataset_name).data
                extracted_data[dataset_name] = {}
                for field in fields:
                    if field in dataset:
                        extracted_data[dataset_name][field] = dataset[field]
                    else:
                        print(f"Warning: field '{field}' not found in dataset '{dataset_name}'")
            except Exception as e:
                print(f"Warning: dataset '{dataset_name}' not found in {ulg_file_path}: {e}")

    except Exception as e:
        print(f"Failed to load ULog file: {e}")
        return None

    return extracted_data







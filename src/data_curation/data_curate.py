import pandas as pd
from src.utils.config import  load_config
from src.utils.ulog_io import  extract_from_ulog
from src.data_curation import  filter_empty_file,filter_zero_voltage,filter_short_distance,filter_no_current 



FILTER_MAP = {
    "filter_empty_files": filter_empty_file,
    "filter_zero_voltage": filter_zero_voltage,
    "filter_short_distance": filter_short_distance,
    "filter_no_current": filter_no_current,
}

def run():
    """Run the data curation step on an existing logbook DataFrame."""
    
    cfg = load_config()
    interim_dir = cfg["paths"]["interim_dir"]
    logbook_path = cfg["paths"]["logbook_path"]
    logbook = pd.read_csv(logbook_path)
    
    
    
    
  

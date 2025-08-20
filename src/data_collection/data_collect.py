from src.utils.config import  load_config
from src.utils.logbook import get_logbook,scan_files
from src.data_collection import data_download



def run():
    cfg = load_config()

    if cfg["pipeline"].get("run_data_collection", False):
        if cfg["download"].get("download_latest", False):
            data_download.run(cfg)

   
    log_book= get_logbook()
    scan_files(log_book)


   
   
from src.utils.config import  get_value
from src.utils.logbook import get_logbook,scan_files
from src.data_collection import data_download



def run():
    # No need to load the config manually if using get_value
    if get_value("pipeline.run_data_collection", False):
        if get_value("download.download_latest", False):
            data_download.run()  # or pass cfg if needed

    log_book = get_logbook()
    scan_files(log_book)


   
   
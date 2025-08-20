from src.utils.config import  load_config
from src.data_collection import  data_collect
from src.data_preprocessing import data_preprocess
from src.data_curation import data_curate
from src.data_preparation import data_prepare
from src.train_test_models import  train, evaluate, visualize


if __name__ == "__main__":
    
    
    config = load_config()
    

    steps = [
         ("run_data_collection", data_collect.run),
         ("run_data_preprocessing", data_preprocess.run),
         ("run_data_curation", data_curate.run),
         ("run_data_prepare", data_prepare.run),
         ("run_train", train.run),
         ("run_evaluate", evaluate.run),
         ("run_visualize", visualize.run)
    ]

    for flag, func in steps:
        if config["pipeline"].get(flag, False):
            print(f"\n🚀 Running step: {flag}")
            func()
        else:
            print(f"⏩ Skipping step: {flag}")
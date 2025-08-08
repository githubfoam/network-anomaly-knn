import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "data"
DATASET_PATHS = [
    "UNSW-NB15_features.csv",
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

def download_unsw_nb15():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    api = KaggleApi()
    api.authenticate()

    # Replace with your Kaggle dataset path
    dataset = "your-kaggle-username/unsw-nb15-private"
    api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)

def load_unsw_nb15():
    if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in DATASET_PATHS):
        download_unsw_nb15()

    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in DATASET_PATHS]
    return pd.concat(dfs, ignore_index=True)

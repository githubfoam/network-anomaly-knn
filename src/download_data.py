import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_unsw_nb15(dataset_owner="your-kaggle-username/dataset-name", data_dir="data"):
    """
    Downloads UNSW-NB15 dataset from Kaggle.
    """
    os.makedirs(data_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset {dataset_owner} to {data_dir}...")
    api.dataset_download_files(dataset_owner, path=data_dir, unzip=True)
    print("Download complete.")

def merge_unsw_parts(data_dir="data"):
    """
    Merges UNSW-NB15 parts and features CSV into one DataFrame.
    """
    features = pd.read_csv(os.path.join(data_dir, "NUSW-NB15_features.csv"))
    df_parts = []
    for i in range(1, 5):
        file_path = os.path.join(data_dir, f"UNSW-NB15_{i}.csv")
        df_parts.append(pd.read_csv(file_path, header=None))
    data_df = pd.concat(df_parts, ignore_index=True)
    data_df.columns = features["Name"].tolist()
    return data_df

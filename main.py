import os
import pandas as pd
from src.download_data import download_unsw_nb15, merge_unsw_parts
from src.preprocess_data import preprocess
from src.train_model import train_knn_model
from src.predict import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    data_dir = "data"

    if not os.path.exists(os.path.join(data_dir, "UNSW-NB15_1.csv")):
        download_unsw_nb15("your-kaggle-username/nusw-nb15", data_dir)

    df = merge_unsw_parts(data_dir)

    # Split into train/test (70/30)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])

    X_train, y_train, scaler = preprocess(train_df, fit_scaler=True)
    X_test, y_test, _ = preprocess(test_df, fit_scaler=False, scaler=scaler)

    model = train_knn_model(X_train, y_train, n_neighbors=5)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

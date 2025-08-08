# Network Traffic Anomaly Detection using K-NN

This project implements a K-Nearest Neighbors (K-NN) based model to detect anomalies in network traffic.

K-NN in cybersecurity — Correct. K-Nearest Neighbors is indeed used for intrusion detection, malware classification, fraud detection, and network traffic analysis. It’s simple, non-parametric, and instance-based, making it a good teaching and baseline model.

Public datasets — All three mentioned (NSL-KDD, CIC-IDS2017/2018, UNSW-NB15) are standard and widely used in research. NSL-KDD is indeed the improved KDD Cup 1999 dataset.

Preprocessing details — One-hot encoding for categorical features and standard scaling for numerical features is the correct approach for K-NN, since distance-based algorithms are sensitive to scale.

Functional and consistent with scikit-learn best practices. Using n_jobs=-1 in KNN is a good optimization.

Feature alignment for train/test — In main.py, when preprocessing the test dataset, you should ensure the same columns as the training set (to avoid missing dummy columns). A safe approach is to fit the encoders/scalers only once on the training set and apply them to the test set without creating new encodings.

Label encoding note — For attack labels, ensure the mapping is consistent between train and test (currently implied but not explicitly shown).

Performance consideration — K-NN can be slow for large datasets since it stores all training samples. A note on dimensionality reduction (PCA) or approximate nearest neighbor search could be useful.


Reproducibility — Setting a random_state in train_test_split ensures repeatable results.

Example snippet for consistent preprocessing:


            # Preprocess training set
            processed_train_df, label_encoder, scaler = preprocess(train_df)

            # Apply SAME encoders/scalers to test set
            processed_test_df = test_df.copy()
            processed_test_df = pd.get_dummies(processed_test_df, columns=['protocol_type', 'service', 'flag'])
            processed_test_df = processed_test_df.reindex(columns=processed_train_df.columns, fill_value=0)
            processed_test_df['label'] = label_encoder.transform(processed_test_df['label'])
            processed_test_df[numerical_cols] = scaler.transform(processed_test_df[numerical_cols])


Key Dataset Info

    Source: Created by the Cyber Range Lab at the Australian Centre for Cyber Security (ACCS)

    Target column: label (0 = Normal, 1 = Attack)

    Categorical columns (need one-hot encoding):

        proto (protocol)

        service (network service)

        state (connection state)

    Numerical features: All others except id, attack_cat, and label.

## Project Structure

- `data/`: Stores raw and processed datasets.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis (EDA) and model experimentation.
- `src/`: Houses the core Python code, including the K-NN model implementation.
- `tests/`: Contains unit tests for the source code.


# K-NN Based Intrusion Detection with UNSW-NB15

## Overview
This repository demonstrates a basic intrusion detection system using the K-Nearest Neighbors (K-NN) algorithm on the UNSW-NB15 dataset.  
The system classifies network flows as **normal** or **attack** based on extracted features.

## Dataset
The UNSW-NB15 dataset was created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS).  
It contains modern attack scenarios and realistic traffic patterns.

## Methodology
1. **Data Preprocessing**: Drop non-predictive columns, one-hot encode categorical features, scale numerical features.
2. **Model Training**: Train a K-NN classifier, tuning `k` via cross-validation.
3. **Evaluation**: Evaluate using accuracy, confusion matrix, precision, recall, and F1-score.

## Setup
```bash
git clone https://github.com/your-username/unsw-nb15-intrusion-detection-knn.git
cd unsw-nb15-intrusion-detection-knn
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
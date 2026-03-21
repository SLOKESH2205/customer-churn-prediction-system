import os
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import build_features
from src.components.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logging.info("RUNNING FULL ML PIPELINE")

    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion(
        file_path="data/raw/online_retail_II.xlsx"
    )

    raw_df = pd.read_csv(ingestion.raw_data_path)
    customer_df, kmeans_model = build_features(raw_df)

    X = customer_df.drop(columns=["retention_status"])
    y = customer_df["retention_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["retention_status"] = y_train.values

    test_df = X_test.copy()
    test_df["retention_status"] = y_test.values

    trainer = ModelTrainer()

    best_model, best_score, report, best_threshold = trainer.initiate_model_training(
        train_df, test_df
    )

    os.makedirs("artifacts", exist_ok=True)

    # Save kmeans model (encoded_features.pkl is saved by model_trainer.py)
    joblib.dump(kmeans_model, "artifacts/kmeans.pkl")

    print("Training Complete")

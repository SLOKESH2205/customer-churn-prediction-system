import pandas as pd
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

import logging
from src.exception import CustomException
from src.components.feature_engineering import build_features


class DataTransformation:

    def __init__(self):
        self.model_features = [
            "recency_log",
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "unique_items_purchased"
        ]
        self.target_column = "target"
        self.scaler_path = os.path.join("artifacts", "scaler.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 🔥 APPLY FEATURE ENGINEERING
            train_df, _ = build_features(train_df)
            test_df, _ = build_features(test_df)

            # ================= VALIDATION ================= #
            missing_cols = [col for col in self.model_features if col not in train_df.columns]
            if missing_cols:
                raise Exception(f"Missing columns: {missing_cols}")

            if self.target_column not in train_df.columns:
                raise Exception(f"Target column '{self.target_column}' not found")

            # ================= SPLIT ================= #
            X_train = train_df[self.model_features].fillna(0)
            X_test = test_df[self.model_features].fillna(0)

            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            # ================= SCALING ================= #
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ================= SAVE ================= #
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)

            logging.info("Data transformation completed")

            return (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                self.scaler_path
            )

        except Exception as e:
            logging.error(f"Error in transformation: {e}")
            raise CustomException(e, sys)

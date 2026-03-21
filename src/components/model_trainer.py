import os
import sys
import joblib
import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from src.exception import CustomException
from src.preprocessing import prepare_model_matrix


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.scaler_path = os.path.join("artifacts", "model_scaler.pkl")

    def evaluate_model(self, y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)
        }

    def initiate_model_training(self, train_df, test_df):
        try:
            logging.info("Starting model training")

            # ================= FEATURES ================= #
            feature_cols = [
                "frequency_log",
                "monetary_log",
                "tenure",
                "avg_order_value",
                "unique_items_purchased",
                "purchase_rate",
                "monetary_per_day",
                "final_kmeans_cluster"
            ]

            target_col = "retention_status"

            # ================= TRAIN DATA ================= #
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]

            # ================= TEST DATA ================= #
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            X_train = prepare_model_matrix(X_train)
            X_test = prepare_model_matrix(X_test, training_columns=X_train.columns.tolist())
            encoded_feature_names = X_train.columns.tolist()
            
            # ================= SCALING ================= #
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)

            # ================= CLASS IMBALANCE ================= #
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            # ================= MODELS ================= #
            models = {
                "Logistic Regression": (
                    LogisticRegression(solver='liblinear', class_weight='balanced'),
                    {"C": [0.01, 0.1, 1, 10]}
                ),
                "Random Forest": (
                    RandomForestClassifier(class_weight='balanced', random_state=42),
                    {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_split": [2, 5]}
                ),
                "XGBoost": (
                    XGBClassifier(
                        eval_metric='logloss',
                        random_state=42,
                        scale_pos_weight=scale_pos_weight
                    ),
                    {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
                )
            }

            best_model = None
            best_score = 0
            best_model_name = None
            best_threshold_global = None
            report = {}

            # ================= TRAIN LOOP ================= #
            for name, (model, params) in models.items():

                print(f"\n===== TRAINING {name} =====")

                grid = GridSearchCV(
                    model,
                    params,
                    cv=3,
                    scoring='f1',
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_

                # ================= PREDICTIONS ================= #

                # 🔥 Threshold tuning (you can tweak later)
                import numpy as np

                y_prob = best_estimator.predict_proba(X_test)[:, 1]

                # 🔥 FIND BEST THRESHOLD
                best_threshold = 0.5
                best_f1_local = 0

                for t in np.arange(0.2, 0.7, 0.05):
                    y_temp = (y_prob > t).astype(int)
                    f1_temp = f1_score(y_test, y_temp)

                    if f1_temp > best_f1_local:
                        best_f1_local = f1_temp
                        best_threshold = t

                # 🔥 FINAL PREDICTION USING BEST THRESHOLD
                y_pred = (y_prob > best_threshold).astype(int)

                print(f"Best Threshold for {name}: {best_threshold}")

                # ================= METRICS ================= #
                scores = self.evaluate_model(y_test, y_pred, y_prob)
                report[name] = scores

                print("\nMetrics:")
                print(scores)

                # 🔥 NEW (IMPORTANT)
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                print("Confusion Matrix:")
                print(confusion_matrix(y_test, y_pred))

                if scores["f1_score"] > best_score:
                    best_score = scores["f1_score"]
                    best_model = best_estimator
                    best_model_name = name
                    best_threshold_global = best_threshold

            # ================= SAVE BEST MODEL ================= #
            os.makedirs("artifacts", exist_ok=True)
            
            # Save model with threshold
            joblib.dump(
                {
                    "model": best_model,
                    "threshold": best_threshold_global,
                    "model_name": best_model_name,
                },
                self.model_path
            )
            
            # Save the encoded feature column names (IMPORTANT!)
            joblib.dump(encoded_feature_names, os.path.join("artifacts", "encoded_features.pkl"))

            print("\n==============================")
            print(f"BEST MODEL: {best_model_name}")
            print(f"BEST F1 SCORE: {best_score}")
            print(f"BEST THRESHOLD: {best_threshold_global}")
            print("==============================")

            return best_model_name, best_score, report, best_threshold_global

        except Exception as e:
            raise CustomException(e, sys)

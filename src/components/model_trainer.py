import os
import sys
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
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
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None and len(set(y_true)) > 1 else None,
        }

    def initiate_model_training(self, train_df, test_df):
        try:
            logging.info("Starting model training")
            os.makedirs("artifacts", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)

            feature_cols = [
                "frequency_log",
                "monetary_log",
                "tenure",
                "avg_order_value",
                "unique_items_purchased",
                "purchase_rate",
                "monetary_per_day",
                "final_kmeans_cluster",
            ]
            target_col = "retention_status"

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            print("\nTrain Distribution:")
            print(y_train.value_counts())

            print("\nTest Distribution:")
            print(y_test.value_counts())

            X_train = prepare_model_matrix(X_train)
            X_test = prepare_model_matrix(X_test, training_columns=X_train.columns.tolist())
            encoded_feature_names = X_train.columns.tolist()

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            joblib.dump(scaler, self.scaler_path)

            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            models = {
                "Logistic Regression": (
                    LogisticRegression(solver="liblinear", class_weight="balanced"),
                    {"C": [0.01, 0.1, 1, 10]},
                ),
                "Random Forest": (
                    RandomForestClassifier(class_weight="balanced", random_state=42),
                    {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_split": [2, 5]},
                ),
                "XGBoost": (
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42,
                        scale_pos_weight=scale_pos_weight,
                    ),
                    {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
                ),
            }

            best_model = None
            best_score = 0
            best_model_name = None
            best_threshold_global = None
            report = {}
            results = []

            import numpy as np

            for name, (model, params) in models.items():
                print(f"\n===== TRAINING {name} =====")

                grid = GridSearchCV(
                    model,
                    params,
                    cv=3,
                    scoring="f1",
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_

                y_pred = best_estimator.predict(X_test)
                y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, "predict_proba") else None

                best_threshold = 0.5
                best_f1_local = 0

                if y_prob is not None:
                    for threshold in np.arange(0.2, 0.7, 0.05):
                        threshold_pred = (y_prob > threshold).astype(int)
                        threshold_f1 = f1_score(y_test, threshold_pred, zero_division=0)
                        if threshold_f1 > best_f1_local:
                            best_f1_local = threshold_f1
                            best_threshold = threshold
                    y_pred = (y_prob > best_threshold).astype(int)

                print(f"Best Threshold for {name}: {best_threshold}")

                metrics = self.evaluate_model(y_test, y_pred, y_prob)
                report[name] = metrics
                results.append({"Model": name, **metrics})

                print("\nMetrics:")
                print(metrics)

                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, zero_division=0))

                cm = confusion_matrix(y_test, y_pred)
                print("Confusion Matrix:")
                print(cm)

                plt.figure(figsize=(5, 4))
                plt.imshow(cm, cmap="Blues")
                plt.title(f"{name} Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.colorbar()
                plt.xticks([0, 1], [0, 1])
                plt.yticks([0, 1], [0, 1])
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
                plt.tight_layout()
                plt.savefig(os.path.join("outputs", f"{name.lower().replace(' ', '_')}_cm.png"))
                plt.close()

                if metrics["f1_score"] > best_score:
                    best_score = metrics["f1_score"]
                    best_model = best_estimator
                    best_model_name = name
                    best_threshold_global = best_threshold

            joblib.dump(
                {
                    "model": best_model,
                    "threshold": best_threshold_global,
                    "model_name": best_model_name,
                },
                self.model_path,
            )
            joblib.dump(encoded_feature_names, os.path.join("artifacts", "encoded_features.pkl"))
            pd.DataFrame(results).to_csv(os.path.join("outputs", "model_metrics.csv"), index=False)

            print("\n==============================")
            print(f"BEST MODEL: {best_model_name}")
            print(f"BEST F1 SCORE: {best_score}")
            print(f"BEST THRESHOLD: {best_threshold_global}")
            print("==============================")

            return best_model_name, best_score, report, best_threshold_global

        except Exception as e:
            raise CustomException(e, sys)

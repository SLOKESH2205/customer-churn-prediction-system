import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import build_customer_features, prepare_model_matrix


class ChurnModelService:
    """Shared inference and explainability layer for the app and pipelines."""

    def __init__(
        self,
        model_path: str = os.path.join("artifacts", "model.pkl"),
        scaler_path: str = os.path.join("artifacts", "model_scaler.pkl"),
        cluster_model_path: str = os.path.join("artifacts", "kmeans.pkl"),
        encoded_features_path: str = os.path.join("artifacts", "encoded_features.pkl"),
        legacy_features_path: str = os.path.join("artifacts", "features.pkl"),
    ):
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict):
            self.model = loaded_obj["model"]
            self.threshold = loaded_obj["threshold"]
        else:
            self.model = loaded_obj
            self.threshold = 0.5

        self.scaler = joblib.load(scaler_path)
        self.kmeans = joblib.load(cluster_model_path)

        try:
            self.train_columns = joblib.load(encoded_features_path)
        except FileNotFoundError:
            self.train_columns = joblib.load(legacy_features_path)

        self.last_processed_df = None
        self.features = []

    def predict_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        customer_df, _ = build_customer_features(df, self.kmeans)
        customer_ids = customer_df["customer_id"].values if "customer_id" in customer_df.columns else None

        model_df = prepare_model_matrix(customer_df, self.train_columns)
        self.last_processed_df = model_df.copy()
        self.features = model_df.columns.tolist()

        scaled = self.scaler.transform(model_df)
        churn_probability = self.model.predict_proba(scaled)[:, 1]
        churn_label = (churn_probability >= self.threshold).astype(int)

        result_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "Churn_Label": churn_label,
                "Churn Probability": churn_probability,
                "cluster": customer_df["final_kmeans_cluster"].values,
            }
        )

        feature_cols = [
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "unique_items_purchased",
            "purchase_rate",
            "monetary_per_day",
        ]
        result_df = result_df.join(customer_df[feature_cols].reset_index(drop=True))
        return result_df

    def get_feature_importance_table(self, top_n: int = 10) -> pd.DataFrame:
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["Feature", "Importance", "Importance (%)"])

        raw_importance = pd.DataFrame(
            {
                "Feature": self.features,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)
        raw_importance["Importance (%)"] = raw_importance["Importance"] * 100
        return raw_importance.head(top_n).reset_index(drop=True)

    def explain_feature_importance(self, top_n: int = 5) -> Dict[str, object]:
        importance_df = self.get_feature_importance_table(top_n=top_n)
        if importance_df.empty:
            return {
                "table": importance_df,
                "summary": "Feature importance is only available for tree-based models with built-in importance scores.",
                "driver_insights": [],
            }

        driver_map = {
            "frequency_log": "Lower repeat purchase frequency is one of the strongest churn signals.",
            "monetary_log": "Customer value matters: changes in spend level are strongly tied to churn risk.",
            "tenure": "Customer tenure is a major retention signal, with newer relationships behaving differently from established ones.",
            "avg_order_value": "Basket size helps separate high-commitment customers from light buyers.",
            "unique_items_purchased": "Narrow product breadth can indicate shallow engagement and weaker attachment.",
            "purchase_rate": "Purchase cadence is a behavioral leading indicator of churn.",
            "monetary_per_day": "Revenue generated per day captures whether value creation is sustained over time.",
            "cluster_1": "Segment membership itself matters, suggesting churn is concentrated in a specific persona.",
            "cluster_2": "Segment membership itself matters, suggesting churn is concentrated in a specific persona.",
        }

        driver_insights = [
            driver_map.get(row["Feature"], f"{row['Feature']} is materially influencing churn predictions.")
            for _, row in importance_df.iterrows()
        ]

        top_features = ", ".join(importance_df["Feature"].head(3).tolist())
        summary = f"Top churn drivers are {top_features}, indicating that retention risk is primarily explained by customer behavior, value, and segment membership."
        return {
            "table": importance_df,
            "summary": summary,
            "driver_insights": driver_insights,
        }

    def explain_customer_churn_pattern(self, result_df: pd.DataFrame) -> List[str]:
        if result_df.empty:
            return []

        high_risk = result_df[result_df["Churn_Label"] == 1]
        low_risk = result_df[result_df["Churn_Label"] == 0]
        if high_risk.empty or low_risk.empty:
            return []

        insights = []
        comparisons = {
            "purchase_rate": "High-risk customers are buying far less often than retained customers, making engagement decline the clearest churn signal.",
            "tenure": "High-risk customers are earlier in their lifecycle, which suggests onboarding and habit formation are not yet strong enough.",
            "monetary_log": "Churn risk is concentrated among customers whose value profile differs materially from the retained base.",
            "avg_order_value": "Basket size differs between churn and retained groups, indicating spend depth matters to retention.",
        }

        for feature, text in comparisons.items():
            if feature not in result_df.columns:
                continue
            high_mean = high_risk[feature].mean()
            low_mean = low_risk[feature].mean()
            if np.isclose(low_mean, 0):
                continue
            ratio = high_mean / low_mean
            if ratio <= 0.85 or ratio >= 1.15:
                insights.append(text)

        return insights[:4]

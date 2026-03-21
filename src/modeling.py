from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import prepare_model_matrix


@dataclass
class DynamicChurnArtifacts:
    model: RandomForestClassifier
    scaler: StandardScaler
    train_columns: List[str]
    threshold: float
    model_name: str = "Random Forest"


class ChurnModelService:
    """Train-on-upload churn service used by the Streamlit app."""

    def __init__(self, artifacts: DynamicChurnArtifacts):
        self.model = artifacts.model
        self.scaler = artifacts.scaler
        self.train_columns = artifacts.train_columns
        self.threshold = artifacts.threshold
        self.model_name = artifacts.model_name
        self.last_processed_df = None
        self.features = artifacts.train_columns

    @classmethod
    def train_from_customer_df(cls, customer_df: pd.DataFrame) -> "ChurnModelService":
        model_df = prepare_model_matrix(customer_df)
        target = customer_df["retention_status"].astype(int)

        imputer = SimpleImputer(strategy="median")
        model_df = pd.DataFrame(
            imputer.fit_transform(model_df),
            columns=model_df.columns,
            index=model_df.index,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            model_df,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target if target.nunique() > 1 else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        )
        model.fit(X_train_scaled, y_train)

        best_threshold = 0.5
        best_f1 = -1.0
        if y_test.nunique() > 1:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            for threshold in np.arange(0.2, 0.75, 0.05):
                y_pred = (y_prob >= threshold).astype(int)
                score = f1_score(y_test, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_threshold = float(threshold)

        artifacts = DynamicChurnArtifacts(
            model=model,
            scaler=scaler,
            train_columns=model_df.columns.tolist(),
            threshold=best_threshold,
        )
        return cls(artifacts)

    def score_customer_df(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        model_df = prepare_model_matrix(customer_df, training_columns=self.train_columns)
        imputer = SimpleImputer(strategy="median")
        model_df = pd.DataFrame(
            imputer.fit_transform(model_df),
            columns=model_df.columns,
            index=model_df.index,
        )
        self.last_processed_df = model_df.copy()
        self.features = model_df.columns.tolist()

        scaled = self.scaler.transform(model_df)
        churn_probability = self.model.predict_proba(scaled)[:, 1]
        churn_label = (churn_probability >= self.threshold).astype(int)

        result_df = pd.DataFrame(
            {
                "customer_id": customer_df["customer_id"].values,
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
        return result_df.join(customer_df[feature_cols].reset_index(drop=True))

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

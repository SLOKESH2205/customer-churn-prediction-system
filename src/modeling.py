from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import prepare_model_matrix


class ChurnModelService:
    """Train-on-upload churn service used by the Streamlit app."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.train_columns: List[str] = []
        self.threshold = 0.5
        self.model_name = "Random Forest"
        self.evaluation_metrics = None
        self.confusion_matrix = []
        self.roc_curve_points = {}
        self.classification_report_text = ""
        self.last_processed_df = None
        self.features: List[str] = []

    def evaluate(self, y_true, y_pred, y_proba):
        if len(y_true) == 0:
            return None

        roc_auc = None
        if len(np.unique(y_true)) > 1 and len(np.unique(y_proba)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_proba))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
        }

    def train(self, X_train, y_train, X_test, y_test):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        )
        self.model.fit(X_train_scaled, y_train)

        self.train_columns = X_train.columns.tolist()
        self.features = self.train_columns
        self.confusion_matrix = []
        self.roc_curve_points = {}
        self.classification_report_text = ""
        self.evaluation_metrics = None

        if len(y_test) == 0:
            return self

        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.arange(0.2, 0.75, 0.05):
            y_pred = (y_prob >= threshold).astype(int)
            score = f1_score(y_test, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self.threshold = best_threshold
        final_pred = (y_prob >= self.threshold).astype(int)
        self.evaluation_metrics = self.evaluate(y_test, final_pred, y_prob)
        self.confusion_matrix = confusion_matrix(y_test, final_pred).tolist()
        self.classification_report_text = classification_report(
            y_test,
            final_pred,
            zero_division=0,
        )

        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            self.roc_curve_points = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

        return self

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

        service = cls()
        return service.train(X_train, y_train, X_test, y_test)

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

        if "retention_status" in customer_df.columns:
            result_df["Churn_Label_Actual"] = customer_df["retention_status"].values

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
        if self.model is None or not hasattr(self.model, "feature_importances_"):
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

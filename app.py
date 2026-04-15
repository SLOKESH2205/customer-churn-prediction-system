import os
import logging
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.clustering import evaluate_clustering
from src.components.segment_analytics import analyze_segments
from src.modeling import ChurnModelService
from src.preprocessing import (
    CLUSTER_FEATURE_COLUMNS,
    build_customer_features,
    normalize_transaction_columns,
    prepare_model_matrix,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

DEFAULT_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
DEFAULT_TEST_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
BUNDLED_TRAIN_PATH = os.path.join(SAMPLE_DATA_DIR, "default_train.csv")
BUNDLED_TEST_PATH = os.path.join(SAMPLE_DATA_DIR, "default_test.csv")


def generate_insights(row):
    insights = []

    engagement_score = row.get("engagement_score", row.get("purchase_rate", 0))
    tenure = row.get("tenure", 0)
    usage = row.get("usage", row.get("frequency_log", 0))

    if engagement_score < 0.3:
        insights.append("Low engagement -> High churn risk")
    if tenure < 6:
        insights.append("New customer -> Higher churn probability")
    if usage < 5:
        insights.append("Low usage -> Potential disengagement")

    return insights


def classify_risk(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    if probability >= 0.4:
        return "Medium Risk"
    return "Low Risk"


def compute_dynamic_risk_thresholds(probabilities: pd.Series) -> tuple[float, float]:
    clean_probabilities = pd.Series(probabilities).dropna().astype(float)
    if clean_probabilities.empty:
        return 0.4, 0.7

    q1 = float(clean_probabilities.quantile(0.33))
    q2 = float(clean_probabilities.quantile(0.66))
    if np.isclose(q1, q2):
        median = float(clean_probabilities.median())
        q1 = median
        q2 = median
    return q1, q2


def map_dynamic_risk(probability: float, low_cutoff: float, high_cutoff: float) -> str:
    if probability >= high_cutoff:
        return "High Risk"
    if probability >= low_cutoff:
        return "Medium Risk"
    return "Low Risk"


def apply_prediction_threshold(result_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scored_df = result_df.copy()
    scored_df["Predicted_Label"] = (scored_df["Churn Probability"] >= threshold).astype(int)
    return scored_df


def apply_dynamic_risk_segmentation(result_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored_df = result_df.copy()
    low_cutoff, high_cutoff = compute_dynamic_risk_thresholds(scored_df["Churn Probability"])
    scored_df["Risk Segment"] = scored_df["Churn Probability"].apply(
        lambda prob: map_dynamic_risk(prob, low_cutoff, high_cutoff)
    )
    return scored_df, {"q1": low_cutoff, "q2": high_cutoff}


@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def default_artifacts_available() -> bool:
    return os.path.exists(DEFAULT_TRAIN_PATH) and os.path.exists(DEFAULT_TEST_PATH)


def bundled_sample_available() -> bool:
    return os.path.exists(BUNDLED_TRAIN_PATH) and os.path.exists(BUNDLED_TEST_PATH)


@st.cache_data
def create_demo_raw_datasets():
    demo_train_rows = [
        {"Invoice": 700001, "StockCode": "D1001", "Description": "Starter Kit", "Quantity": 4, "InvoiceDate": "2010-09-05 10:00", "Price": 12.50, "Customer ID": 30001, "Country": "United Kingdom"},
        {"Invoice": 700101, "StockCode": "D1002", "Description": "Premium Mug", "Quantity": 3, "InvoiceDate": "2010-12-20 09:15", "Price": 18.00, "Customer ID": 30001, "Country": "United Kingdom"},
        {"Invoice": 700002, "StockCode": "D1003", "Description": "Desk Organizer", "Quantity": 2, "InvoiceDate": "2010-08-12 11:30", "Price": 22.00, "Customer ID": 30002, "Country": "Germany"},
        {"Invoice": 700102, "StockCode": "D1004", "Description": "Cable Pack", "Quantity": 6, "InvoiceDate": "2010-12-18 15:20", "Price": 7.50, "Customer ID": 30002, "Country": "Germany"},
        {"Invoice": 700003, "StockCode": "D1005", "Description": "Notebook Set", "Quantity": 5, "InvoiceDate": "2010-10-02 13:45", "Price": 9.00, "Customer ID": 30003, "Country": "France"},
        {"Invoice": 700103, "StockCode": "D1006", "Description": "Travel Bottle", "Quantity": 2, "InvoiceDate": "2010-12-22 16:40", "Price": 25.00, "Customer ID": 30003, "Country": "France"},
        {"Invoice": 700004, "StockCode": "D1007", "Description": "Laptop Stand", "Quantity": 1, "InvoiceDate": "2010-07-15 14:10", "Price": 45.00, "Customer ID": 30004, "Country": "Netherlands"},
        {"Invoice": 700104, "StockCode": "D1008", "Description": "Wireless Mouse", "Quantity": 2, "InvoiceDate": "2010-12-28 12:05", "Price": 20.00, "Customer ID": 30004, "Country": "Netherlands"},
        {"Invoice": 700005, "StockCode": "D1009", "Description": "Seasonal Card", "Quantity": 2, "InvoiceDate": "2010-03-10 10:00", "Price": 6.00, "Customer ID": 30101, "Country": "Spain"},
        {"Invoice": 700105, "StockCode": "D1010", "Description": "Mini Basket", "Quantity": 1, "InvoiceDate": "2010-05-12 13:30", "Price": 16.00, "Customer ID": 30101, "Country": "Spain"},
        {"Invoice": 700006, "StockCode": "D1011", "Description": "Storage Tin", "Quantity": 3, "InvoiceDate": "2010-02-22 12:20", "Price": 13.00, "Customer ID": 30102, "Country": "Ireland"},
        {"Invoice": 700106, "StockCode": "D1012", "Description": "Apron", "Quantity": 2, "InvoiceDate": "2010-06-01 15:10", "Price": 19.00, "Customer ID": 30102, "Country": "Ireland"},
    ]
    demo_test_rows = [
        {"Invoice": 800001, "StockCode": "E1001", "Description": "Starter Kit", "Quantity": 3, "InvoiceDate": "2010-10-10 10:10", "Price": 12.00, "Customer ID": 40001, "Country": "United Kingdom"},
        {"Invoice": 800101, "StockCode": "E1002", "Description": "Premium Mug", "Quantity": 2, "InvoiceDate": "2010-12-21 09:20", "Price": 18.50, "Customer ID": 40001, "Country": "United Kingdom"},
        {"Invoice": 800002, "StockCode": "E1003", "Description": "Desk Organizer", "Quantity": 2, "InvoiceDate": "2010-09-02 11:00", "Price": 21.00, "Customer ID": 40002, "Country": "Germany"},
        {"Invoice": 800102, "StockCode": "E1004", "Description": "Cable Pack", "Quantity": 5, "InvoiceDate": "2010-12-27 14:10", "Price": 7.00, "Customer ID": 40002, "Country": "Germany"},
        {"Invoice": 800003, "StockCode": "E1005", "Description": "Notebook Set", "Quantity": 4, "InvoiceDate": "2010-02-14 10:30", "Price": 8.50, "Customer ID": 40101, "Country": "France"},
        {"Invoice": 800103, "StockCode": "E1006", "Description": "Travel Bottle", "Quantity": 2, "InvoiceDate": "2010-05-18 15:00", "Price": 24.00, "Customer ID": 40101, "Country": "France"},
        {"Invoice": 800004, "StockCode": "E1007", "Description": "Tea Set", "Quantity": 1, "InvoiceDate": "2010-01-25 09:40", "Price": 27.00, "Customer ID": 40102, "Country": "Portugal"},
        {"Invoice": 800104, "StockCode": "E1008", "Description": "Candle Trio", "Quantity": 4, "InvoiceDate": "2010-06-02 18:20", "Price": 7.50, "Customer ID": 40102, "Country": "Portugal"},
    ]
    return pd.DataFrame(demo_train_rows), pd.DataFrame(demo_test_rows)


def get_default_dataset_status():
    if default_artifacts_available():
        return {
            "source": "artifacts",
            "warning": "",
            "caption": "Default mode loads transaction data from `artifacts/`, rebuilds customer features, and creates a fresh stratified customer-level holdout for evaluation.",
        }
    if bundled_sample_available():
        return {
            "source": "sample_data",
            "warning": "Artifact datasets were not found. Using the bundled sample dataset for demo mode.",
            "caption": "Default mode is using bundled sample CSVs from `sample_data/` because deployment environments may not include `artifacts/`.",
        }
    return {
        "source": "demo",
        "warning": "Demo dataset not found. Please upload data or switch mode.",
        "caption": "Default mode is running on a small in-memory demo dataset.",
    }


@st.cache_data
def load_default_raw_datasets():
    dataset_status = get_default_dataset_status()
    if dataset_status["source"] == "artifacts":
        return pd.read_csv(DEFAULT_TRAIN_PATH), pd.read_csv(DEFAULT_TEST_PATH)
    if dataset_status["source"] == "sample_data":
        return pd.read_csv(BUNDLED_TRAIN_PATH), pd.read_csv(BUNDLED_TEST_PATH)
    return create_demo_raw_datasets()


@st.cache_data
def compute_segment_outputs(result_df: pd.DataFrame):
    return analyze_segments(result_df)


@st.cache_data
def preprocess_uploaded_data(raw_df: pd.DataFrame):
    processed_df, kmeans_model = build_customer_features(raw_df)
    clustering_metrics = evaluate_clustering(
        processed_df[CLUSTER_FEATURE_COLUMNS],
        processed_df["final_kmeans_cluster"],
    )
    if hasattr(kmeans_model, "inertia_"):
        clustering_metrics["inertia"] = float(kmeans_model.inertia_)
    clustering_meta = {
        "selected_k": int(getattr(kmeans_model, "selected_k_", kmeans_model.n_clusters)),
        "selection_summary": getattr(kmeans_model, "selection_summary_", []),
        "nan_summary": processed_df.attrs.get("nan_summary", pd.Series(dtype=int)).to_dict(),
        "clustering_metrics": clustering_metrics,
    }
    return processed_df, clustering_meta


@st.cache_data
def prepare_default_datasets():
    from sklearn.model_selection import train_test_split

    train_raw_df, test_raw_df = load_default_raw_datasets()
    combined_raw_df = pd.concat([train_raw_df, test_raw_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    normalized_combined_raw = normalize_transaction_columns(combined_raw_df)
    combined_reference_date = pd.to_datetime(
        normalized_combined_raw["invoicedate"],
        errors="coerce",
    ).max() + pd.Timedelta(days=1)

    combined_customer_df, kmeans_model = build_customer_features(
        combined_raw_df,
        reference_date=combined_reference_date,
    )

    stratify_target = combined_customer_df["retention_status"]
    train_customer_df, test_customer_df = train_test_split(
        combined_customer_df,
        test_size=0.35,
        random_state=42,
        stratify=stratify_target if stratify_target.nunique() > 1 else None,
    )
    train_customer_df = train_customer_df.reset_index(drop=True)
    test_customer_df = test_customer_df.reset_index(drop=True)

    train_customer_ids = set(train_customer_df["customer_id"].astype(str))
    test_customer_ids = set(test_customer_df["customer_id"].astype(str))
    train_raw_df = combined_raw_df[normalized_combined_raw["customer_id"].astype(str).isin(train_customer_ids)].copy()
    test_raw_df = combined_raw_df[normalized_combined_raw["customer_id"].astype(str).isin(test_customer_ids)].copy()

    clustering_metrics = evaluate_clustering(
        train_customer_df[CLUSTER_FEATURE_COLUMNS],
        train_customer_df["final_kmeans_cluster"],
    )
    if hasattr(kmeans_model, "inertia_"):
        clustering_metrics["inertia"] = float(kmeans_model.inertia_)

    clustering_meta = {
        "selected_k": int(getattr(kmeans_model, "selected_k_", kmeans_model.n_clusters)),
        "selection_summary": getattr(kmeans_model, "selection_summary_", []),
        "nan_summary": train_customer_df.attrs.get("nan_summary", pd.Series(dtype=int)).to_dict(),
        "clustering_metrics": clustering_metrics,
    }
    return train_raw_df, test_raw_df, train_customer_df, test_customer_df, clustering_meta


def model_service_input_matrix(customer_df: pd.DataFrame) -> pd.DataFrame:
    model_df = prepare_model_matrix(customer_df)
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(
        imputer.fit_transform(model_df),
        columns=model_df.columns,
        index=model_df.index,
    )


def build_train_test_model_matrices(train_customer_df: pd.DataFrame, test_customer_df: pd.DataFrame):
    train_model_df = prepare_model_matrix(train_customer_df)
    test_model_df = prepare_model_matrix(
        test_customer_df,
        training_columns=train_model_df.columns.tolist(),
    )
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(train_model_df),
        columns=train_model_df.columns,
        index=train_model_df.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(test_model_df),
        columns=test_model_df.columns,
        index=test_model_df.index,
    )
    return X_train, X_test


def train_model_service(model_service: ChurnModelService, model_df: pd.DataFrame, target: pd.Series):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        model_df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if target.nunique() > 1 else None,
    )
    model_service.train(X_train, y_train, X_test, y_test)
    return model_service


@st.cache_resource
def train_dynamic_model(customer_df: pd.DataFrame):
    model_service = ChurnModelService()
    model_df = model_service_input_matrix(customer_df)
    target = customer_df["retention_status"].astype(int)
    return train_model_service(model_service, model_df, target)


@st.cache_resource
def train_default_model_service():
    _, _, train_customer_df, test_customer_df, _ = prepare_default_datasets()
    X_train, X_test = build_train_test_model_matrices(train_customer_df, test_customer_df)
    y_train = train_customer_df["retention_status"].astype(int)
    y_test = test_customer_df["retention_status"].astype(int)
    model_service = ChurnModelService()
    model_service.train(X_train, y_train, X_test, y_test)
    return model_service


def make_display_table(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    display_df = df.rename(columns=rename_map).copy()
    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(2)
    return display_df


def render_section_header(title: str, description: str):
    st.markdown(f"## {title}")
    st.markdown(description)


def compute_live_threshold_metrics(result_df: pd.DataFrame):
    if "Churn_Label_Actual" not in result_df.columns or "Predicted_Label" not in result_df.columns:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "predictions_imbalanced": False,
            "confusion_matrix": None,
        }

    y_true = result_df["Churn_Label_Actual"]
    y_pred = result_df["Predicted_Label"]
    y_prob = result_df["Churn Probability"]
    roc_auc = None
    if len(np.unique(y_true)) > 1 and len(np.unique(y_prob)) > 1:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(y_true, y_prob))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "predictions_imbalanced": len(np.unique(y_pred)) == 1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def format_metric_value(value, fmt: str = ".2f") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not Available"
    return f"{value:{fmt}}"


def build_threshold_tradeoff_table(result_df: pd.DataFrame) -> pd.DataFrame:
    if "Churn_Label_Actual" not in result_df.columns or result_df.empty:
        return pd.DataFrame()

    y_true = result_df["Churn_Label_Actual"].astype(int)
    y_prob = result_df["Churn Probability"].astype(float)
    rows = []
    for threshold in np.arange(0.2, 0.85, 0.1):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
                "Predicted Positives": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_impact_table(segment_outputs: dict, save_rate: float, cost_per_customer: float, horizon_days: int):
    rows = []
    for persona in segment_outputs["personas"].values():
        actions = persona["recommended_actions"]
        at_risk_customers = persona["customer_count"] * persona["predicted_churn_rate_pct"] / 100
        estimated_saved_customers = at_risk_customers * save_rate
        avg_daily_value = 0.0
        summary_row = segment_outputs["summary"]
        summary_row = summary_row[summary_row["cluster"] == persona["cluster_id"]].iloc[0]
        if "monetary_per_day" in summary_row.index:
            avg_daily_value = float(summary_row["monetary_per_day"])
        protected_revenue = estimated_saved_customers * avg_daily_value * horizon_days
        campaign_cost = persona["customer_count"] * cost_per_customer
        net_impact = protected_revenue - campaign_cost
        rows.append(
            {
                "Persona": persona["persona_name"],
                "Risk Level": persona["risk_level"],
                "Primary Action": actions["primary_action"],
                "Targeting Rule": actions["targeting_rule"],
                "Owner": actions["owner"],
                "At-Risk Customers": round(at_risk_customers, 1),
                "Estimated Saved Customers": round(estimated_saved_customers, 1),
                "Protected Revenue": round(protected_revenue, 2),
                "Campaign Cost": round(campaign_cost, 2),
                "Net Impact": round(net_impact, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Net Impact", ascending=False)


def build_report_markdown(
    overview: dict,
    persona_table: pd.DataFrame,
    importance_table: pd.DataFrame,
    churn_segment_table: pd.DataFrame,
    insights: list,
    action_table: pd.DataFrame,
    impact_table: pd.DataFrame,
    clustering_note: str,
) -> str:
    report = [
        "# Customer Churn Intelligence Report",
        "",
        "## Portfolio Overview",
        f"- Transactions uploaded: {overview['transactions_uploaded']}",
        f"- Customers scored: {overview['customers_scored']}",
        f"- Portfolio churn rate: {overview['portfolio_churn_rate']:.2f}%",
        f"- Unique segments: {overview['unique_segments']}",
        "",
        "## Clustering Health",
        f"- {clustering_note}",
        "",
        "## Customer Personas",
        "```csv",
        persona_table.to_csv(index=False).strip(),
        "```",
        "",
        "## Feature Importance",
        "```csv" if not importance_table.empty else "Feature importance not available for current saved model.",
        importance_table.to_csv(index=False).strip() if not importance_table.empty else "",
        "```" if not importance_table.empty else "",
        "",
        "## Churn by Segment",
        "```csv",
        churn_segment_table.to_csv(index=False).strip(),
        "```",
        "",
        "## Key Insights",
    ]
    report.extend([f"- {insight}" for insight in insights])
    report.extend(
        [
            "",
            "## Business Recommendations",
            "```csv",
            action_table.to_csv(index=False).strip(),
            "```",
            "",
            "## What-If Impact Simulation",
            "```csv",
            impact_table.to_csv(index=False).strip(),
            "```",
            "",
        ]
    )
    return "\n".join(report)


def build_excel_report(
    input_preview: pd.DataFrame,
    result_df: pd.DataFrame,
    persona_table: pd.DataFrame,
    importance_table: pd.DataFrame,
    churn_segment_table: pd.DataFrame,
    action_table: pd.DataFrame,
    impact_table: pd.DataFrame,
    insights: list,
    clustering_note: str,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_preview.to_excel(writer, sheet_name="Data Overview", index=False)
        result_df.to_excel(writer, sheet_name="Customer Predictions", index=False)
        persona_table.to_excel(writer, sheet_name="Personas", index=False)
        churn_segment_table.to_excel(writer, sheet_name="Churn by Segment", index=False)
        action_table.to_excel(writer, sheet_name="Recommendations", index=False)
        impact_table.to_excel(writer, sheet_name="Impact Simulator", index=False)

        if not importance_table.empty:
            importance_table.to_excel(writer, sheet_name="Feature Importance", index=False)

        insights_df = pd.DataFrame(
            {
                "Section": ["Clustering Health", *["Key Insight"] * len(insights)],
                "Detail": [clustering_note, *insights],
            }
        )
        insights_df.to_excel(writer, sheet_name="Insights", index=False)

    output.seek(0)
    return output.getvalue()


def render_confusion_matrix(confusion_values, title: str):
    if confusion_values is None:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        pd.DataFrame(
            confusion_values,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        ),
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_roc_curve(roc_curve_points, title: str):
    if not roc_curve_points:
        st.info("ROC curve is not available for the current dataset.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(roc_curve_points["fpr"], roc_curve_points["tpr"], linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_model_insights_section(model_service, live_metrics, explainability, context_label: str):
    st.header("📊 Model Insights")
    st.caption(f"Evaluation view for {context_label}")
    if live_metrics.get("predictions_imbalanced"):
        st.warning("Model predictions are imbalanced. Adjust threshold.")

    metrics_row = st.columns(5)
    metrics_row[0].metric("Accuracy", format_metric_value(live_metrics["accuracy"], ".2%"))
    metrics_row[1].metric("Precision", format_metric_value(live_metrics["precision"]))
    metrics_row[2].metric("Recall", format_metric_value(live_metrics["recall"]))
    metrics_row[3].metric("F1 Score", format_metric_value(live_metrics["f1_score"]))
    metrics_row[4].metric("ROC-AUC", format_metric_value(live_metrics.get("roc_auc")))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Confusion Matrix**")
        if live_metrics["confusion_matrix"] is not None:
            render_confusion_matrix(live_metrics["confusion_matrix"], "Confusion Matrix")
        else:
            st.info("Confusion matrix is not available because actual labels are missing.")
    with chart_col2:
        st.markdown("**ROC Curve**")
        render_roc_curve(model_service.roc_curve_points, "ROC Curve")

    st.markdown("**Feature Importance**")
    if not explainability["table"].empty:
        feature_chart = explainability["table"].copy()
        feature_chart["Importance (%)"] = feature_chart["Importance (%)"].round(2)
        st.bar_chart(feature_chart.set_index("Feature")[["Importance (%)"]], use_container_width=True)
    else:
        st.info("Feature importance is not available for the current model.")


def render_dataset_dashboard(
    input_df: pd.DataFrame,
    clustering_meta: dict,
    model_service: ChurnModelService,
    base_result_df: pd.DataFrame,
    threshold: float,
    intervention_save_rate: float,
    campaign_cost_per_customer: float,
    impact_horizon_days: int,
    show_advanced: bool,
    dataset_label: str,
    success_message: str | None = None,
):
    result_df = base_result_df.copy()
    logger.info("Churn probability distribution:\n%s", result_df["Churn Probability"].describe().to_string())
    logger.info("Max churn probability: %.4f", float(result_df["Churn Probability"].max()))
    result_df = apply_prediction_threshold(result_df, threshold)
    result_df, risk_cutoffs = apply_dynamic_risk_segmentation(result_df)

    if success_message:
        st.success(success_message)

    st.info(
        "Prediction vs Risk Segmentation:\n\n"
        f"- Prediction threshold for metrics and binary labels: {threshold:.2f}\n"
        f"- Low/Medium cutoff (33rd percentile): {risk_cutoffs['q1']:.3f}\n"
        f"- Medium/High cutoff (66th percentile): {risk_cutoffs['q2']:.3f}\n\n"
        "Binary model metrics use the threshold-based `Predicted_Label`, while dashboard targeting and personas use the quantile-based `Risk Segment`."
    )

    segment_outputs = compute_segment_outputs(result_df)
    explainability = model_service.explain_feature_importance(top_n=10)
    churn_driver_insights = model_service.explain_customer_churn_pattern(result_df)
    live_metrics = compute_live_threshold_metrics(result_df)
    threshold_tradeoff = build_threshold_tradeoff_table(result_df)
    impact_table = build_impact_table(
        segment_outputs,
        save_rate=intervention_save_rate,
        cost_per_customer=campaign_cost_per_customer,
        horizon_days=impact_horizon_days,
    )

    largest_share = (
        segment_outputs["summary"]["segment_share_pct"].max()
        if not segment_outputs["summary"].empty
        else 0
    )
    clustering_note = (
        f"The current segmentation model selected K={clustering_meta['selected_k']}. "
        f"The largest segment represents {largest_share:.1f}% of customers."
    )

    render_section_header("Data Overview", f"This section summarizes the source data and scoring output for {dataset_label}.")
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Transactions Loaded", len(input_df))
    overview_col2.metric("Customers Scored", len(result_df))
    overview_col3.metric("Unique Segments", result_df["cluster"].nunique())
    st.dataframe(input_df.head(), use_container_width=True)

    if largest_share >= 50:
        st.warning(
            f"Segment imbalance detected: the largest segment holds {largest_share:.1f}% of customers. "
            "Retraining with automatic K search across 3-5 clusters is supported in the preprocessing pipeline."
        )
    else:
        st.success(clustering_note)

    render_section_header(
        "Customer Segmentation (Personas)",
        "Clusters are translated into business personas using relative feature averages, calibrated churn risk bands, and behavioral interpretation.",
    )
    persona_table = make_display_table(
        segment_outputs["profiles"],
        {
            "cluster": "Cluster ID",
            "persona_name": "Persona",
            "customer_count": "Customers",
            "segment_share_pct": "Share (%)",
            "predicted_churn_rate_pct": "High Risk Share (%)",
            "avg_churn_probability_pct": "Avg Churn Probability (%)",
            "risk_level": "Risk Level",
            "frequency_log": "Avg Frequency (log)",
            "monetary_log": "Avg Monetary (log)",
            "tenure": "Avg Tenure",
            "avg_order_value": "Avg Order Value",
            "purchase_rate": "Avg Purchase Rate",
            "unique_items_purchased": "Avg Unique Items",
            "monetary_per_day": "Avg Monetary / Day",
        },
    )
    st.dataframe(persona_table, use_container_width=True)

    for _, persona in sorted(segment_outputs["personas"].items()):
        actions = persona["recommended_actions"]
        with st.expander(
            f"{persona['persona_name']} | {persona['risk_badge']} risk | Customers: {persona['customer_count']}",
            expanded=True,
        ):
            st.markdown(f"**Key characteristics:** {', '.join(persona['key_characteristics'])}")
            st.markdown(f"**Behavioral interpretation:** {persona['behavioral_interpretation']}")
            st.markdown(f"**Risk level:** {persona['risk_badge']} ({persona['predicted_churn_rate_pct']:.1f}% high-risk share)")
            st.markdown(f"**Primary action:** {actions['primary_action']}")
            st.markdown(f"**Targeting rule:** {actions['targeting_rule']}")
            st.markdown(f"**Action owner:** {actions['owner']}")

    render_section_header(
        "Prediction Results",
        "The model scores each customer, highlights risk concentration, and adds business-style risk segmentation.",
    )
    churn_col1, churn_col2, churn_col3, churn_col4 = st.columns(4)
    predicted_churn = int(result_df["Predicted_Label"].sum())
    churn_rate = (predicted_churn / len(result_df) * 100) if len(result_df) else 0
    churn_col1.metric("Predicted Churn Customers", predicted_churn)
    churn_col2.metric("Portfolio Churn Rate", f"{churn_rate:.2f}%")
    churn_col3.metric("Avg Churn Probability", f"{result_df['Churn Probability'].mean() * 100:.2f}%")
    churn_col4.metric("High Risk Customers", int((result_df["Risk Segment"] == "High Risk").sum()))
    risk_distribution = result_df["Risk Segment"].value_counts().rename_axis("Risk").reset_index(name="Customers")
    st.bar_chart(risk_distribution.set_index("Risk"), use_container_width=True)
    st.dataframe(result_df.sort_values("Churn Probability", ascending=False).head(20), use_container_width=True)

    if not threshold_tradeoff.empty:
        st.markdown("**Threshold Impact**")
        st.line_chart(
            threshold_tradeoff.set_index("Threshold")[["Precision", "Recall", "F1 Score"]],
            use_container_width=True,
        )
        st.dataframe(threshold_tradeoff, use_container_width=True)

    render_model_insights_section(model_service, live_metrics, explainability, dataset_label)

    render_section_header(
        "Feature Importance / Explainability",
        "Tree-model importance ranks the strongest churn drivers so the model output can be linked to interpretable customer behavior.",
    )
    st.markdown(explainability["summary"])
    if not explainability["table"].empty:
        importance_table = explainability["table"].copy()
        importance_table["Importance"] = importance_table["Importance"].round(4)
        importance_table["Importance (%)"] = importance_table["Importance (%)"].round(2)
        st.dataframe(importance_table, use_container_width=True)
        st.bar_chart(importance_table.set_index("Feature")[["Importance (%)"]], use_container_width=True)
        for item in explainability["driver_insights"]:
            st.write(f"- {item}")
    else:
        importance_table = explainability["table"].copy()
        st.info("The current model does not expose tree-based feature importance.")

    render_section_header(
        "Churn by Segment",
        "This section connects clustering to churn so teams can see which persona deserves immediate attention.",
    )
    churn_segment_table = make_display_table(
        segment_outputs["churn_table"],
        {
            "persona_name": "Persona",
            "customer_count": "Customers",
            "segment_share_pct": "Share (%)",
            "predicted_churn_rate_pct": "High Risk Share (%)",
            "avg_churn_probability_pct": "Avg Churn Probability (%)",
            "risk_level": "Risk Level",
        },
    )
    st.dataframe(churn_segment_table, use_container_width=True)
    churn_chart = segment_outputs["churn_table"].set_index("persona_name")[["predicted_churn_rate_pct"]]
    churn_chart.columns = ["High Risk Share (%)"]
    st.bar_chart(churn_chart, use_container_width=True)

    render_section_header(
        "Key Insights",
        "Automatically generated analyst-style conclusions based on the actual segment and churn distributions in the scored data.",
    )
    insight_list = []
    insight_list.extend(segment_outputs["insights"])
    insight_list.extend(churn_driver_insights)
    risky = result_df[result_df["Predicted_Label"] == 1]
    stable = result_df[result_df["Predicted_Label"] == 0]
    if not risky.empty and not stable.empty:
        if risky["avg_order_value"].mean() > stable["avg_order_value"].mean() and risky["purchase_rate"].mean() < stable["purchase_rate"].mean():
            insight_list.append(
                "Customers with higher order values but weaker purchase cadence show elevated churn, which points to revenue-rich accounts losing engagement between purchases."
            )
        if risky["tenure"].mean() < stable["tenure"].mean():
            insight_list.append(
                "Newer customers are churning earlier than the retained base, suggesting onboarding and early-value realization remain key retention gaps."
            )

    seen = set()
    clean_insights = []
    for insight in insight_list:
        if insight not in seen:
            st.write(f"- {insight}")
            clean_insights.append(insight)
            seen.add(insight)

    st.subheader("Additional Customer Insights")
    if not result_df.empty:
        top_customer = result_df.sort_values("Churn Probability", ascending=False).iloc[0]
        generated_insights = generate_insights(top_customer)
        if generated_insights:
            for item in generated_insights:
                st.write(f"- {item}")
        else:
            st.write("- No rule-based insights triggered for the highest-risk customer.")

    render_section_header(
        "Business Recommendations",
        "This is the explicit persona-to-action layer: each segment is mapped to a primary action, targeting rule, owner, and execution play.",
    )
    st.dataframe(segment_outputs["action_table"], use_container_width=True)

    render_section_header(
        "What-If Impact Simulator",
        "This section estimates how many customers could be saved and how much revenue could be protected under your assumed intervention uplift and campaign cost.",
    )
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_col1.metric("Expected Save Rate", f"{intervention_save_rate * 100:.0f}%")
    sim_col2.metric("Campaign Cost / Customer", f"{campaign_cost_per_customer:.2f}")
    sim_col3.metric("Impact Horizon", f"{impact_horizon_days} days")
    st.dataframe(impact_table, use_container_width=True)
    st.bar_chart(impact_table.set_index("Persona")[["Net Impact"]], use_container_width=True)

    report_overview = {
        "transactions_uploaded": len(input_df),
        "customers_scored": len(result_df),
        "portfolio_churn_rate": churn_rate,
        "unique_segments": result_df["cluster"].nunique(),
    }
    report_markdown = build_report_markdown(
        overview=report_overview,
        persona_table=persona_table,
        importance_table=importance_table,
        churn_segment_table=churn_segment_table,
        insights=clean_insights,
        action_table=segment_outputs["action_table"],
        impact_table=impact_table,
        clustering_note=clustering_note,
    )
    excel_report = build_excel_report(
        input_preview=input_df.head(100),
        result_df=result_df,
        persona_table=persona_table,
        importance_table=importance_table,
        churn_segment_table=churn_segment_table,
        action_table=segment_outputs["action_table"],
        impact_table=impact_table,
        insights=clean_insights,
        clustering_note=clustering_note,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Full Report",
            data=report_markdown.encode("utf-8"),
            file_name="customer_churn_intelligence_report.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            "Download All Dashboard Data",
            data=excel_report,
            file_name="customer_churn_intelligence_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if show_advanced:
        render_section_header("Advanced Model Diagnostics", "Optional model internals for technical review and recruiter walkthroughs.")
        st.write(f"Min churn probability: {result_df['Churn Probability'].min():.4f}")
        st.write(f"Max churn probability: {result_df['Churn Probability'].max():.4f}")
        st.write(f"Mean churn probability: {result_df['Churn Probability'].mean():.4f}")

        selection_summary = clustering_meta.get("selection_summary")
        if selection_summary:
            st.markdown("**Cluster search summary used during training**")
            st.dataframe(pd.DataFrame(selection_summary), use_container_width=True)

        nan_summary = clustering_meta.get("nan_summary", {})
        if nan_summary:
            st.markdown("**Feature-engineering null check**")
            st.dataframe(
                pd.DataFrame({"column": list(nan_summary.keys()), "null_count": list(nan_summary.values())}),
                use_container_width=True,
            )

        if model_service.classification_report_text:
            st.markdown("**Classification Report**")
            st.code(model_service.classification_report_text)

        try:
            import shap

            explainer = shap.TreeExplainer(model_service.model)
            shap_values = explainer.shap_values(model_service.last_processed_df.values)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                model_service.last_processed_df.values,
                feature_names=model_service.features,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as shap_error:
            st.info(f"SHAP chart unavailable for the current model: {shap_error}")


@st.cache_data
def get_default_persona_lookup():
    _, _, train_customer_df, _, _ = prepare_default_datasets()
    model_service = train_default_model_service()
    train_predictions = model_service.score_customer_df(train_customer_df)
    train_predictions, risk_cutoffs = apply_dynamic_risk_segmentation(train_predictions)
    segment_outputs = compute_segment_outputs(train_predictions)

    persona_options = []
    for cluster_id, persona in sorted(segment_outputs["personas"].items()):
        persona_options.append(
            {
                "label": f"{persona['persona_name']} (Cluster {cluster_id})",
                "cluster_id": cluster_id,
                "details": persona,
            }
        )
    return persona_options, risk_cutoffs


def render_single_prediction_mode():
    st.header("Single Prediction")
    st.write("Score one customer profile using the default trained model and map the result to a business-friendly risk segment.")
    dataset_status = get_default_dataset_status()
    if dataset_status["warning"]:
        st.warning(dataset_status["warning"])
    st.caption(dataset_status["caption"])

    model_service = train_default_model_service()
    persona_options, risk_cutoffs = get_default_persona_lookup()
    persona_labels = [option["label"] for option in persona_options]

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_persona_label = st.selectbox("Customer Segment", persona_labels)
            frequency = st.number_input("Purchase Frequency", min_value=0.0, value=6.0, step=1.0)
            monetary = st.number_input("Total Monetary Value", min_value=0.0, value=600.0, step=50.0)
            tenure = st.slider("Tenure (days)", min_value=0, max_value=1500, value=180)
            avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=100.0, step=10.0)
        with col2:
            unique_items_purchased = st.slider("Unique Items Purchased", min_value=1, max_value=250, value=12)
            purchase_rate = st.number_input("Purchase Rate", min_value=0.0, value=0.15, step=0.01, format="%.2f")
            monetary_per_day = st.number_input("Monetary Per Day", min_value=0.0, value=3.50, step=0.10, format="%.2f")
            customer_id = st.text_input("Customer ID", value="CUSTOMER_DEMO_001")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    selected_persona = next(option for option in persona_options if option["label"] == selected_persona_label)
    input_customer_df = pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "frequency_log": np.log1p(frequency),
                "monetary_log": np.log1p(monetary),
                "tenure": tenure,
                "avg_order_value": avg_order_value,
                "unique_items_purchased": unique_items_purchased,
                "purchase_rate": purchase_rate,
                "monetary_per_day": monetary_per_day,
                "final_kmeans_cluster": selected_persona["cluster_id"],
            }
        ]
    )

    prediction_df = model_service.score_customer_df(input_customer_df)
    probability = float(prediction_df.loc[0, "Churn Probability"])
    risk_segment = map_dynamic_risk(probability, risk_cutoffs["q1"], risk_cutoffs["q2"])
    predicted_label = int(probability >= threshold)
    actions = selected_persona["details"]["recommended_actions"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{probability:.2%}")
    col2.metric("Predicted Label", "Churn" if predicted_label == 1 else "Retained")
    col3.metric("Risk Segment", risk_segment)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Customer ID": customer_id,
                    "Segment": selected_persona_label,
                    "Churn Probability": round(probability, 4),
                    "Predicted Label": predicted_label,
                    "Risk Segment": risk_segment,
                }
            ]
        ),
        use_container_width=True,
    )
    st.markdown("### Recommended Action")
    st.write(f"Primary action: {actions['primary_action']}")
    st.write(f"Targeting rule: {actions['targeting_rule']}")
    st.write(f"Owner: {actions['owner']}")


st.set_page_config(page_title="Customer Churn Intelligence Hub", layout="wide")

with st.sidebar:
    st.title("Decision Controls")
    mode = st.selectbox("Choose Mode", ["Model Insights", "Upload Dataset", "Single Prediction"])
    threshold = st.slider(
        "Churn Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Higher threshold = fewer customers targeted (higher precision). Lower threshold = more customers targeted (higher recall).",
    )
    intervention_save_rate = st.slider("Expected Save Rate", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    campaign_cost_per_customer = st.number_input("Campaign Cost per Customer", min_value=0.0, value=8.0, step=1.0)
    impact_horizon_days = st.slider("Impact Horizon (Days)", min_value=30, max_value=180, value=90, step=30)
    show_advanced = st.checkbox("Show SHAP and model internals")
    st.markdown("---")
    st.markdown("### Dashboard Purpose")
    st.write("Move from raw predictions to segment-led retention decisions.")

st.title("Customer Churn Intelligence Hub")
st.markdown(
    "A business dashboard that combines clustering, churn prediction, explainability, recommended actions, and what-if retention impact at the customer-segment level."
)

try:
    if mode == "Model Insights":
        dataset_status = get_default_dataset_status()
        if dataset_status["warning"]:
            st.warning(dataset_status["warning"])
        _, test_raw_df, _, test_customer_df, clustering_meta = prepare_default_datasets()
        model_service = train_default_model_service()
        result_df = model_service.score_customer_df(test_customer_df)
        st.caption(dataset_status["caption"])
        render_dataset_dashboard(
            input_df=test_raw_df,
            clustering_meta=clustering_meta,
            model_service=model_service,
            base_result_df=result_df,
            threshold=threshold,
            intervention_save_rate=intervention_save_rate,
            campaign_cost_per_customer=campaign_cost_per_customer,
            impact_horizon_days=impact_horizon_days,
            show_advanced=show_advanced,
            dataset_label="the rebuilt default customer holdout dataset",
            success_message=(
                "Default model insights loaded from artifact data with a rebuilt stratified customer holdout."
                if dataset_status["source"] == "artifacts"
                else "Default model insights loaded with fallback demo data."
            ),
        )
    elif mode == "Upload Dataset":
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is None:
            st.info("Upload a CSV or Excel file to run the full segmentation, churn prediction, explainability, and impact workflow on your own dataset.")
        else:
            input_df = load_uploaded_data(uploaded_file)
            processed_df, clustering_meta = preprocess_uploaded_data(input_df)
            model_service = train_dynamic_model(processed_df)
            result_df = model_service.score_customer_df(processed_df)
            render_dataset_dashboard(
                input_df=input_df,
                clustering_meta=clustering_meta,
                model_service=model_service,
                base_result_df=result_df,
                threshold=threshold,
                intervention_save_rate=intervention_save_rate,
                campaign_cost_per_customer=campaign_cost_per_customer,
                impact_horizon_days=impact_horizon_days,
                show_advanced=show_advanced,
                dataset_label="the uploaded dataset",
                success_message="Model trained dynamically on the uploaded dataset.",
            )
    else:
        render_single_prediction_mode()
except Exception as exc:
    logger.exception("Dashboard failed")
    st.error(str(exc))

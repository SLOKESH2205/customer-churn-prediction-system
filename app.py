import logging
from io import BytesIO

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.clustering import evaluate_clustering
from src.components.segment_analytics import analyze_segments
from src.modeling import ChurnModelService
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, build_customer_features


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


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


@st.cache_resource
def train_dynamic_model(customer_df: pd.DataFrame):
    model_service = ChurnModelService()
    model_df = model_service_input_matrix(customer_df)
    target = customer_df["retention_status"].astype(int)
    return train_model_service(model_service, model_df, target)


def model_service_input_matrix(customer_df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from src.preprocessing import prepare_model_matrix

    model_df = prepare_model_matrix(customer_df)
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(
        imputer.fit_transform(model_df),
        columns=model_df.columns,
        index=model_df.index,
    )


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


def make_display_table(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    display_df = df.rename(columns=rename_map).copy()
    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(2)
    return display_df


def render_section_header(title: str, description: str):
    st.markdown(f"## {title}")
    st.markdown(description)


def compute_live_threshold_metrics(result_df: pd.DataFrame):
    if "Churn_Label_Actual" not in result_df.columns:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "confusion_matrix": None,
        }

    y_true = result_df["Churn_Label_Actual"]
    y_pred = result_df["Churn_Label"]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def build_impact_table(
    segment_outputs: dict,
    save_rate: float,
    cost_per_customer: float,
    horizon_days: int,
):
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


st.set_page_config(
    page_title="Customer Churn Intelligence Hub",
    layout="wide",
)

with st.sidebar:
    st.title("Decision Controls")
    threshold = st.slider(
        "Churn Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Higher threshold = fewer customers targeted (higher precision). Lower threshold = more customers targeted (higher recall).",
    )
    intervention_save_rate = st.slider(
        "Expected Save Rate",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
    )
    campaign_cost_per_customer = st.number_input(
        "Campaign Cost per Customer",
        min_value=0.0,
        value=8.0,
        step=1.0,
    )
    impact_horizon_days = st.slider(
        "Impact Horizon (Days)",
        min_value=30,
        max_value=180,
        value=90,
        step=30,
    )
    show_advanced = st.checkbox("Show SHAP and model internals")
    st.markdown("---")
    st.markdown("### Dashboard Purpose")
    st.write("Move from raw predictions to segment-led retention decisions.")

st.title("Customer Churn Intelligence Hub")
st.markdown(
    "A business dashboard that combines clustering, churn prediction, explainability, recommended actions, and what-if retention impact at the customer-segment level."
)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        input_df = load_uploaded_data(uploaded_file)
        processed_df, clustering_meta = preprocess_uploaded_data(input_df)
        model_service = train_dynamic_model(processed_df)
        result_df = model_service.score_customer_df(processed_df)
        result_df["Churn_Label"] = (
            result_df["Churn Probability"] >= threshold
        ).astype(int)
        st.success("Model trained dynamically on uploaded data")
        st.info(
            "Threshold Trade-off:\n\n"
            "- Lower threshold -> capture more churn customers (high recall) but more false positives\n"
            "- Higher threshold -> fewer false positives (high precision) but may miss some churners\n\n"
            "Use this control to balance marketing cost versus retention coverage."
        )

        segment_outputs = compute_segment_outputs(result_df)
        explainability = model_service.explain_feature_importance(top_n=10)
        churn_driver_insights = model_service.explain_customer_churn_pattern(result_df)
        live_metrics = compute_live_threshold_metrics(result_df)
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
        selected_k = clustering_meta["selected_k"]
        clustering_note = (
            f"The current uploaded-data segmentation model selected K={selected_k}. The largest segment represents {largest_share:.1f}% of customers."
        )

        render_section_header(
            "Data Overview",
            "This section summarizes the uploaded transaction data and the customer-level scoring base used in the dashboard.",
        )
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        overview_col1.metric("Transactions Uploaded", len(input_df))
        overview_col2.metric("Customers Scored", len(result_df))
        overview_col3.metric("Unique Segments", result_df["cluster"].nunique())
        st.dataframe(input_df.head(), use_container_width=True)

        if largest_share >= 50:
            st.warning(
                f"Segment imbalance detected: the largest segment holds {largest_share:.1f}% of customers. Retraining with automatic K search across 3-5 clusters is now supported in the preprocessing and clustering modules."
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
                "predicted_churn_rate_pct": "Predicted Churn (%)",
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

        for cluster_id, persona in sorted(segment_outputs["personas"].items()):
            actions = persona["recommended_actions"]
            with st.expander(
                f"{persona['persona_name']} | {persona['risk_badge']} risk | Customers: {persona['customer_count']}",
                expanded=True,
            ):
                st.markdown(f"**Key characteristics:** {', '.join(persona['key_characteristics'])}")
                st.markdown(f"**Behavioral interpretation:** {persona['behavioral_interpretation']}")
                st.markdown(f"**Risk level:** {persona['risk_badge']} ({persona['predicted_churn_rate_pct']:.1f}% churn)")
                st.markdown(f"**Primary action:** {actions['primary_action']}")
                st.markdown(f"**Targeting rule:** {actions['targeting_rule']}")
                st.markdown(f"**Action owner:** {actions['owner']}")

        render_section_header(
            "Churn Prediction",
            "The model scores each customer, highlights risk concentration, and explains which features drive churn predictions.",
        )
        churn_col1, churn_col2, churn_col3 = st.columns(3)
        predicted_churn = int(result_df["Churn_Label"].sum())
        churn_rate = (predicted_churn / len(result_df) * 100) if len(result_df) else 0
        churn_col1.metric("Predicted Churn Customers", predicted_churn)
        churn_col2.metric("Portfolio Churn Rate", f"{churn_rate:.2f}%")
        churn_col3.metric("Avg Churn Probability", f"{result_df['Churn Probability'].mean() * 100:.2f}%")

        st.markdown("### Targeting Impact")
        target_col1, target_col2 = st.columns(2)
        target_col1.metric("Customers Targeted", predicted_churn)
        target_col2.metric("Target Rate (%)", f"{churn_rate:.2f}")
        st.dataframe(
            result_df.sort_values("Churn Probability", ascending=False).head(20),
            use_container_width=True,
        )

        render_section_header(
            "Model Performance",
            "Clustering quality and churn-model validation metrics are shown here so the pipeline can be evaluated, not just demonstrated.",
        )
        clustering_metrics = clustering_meta.get("clustering_metrics", {})
        cluster_col1, cluster_col2, cluster_col3 = st.columns(3)
        cluster_col1.metric(
            "Silhouette Score",
            f"{clustering_metrics.get('silhouette_score', float('nan')):.3f}",
        )
        cluster_col2.metric(
            "Davies-Bouldin Index",
            f"{clustering_metrics.get('davies_bouldin_index', float('nan')):.3f}",
        )
        cluster_col3.metric(
            "Inertia",
            f"{clustering_metrics.get('inertia', float('nan')):.1f}",
        )

        clf_metrics = live_metrics
        if clf_metrics["accuracy"] is None:
            st.warning("Threshold-based evaluation metrics are not available because actual labels are not present in the current result set.")
            st.info(
                "This dataset does not contain actual churn labels.\n\n"
                "The model outputs predicted churn probabilities, but evaluation metrics "
                "(accuracy, precision, recall) require known outcomes.\n\n"
                "In real-world deployment, these metrics are computed after observing actual customer behavior over time."
            )

        perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
        perf_col1.metric("Accuracy", round(clf_metrics["accuracy"], 2) if clf_metrics["accuracy"] is not None else "NA")
        perf_col2.metric("Precision", round(clf_metrics["precision"], 2) if clf_metrics["precision"] is not None else "NA")
        perf_col3.metric("Recall", round(clf_metrics["recall"], 2) if clf_metrics["recall"] is not None else "NA")
        perf_col4.metric("F1 Score", round(clf_metrics["f1_score"], 2) if clf_metrics["f1_score"] is not None else "NA")
        perf_col5.metric(
            "ROC-AUC",
            f"{model_service.evaluation_metrics.get('roc_auc', float('nan')):.2f}" if model_service.evaluation_metrics else "NA",
        )

        if clf_metrics["confusion_matrix"] is not None:
            st.markdown("**Confusion Matrix**")
            confusion_df = pd.DataFrame(
                clf_metrics["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            )
            st.dataframe(confusion_df, use_container_width=True)

        if model_service.roc_curve_points:
            st.markdown("**ROC Curve**")
            roc_df = pd.DataFrame(model_service.roc_curve_points)
            st.line_chart(roc_df.rename(columns={"fpr": "False Positive Rate", "tpr": "True Positive Rate"}))

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
            st.bar_chart(explainability["table"].set_index("Feature")[["Importance (%)"]])
            for item in explainability["driver_insights"]:
                st.write(f"- {item}")
        else:
            importance_table = explainability["table"].copy()
            st.info("The current saved model does not expose tree-based feature importance.")

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
                "predicted_churn_rate_pct": "Predicted Churn (%)",
                "avg_churn_probability_pct": "Avg Churn Probability (%)",
                "risk_level": "Risk Level",
            },
        )
        st.dataframe(churn_segment_table, use_container_width=True)
        churn_chart = segment_outputs["churn_table"].set_index("persona_name")[
            ["predicted_churn_rate_pct"]
        ]
        churn_chart.columns = ["Predicted Churn (%)"]
        st.bar_chart(churn_chart)

        render_section_header(
            "Key Insights",
            "Automatically generated analyst-style conclusions based on the actual segment and churn distributions in the uploaded data.",
        )
        insight_list = []
        insight_list.extend(segment_outputs["insights"])
        insight_list.extend(churn_driver_insights)
        risky = result_df[result_df["Churn_Label"] == 1]
        stable = result_df[result_df["Churn_Label"] == 0]
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

        st.subheader("💡 Insights")
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

        st.subheader("📌 Business Impact")
        st.write("This system helps identify high-risk customers early and enables targeted retention strategies.")

        render_section_header(
            "What-If Impact Simulator",
            "This section estimates how many customers could be saved and how much revenue could be protected under your assumed intervention uplift and campaign cost.",
        )
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        sim_col1.metric("Expected Save Rate", f"{intervention_save_rate * 100:.0f}%")
        sim_col2.metric("Campaign Cost / Customer", f"{campaign_cost_per_customer:.2f}")
        sim_col3.metric("Impact Horizon", f"{impact_horizon_days} days")
        st.dataframe(impact_table, use_container_width=True)
        st.bar_chart(impact_table.set_index("Persona")[["Net Impact"]])

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
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download Full Report",
                data=report_markdown.encode("utf-8"),
                file_name="customer_churn_intelligence_report.md",
                mime="text/markdown",
            )
        with download_col2:
            st.download_button(
                "Download All Dashboard Data",
                data=excel_report,
                file_name="customer_churn_intelligence_dashboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if show_advanced:
            render_section_header(
                "Advanced Model Diagnostics",
                "Optional model internals for technical review and recruiter walkthroughs.",
            )
            st.write(f"Min churn probability: {result_df['Churn Probability'].min():.4f}")
            st.write(f"Max churn probability: {result_df['Churn Probability'].max():.4f}")
            st.write(f"Mean churn probability: {result_df['Churn Probability'].mean():.4f}")

            selection_summary = clustering_meta.get("selection_summary")
            if selection_summary:
                st.markdown("**Cluster search summary used during dynamic training**")
                st.dataframe(pd.DataFrame(selection_summary), use_container_width=True)

            nan_summary = clustering_meta.get("nan_summary", {})
            if nan_summary:
                st.markdown("**Feature-engineering null check**")
                st.dataframe(
                    pd.DataFrame(
                        {"column": list(nan_summary.keys()), "null_count": list(nan_summary.values())}
                    ),
                    use_container_width=True,
                )

            if model_service.classification_report_text:
                st.markdown("**Classification Report**")
                st.code(model_service.classification_report_text)

            try:
                import matplotlib.pyplot as plt
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
                st.pyplot(fig)
                plt.close(fig)
            except Exception as shap_error:
                st.info(f"SHAP chart unavailable for the current saved model: {shap_error}")

    except Exception as exc:
        logger.exception("Dashboard failed")
        st.error(str(exc))

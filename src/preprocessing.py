from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REQUIRED_TRANSACTION_COLUMNS = [
    "customer_id",
    "invoicedate",
    "invoice",
    "price",
    "quantity",
]

MODEL_FEATURE_COLUMNS = [
    "frequency_log",
    "monetary_log",
    "tenure",
    "avg_order_value",
    "unique_items_purchased",
    "purchase_rate",
    "monetary_per_day",
    "final_kmeans_cluster",
]

CLUSTER_FEATURE_COLUMNS = [
    "frequency_log",
    "monetary_log",
    "tenure",
    "purchase_rate",
]


def normalize_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()
    cleaned.rename(
        columns={
            "customer id": "customer_id",
            "invoicedate": "invoicedate",
            "stockcode": "stockcode",
            "invoice": "invoice",
            "price": "price",
            "quantity": "quantity",
        },
        inplace=True,
    )
    return cleaned


def validate_transaction_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_TRANSACTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def build_customer_features(
    df: pd.DataFrame,
    kmeans_model: Optional[KMeans] = None,
    n_clusters: int = 3,
    cluster_options: Iterable[int] = (3, 4, 5),
    reference_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, KMeans]:
    customer_df = normalize_transaction_columns(df)
    validate_transaction_columns(customer_df)

    # Remove rows that cannot support reliable customer-level aggregation.
    customer_df = customer_df.dropna(
        subset=["customer_id", "invoice", "price", "quantity"]
    ).copy()

    customer_df["total_price"] = customer_df["price"] * customer_df["quantity"]
    customer_df["invoicedate"] = pd.to_datetime(customer_df["invoicedate"], errors="coerce")
    customer_df = customer_df.dropna(subset=["invoicedate"]).sort_values(
        ["customer_id", "invoicedate"]
    )

    if reference_date is None:
        reference_date = customer_df["invoicedate"].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    rfm = customer_df.groupby("customer_id").agg(
        invoicedate=("invoicedate", lambda x: (reference_date - x.max()).days),
        invoice=("invoice", "nunique"),
        total_price=("total_price", "sum"),
    ).reset_index()
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

    tenure = customer_df.groupby("customer_id")["invoicedate"].agg(
        min_date="min",
        max_date="max",
    ).reset_index()
    tenure["tenure"] = (tenure["max_date"] - tenure["min_date"]).dt.days
    rfm = rfm.merge(tenure[["customer_id", "tenure"]], on="customer_id", how="left")

    items = customer_df.groupby("customer_id")["stockcode"].nunique().reset_index()
    items.rename(columns={"stockcode": "unique_items_purchased"}, inplace=True)
    rfm = rfm.merge(items, on="customer_id", how="left")

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].replace(0, 1)
    rfm["purchase_rate"] = rfm["frequency"] / (rfm["tenure"] + 1)
    rfm["monetary_per_day"] = rfm["monetary"] / (rfm["tenure"] + 1)

    for col in ["recency", "frequency", "monetary"]:
        rfm[f"{col}_log"] = np.log1p(rfm[col])

    # A customer is treated as churned when they have been inactive for more than 90 days.
    rfm["retention_status"] = (rfm["recency"] > 90).astype(int)

    # Expose feature-level null counts for debugging data quality issues upstream.
    rfm.attrs["nan_summary"] = rfm.isna().sum()

    # Impute clustering inputs instead of forcing zeros so segment structure stays realistic.
    cluster_imputer = SimpleImputer(strategy="median")
    rfm[CLUSTER_FEATURE_COLUMNS] = cluster_imputer.fit_transform(
        rfm[CLUSTER_FEATURE_COLUMNS]
    )

    # Keep the remaining engineered numeric features safe for downstream modeling.
    numeric_columns = rfm.select_dtypes(include=[np.number]).columns
    if rfm[numeric_columns].isna().any().any():
        numeric_imputer = SimpleImputer(strategy="median")
        rfm[numeric_columns] = numeric_imputer.fit_transform(rfm[numeric_columns])

    cluster_matrix = rfm[CLUSTER_FEATURE_COLUMNS]
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_matrix)

    if kmeans_model is None:
        candidate_scores = []
        best_model = None
        best_labels = None
        best_score = -1.0

        for candidate_k in cluster_options:
            model = KMeans(n_clusters=candidate_k, random_state=42, n_init=20)
            labels = model.fit_predict(cluster_scaled)
            score = silhouette_score(cluster_scaled, labels)
            candidate_scores.append(
                {
                    "k": int(candidate_k),
                    "silhouette_score": float(score),
                }
            )
            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels

        kmeans_model = best_model
        kmeans_model.selection_summary_ = candidate_scores
        kmeans_model.selected_k_ = int(kmeans_model.n_clusters)
        rfm["final_kmeans_cluster"] = best_labels
    else:
        rfm["final_kmeans_cluster"] = kmeans_model.predict(cluster_scaled)

    rfm.drop(columns=["recency", "recency_log"], inplace=True)
    return rfm, kmeans_model


def prepare_model_matrix(
    customer_df: pd.DataFrame,
    training_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    model_df = customer_df[MODEL_FEATURE_COLUMNS].copy()
    model_df = pd.get_dummies(
        model_df,
        columns=["final_kmeans_cluster"],
        prefix="cluster",
        drop_first=True,
    )
    if training_columns is not None:
        model_df = model_df.reindex(columns=training_columns, fill_value=0)
    return model_df

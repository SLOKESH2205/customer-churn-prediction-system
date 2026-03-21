import os
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.preprocessing import CLUSTER_FEATURE_COLUMNS


class SegmentClusterer:
    """Reusable clustering service for training and benchmarking customer segments."""

    def __init__(self, model_path: str = os.path.join("artifacts", "kmeans.pkl")):
        self.model_path = model_path

    def fit(self, cluster_frame: pd.DataFrame, n_clusters: int = 3) -> Tuple[KMeans, pd.Series, float]:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(cluster_frame[CLUSTER_FEATURE_COLUMNS])
        score = silhouette_score(cluster_frame[CLUSTER_FEATURE_COLUMNS], labels)
        return model, pd.Series(labels, index=cluster_frame.index), float(score)

    def fit_best_k(
        self,
        cluster_frame: pd.DataFrame,
        cluster_options: Iterable[int] = (3, 4, 5),
    ) -> Tuple[KMeans, pd.Series, pd.DataFrame]:
        X = cluster_frame[CLUSTER_FEATURE_COLUMNS]
        benchmarks = []
        best_model = None
        best_labels = None
        best_score = -1.0

        for n_clusters in cluster_options:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            share = pd.Series(labels).value_counts(normalize=True).max()
            benchmarks.append(
                {
                    "k": int(n_clusters),
                    "silhouette_score": float(score),
                    "largest_segment_share_pct": float(share * 100),
                }
            )
            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels

        benchmark_df = pd.DataFrame(benchmarks).sort_values(
            "silhouette_score", ascending=False
        )
        best_model.selection_summary_ = benchmark_df.to_dict(orient="records")
        best_model.selected_k_ = int(best_model.n_clusters)
        return best_model, pd.Series(best_labels, index=cluster_frame.index), benchmark_df

    def benchmark(self, cluster_frame: pd.DataFrame, n_clusters: int = 3) -> Dict[str, float]:
        X = cluster_frame[CLUSTER_FEATURE_COLUMNS]
        kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(X)
        gmm_labels = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10).fit_predict(X)
        mini_labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            batch_size=256,
        ).fit_predict(X)
        return {
            "KMeans": float(silhouette_score(X, kmeans_labels)),
            "GaussianMixture": float(silhouette_score(X, gmm_labels)),
            "MiniBatchKMeans": float(silhouette_score(X, mini_labels)),
        }

    def save(self, model: KMeans) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

    def load(self) -> KMeans:
        return joblib.load(self.model_path)

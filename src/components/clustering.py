import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.logger import logging
from src.exception import CustomException


class CustomerClustering:

    def __init__(self):
        self.model_features = [
            "recency_log",
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "unique_items_purchased"
        ]

        self.kmeans_model_path = os.path.join("artifacts", "kmeans.pkl")

    # ================= TRAIN ================= #
    def fit_clustering(self, df, X_scaled):
        try:
            print("\nRUNNING KMEANS TRAINING")

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            df["cluster"] = labels

            print("\nTrain cluster distribution:")
            print(df["cluster"].value_counts())

            # Save model (IMPORTANT: save original model without modification)
            os.makedirs("artifacts", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(kmeans, self.kmeans_model_path)

            print(f"\nModel saved at: {self.kmeans_model_path}")

            # Score
            score = silhouette_score(X_scaled, labels)
            print(f"KMeans Silhouette Score: {score:.4f}")

            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_scaled)
            plt.figure(figsize=(7, 5))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", s=20)
            plt.title("Customer Segmentation (PCA)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(os.path.join("outputs", "cluster_plot.png"))
            plt.close()

            return df, score

        except Exception as e:
            raise CustomException(e, sys)

    # ================= MODEL COMPARISON ================= #
    def compare_models(self, X_scaled):
        try:
            print("\nCOMPARING MODELS")

            # KMeans
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            k_labels = kmeans.fit_predict(X_scaled)
            k_score = silhouette_score(X_scaled, k_labels)

            # GMM
            gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
            g_labels = gmm.fit_predict(X_scaled)
            g_score = silhouette_score(X_scaled, g_labels)

            # MiniBatch
            mb = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=10, batch_size=256)
            mb_labels = mb.fit_predict(X_scaled)
            mb_score = silhouette_score(X_scaled, mb_labels)

            print("\nMODEL SCORES:")
            print(f"KMeans: {k_score:.4f}")
            print(f"GMM: {g_score:.4f}")
            print(f"MiniBatch: {mb_score:.4f}")

            return k_score, g_score, mb_score

        except Exception as e:
            raise CustomException(e, sys)

    # ================= TEST PREDICTION ================= #
    def predict_clusters(self, X_scaled):
        try:
            print("\nPREDICTING CLUSTERS ON TEST DATA")

            if not os.path.exists(self.kmeans_model_path):
                raise Exception("KMeans model not found! Train first.")

            kmeans = joblib.load(self.kmeans_model_path)

            labels = kmeans.predict(X_scaled)

            print("Test cluster prediction completed")

            return labels

        except Exception as e:
            raise CustomException(e, sys)

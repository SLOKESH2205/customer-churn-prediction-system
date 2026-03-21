from src.components.clustering import CustomerClustering
from src.components.feature_engineering import build_features
from src.preprocessing import CLUSTER_FEATURE_COLUMNS
import pandas as pd
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    print("RUNNING FULL CLUSTERING PIPELINE")

    train_df = pd.read_csv("artifacts/train.csv")
    test_df = pd.read_csv("artifacts/test.csv")

    train_df, _ = build_features(train_df)
    test_df, _ = build_features(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[CLUSTER_FEATURE_COLUMNS].fillna(0))
    X_test = scaler.transform(test_df[CLUSTER_FEATURE_COLUMNS].fillna(0))

    clustering = CustomerClustering()

    train_df, _ = clustering.fit_clustering(train_df, X_train)
    clustering.compare_models(X_train)

    test_clusters = clustering.predict_clusters(X_test)
    test_df["cluster"] = test_clusters

    print("\nTRAIN CLUSTER DISTRIBUTION:")
    print(train_df["cluster"].value_counts())

    print("\nTEST CLUSTER DISTRIBUTION:")
    print(test_df["cluster"].value_counts())

    print("\nPIPELINE COMPLETED SUCCESSFULLY")

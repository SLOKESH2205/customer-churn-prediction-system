from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import build_features
from src.components.clustering import CustomerClustering
import pandas as pd

if __name__ == "__main__":

    print("RUNNING FULL CLUSTERING PIPELINE")

    # Step 1: Transformation
    transformer = DataTransformation()

    X_train, X_test, _, _, _ = transformer.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )

    # Step 2: Feature Engineering
    train_df = pd.read_csv("artifacts/train.csv")
    test_df = pd.read_csv("artifacts/test.csv")

    train_df, _ = build_features(train_df)
    test_df, _ = build_features(test_df)

    # Step 3: Clustering
    clustering = CustomerClustering()

    # TRAIN
    train_df, _ = clustering.fit_clustering(train_df, X_train)

    # MODEL COMPARISON
    clustering.compare_models(X_train)

    # TEST
    test_clusters = clustering.predict_clusters(X_test)
    test_df["cluster"] = test_clusters

    print("\nTRAIN CLUSTER DISTRIBUTION:")
    print(train_df["cluster"].value_counts())

    print("\nTEST CLUSTER DISTRIBUTION:")
    print(test_df["cluster"].value_counts())

    print("\nPIPELINE COMPLETED SUCCESSFULLY")

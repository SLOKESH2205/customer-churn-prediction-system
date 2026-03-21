import pandas as pd

from src.modeling import ChurnModelService
from src.preprocessing import build_customer_features


class PredictPipeline:
    """Train-on-demand pipeline for a single uploaded dataset."""

    def __init__(self, df: pd.DataFrame):
        self.customer_df, self.kmeans = build_customer_features(df)
        self.service = ChurnModelService.train_from_customer_df(self.customer_df)
        self.model = self.service.model
        self.threshold = self.service.threshold
        self.scaler = self.service.scaler
        self.features = self.service.features

    def preprocess(self, df: pd.DataFrame):
        customer_df, _ = build_customer_features(df)
        self.last_processed_df = customer_df.copy()
        return customer_df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        customer_df, _ = build_customer_features(df)
        result_df = self.service.score_customer_df(customer_df)
        self.last_processed_df = self.service.last_processed_df
        self.features = self.service.features
        self.customer_df = customer_df
        return result_df

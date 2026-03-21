import pandas as pd

from src.modeling import ChurnModelService


class PredictPipeline:
    """Thin adapter kept for compatibility with the existing app and tests."""

    def __init__(self):
        self.service = ChurnModelService()
        self.model = self.service.model
        self.threshold = self.service.threshold
        self.kmeans = self.service.kmeans
        self.scaler = self.service.scaler
        self.train_columns = self.service.train_columns

    def preprocess(self, df: pd.DataFrame):
        result_df = self.service.predict_customers(df)
        self.last_processed_df = self.service.last_processed_df
        self.features = self.service.features
        self.last_result_df = result_df
        return self.last_processed_df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = self.service.predict_customers(df)
        self.last_processed_df = self.service.last_processed_df
        self.features = self.service.features
        self.last_result_df = result_df
        return result_df

import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.preprocessing import build_customer_features


def build_features(df: pd.DataFrame, kmeans_model=None):
    """Backward-compatible wrapper around the shared preprocessing module."""
    try:
        logging.info("Starting feature engineering")
        engineered_df, fitted_kmeans = build_customer_features(df, kmeans_model)
        logging.info("Feature engineering + clustering completed")
        return engineered_df, fitted_kmeans
    except Exception as e:
        raise CustomException(e, sys)

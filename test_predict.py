import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":

    df = pd.read_csv("artifacts/test.csv")

    pipe = PredictPipeline(df)

    results = pipe.predict(df)

    print(results.head(10).to_string())

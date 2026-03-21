import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

import logging
from src.logger import setup_logger
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self, file_path: str, split_type: str = "time"):
        try:
            logging.info("Starting data ingestion")

            # Check file existence
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")

            # Load dataset (Excel or CSV)
            if file_path.endswith(".xlsx"):
                logging.info("Reading Excel file with multiple sheets")

                df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
                df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

                df = pd.concat([df1, df2], ignore_index=True)

                logging.info(f"Excel sheets merged. Shape: {df.shape}")

            else:
                logging.info("Reading CSV file")
                df = pd.read_csv(file_path)

            logging.info("Dataset loaded successfully")

            # Standardize column names
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
            )

            # Required columns check
            required_columns = ["invoicedate", "customer_id", "invoice", "quantity", "price"]
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                raise Exception(f"Missing columns in dataset: {missing_cols}")

            # Convert invoice date
            df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
            df = df.dropna(subset=["invoicedate"])

            # Drop missing customer_id
            df = df.dropna(subset=["customer_id"])
            df["customer_id"] = df["customer_id"].astype(int)

            # Remove cancelled invoices
            df = df[~df["invoice"].astype(str).str.startswith("C")]

            # Remove invalid rows
            df = df[(df["quantity"] > 0) & (df["price"] > 0)]

            # Create total price feature
            df["total_price"] = df["quantity"] * df["price"]

            logging.info(f"Final dataset shape after cleaning: {df.shape}")

            # Create artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # Save cleaned raw data
            df.to_csv(self.raw_data_path, index=False)

            # ================= SPLIT LOGIC ================= #
            logging.info(f"Using split type: {split_type}")

            if split_type == "random":
                train_set, test_set = train_test_split(
                    df, test_size=0.2, random_state=42
                )

            elif split_type == "time":
                df = df.sort_values("invoicedate")

                split_index = int(0.8 * len(df))

                train_set = df.iloc[:split_index]
                test_set = df.iloc[split_index:]

            else:
                raise Exception("Invalid split type. Choose 'random' or 'time'.")

            # ================================================= #

            # Save splits
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info(f"Train shape: {train_set.shape}")
            logging.info(f"Test shape: {test_set.shape}")
            logging.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)


# Entry point
if __name__ == "__main__":
    try:
        setup_logger()

        print("MAIN BLOCK RUNNING")

        ingestion = DataIngestion()

        file_path = os.path.join("data", "raw", "online_retail_II.xlsx")

        train_path, test_path = ingestion.initiate_data_ingestion(
            file_path=file_path,
            split_type="time"   # 🔥 change to "random" if needed
        )

        print("\nData ingestion completed successfully")
        print(f"Train file: {train_path}")
        print(f"Test file: {test_path}")

    except Exception as e:
        print(f"Failed: {e}")
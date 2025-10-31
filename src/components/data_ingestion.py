import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train_data.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test_data.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            logging.info("Reading the source data file")
            data = pd.read_csv(os.path.join("data-source", "adult.csv"))
            logging.info("Source data read successfully.")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )

            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            logging.info("Data split into train and test sets.")

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    logging.info("Starting data ingestion process")
    train_data, test_data = obj.initiate_data_ingestion()
    logging.info("Data ingestion process completed")
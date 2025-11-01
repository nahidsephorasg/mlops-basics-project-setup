from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from src.exception import CustomException
import logging


if __name__ == "__main__":
    obj = DataIngestion()
    logging.info("Starting data ingestion process")
    train_data, test_data = obj.initiate_data_ingestion()
    logging.info("Data ingestion process completed")

    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

    logging.info("Data transformation process completed")

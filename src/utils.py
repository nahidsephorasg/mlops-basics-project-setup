from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}")
        raise CustomException(e, sys)

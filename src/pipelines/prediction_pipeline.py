import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import load_object


@dataclass
class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading preprocessor and model for prediction")
            preprocessor_path = os.path.join(
                "artifacts/data_transformation", "preprocessor.pkl"
            )
            model_path = os.path.join(
                "artifacts/model_trainer", "model.pkl"
            )

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Transforming features for prediction")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.error("Error occurred during prediction.")
            raise CustomException(e, sys)


class CustomClass:
    def __init__(
        self,
        age,
        workclass,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        sex,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country,
    ):
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education.num": [self.education_num],
                "marital.status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "capital.gain": [self.capital_gain],
                "capital.loss": [self.capital_loss],
                "hours.per.week": [self.hours_per_week],
                "native.country": [self.native_country],
            }
            data = pd.DataFrame(custom_data_input_dict)

            return data
        except Exception as e:
            logging.error("Error occurred while converting input data to DataFrame.")
            raise CustomException(e, sys)

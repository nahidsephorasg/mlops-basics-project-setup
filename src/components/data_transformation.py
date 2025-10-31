import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feast import Field, Feature, FeatureView, FileSource, Entity, FeatureStore
from feast.types import Float32, Int64, String
from feast.value_type import ValueType
from datetime import timedelta, datetime
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_object_path = os.path.join(
        "artifacts/data_transformation", "preprocessor.pkl"
    )
    feature_store_repo_path = "feature_repo"


class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            repo_path = os.path.abspath(
                self.data_transformation_config.feature_store_repo_path
            )
            os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)

            feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
            feature_store_yaml = f"""project: mlops_project
registry: {os.path.join(repo_path, "registry.db")}
provider: local
online_store:
    type: sqlite
offline_store:
    type: file
entity_key_serialization_version: 3
"""

            with open(feature_store_yaml_path, "w") as f:
                f.write(feature_store_yaml)
            logging.info(f"Feature store YAML created at {feature_store_yaml_path}")

            with open(feature_store_yaml_path, "r") as f:
                content = f.read()
                logging.info(f"Feature store YAML content:\n{content}")

            self.feature_store = FeatureStore(repo_path=repo_path)
            logging.info(f"Feature store initialized at {repo_path}")

        except Exception as e:
            logging.error("Error initializing DataTransformation class: {str(e)}")
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object.")
            numerical_features = [
                "age",
                "education.num",
                "capital.gain",
                "capital.loss",
                "hours.per.week",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_features),
                ]
            )
            logging.info("Data transformer object created successfully.")
            return preprocessor
        except Exception as e:
            logging.error("Error creating data transformer object.")
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR

            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit
            return df

        except Exception as e:
            logging.error(
                f"Outierns handing processes are not working properly {str(e)}"
            )
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "income"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing datasets."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Starting feature store operations")

            self.push_features_to_store(train_df, "train")
            logging.info("Train features pushed to feature store successfully.")

            self.push_features_to_store(test_df, "test")
            logging.info("Test features pushed to feature store successfully.")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_object_path,
                obj=preprocessing_obj,
            )
            logging.info("Preprocessor object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_path,
            )

        except Exception as e:
            logging.error(f"Error during data transformation process {str(e)}")
            raise CustomException(e, sys)

    def push_features_to_store(self, df, entity_id):
        try:
            if "event_timestamp" not in df.columns:
                df["event_timestamp"] = pd.Timestamp.now()

            if "entity_id" not in df.columns:
                df["entity_id"] = range(len(df))

            data_path = os.path.join(
                self.data_transformation_config.feature_store_repo_path, "data"
            )
            parquet_file_path = os.path.join(data_path, f"{entity_id}_features.parquet")
            os.makedirs(data_path, exist_ok=True)
            df.to_parquet(parquet_file_path, index=False)
            logging.info(f"Data saved to {parquet_file_path}")

            # Use relative path from the feature store repo
            file_source = FileSource(
                path=f"data/{entity_id}_features.parquet",
                timestamp_field="event_timestamp",
            )

            entity = Entity(
                name="entity_id",
                value_type=ValueType.INT64,
                description="Entity ID",
            )

            # Apply entity first
            self.feature_store.apply([entity])
            
            features_view = FeatureView(
                name=f"{entity_id}_features",
                entities=[entity],
                schema=[
                    Field(name="age", dtype=Int64),
                    Field(name="workclass", dtype=String),
                    Field(name="education.num", dtype=Int64),
                    Field(name="marital.status", dtype=String),
                    Field(name="occupation", dtype=String),
                    Field(name="relationship", dtype=String),
                    Field(name="race", dtype=String),
                    Field(name="sex", dtype=String),
                    Field(name="capital.gain", dtype=Int64),
                    Field(name="capital.loss", dtype=Int64),
                    Field(name="hours.per.week", dtype=Int64),
                    Field(name="native.country", dtype=String),
                ],
                source=file_source,
                online=True,
                tags={
                    "created_by": "DataTransformationModule",
                    "entity_id": entity_id,
                    "team": "MLops",
                },
                ttl=timedelta(days=365),
            )
            self.feature_store.apply([features_view])
            logging.info(
                f"Features for {entity_id} pushed to feature store successfully."
            )

            self.feature_store.materialize(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now() + timedelta(days=1),
            )
            logging.info(f"Feature store materialization completed for {entity_id}.")

        except Exception as e:
            logging.error(f"Error pushing features to store for {entity_id}: {str(e)}")
            raise CustomException(e, sys)

    def retrieve_features_from_store(self, entity_id, num_entities):
        try:
            logging.info(f"Retrieving features for {entity_id} from feature store.")
            feature_service_name = f"{entity_id}_features"
            feature_vector = self.feature_store.get_online_features(
                feature_refs=[
                    f"{entity_id}_features:age",
                    f"{entity_id}_features:workclass",
                    f"{entity_id}_features:education.num",
                    f"{entity_id}_features:marital.status",
                    f"{entity_id}_features:occupation",
                    f"{entity_id}_features:relationship",
                    f"{entity_id}_features:race",
                    f"{entity_id}_features:sex",
                    f"{entity_id}_features:capital.gain",
                    f"{entity_id}_features:capital.loss",
                    f"{entity_id}_features:hours.per.week",
                    f"{entity_id}_features:native.country",
                ],
                entity_rows=[{"entity_id": i} for i in range(num_entities)],
            ).to_df()
            logging.info(
                f"Features for {entity_id} retrieved successfully from feature store."
            )
            return feature_vector
        except Exception as e:
            logging.error(
                f"Error retrieving features from store for {entity_id}: {str(e)}"
            )
            raise CustomException(e, sys)

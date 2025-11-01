import os, sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from src.utils import save_object
import matplotlib.pyplot as plt


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts/model_trainer", "model.pkl")
    mlflow_experiment_name = "Model_Training_Experiment"
    mlflow_tracking_uri = "./mlruns"  # Use local file-based tracking
    mlflow_run_name = f"Model_Training_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.model_trainer_config.mlflow_experiment_name)
        
        self.run_name = f"Model_Training_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_model_metrics(self, y_true, y_pred, prefix=""):
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, average="weighted"),
            f"{prefix}recall": recall_score(y_true, y_pred, average="weighted"),
            f"{prefix}f1_score": f1_score(y_true, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)
        return metrics

    def train_model(self, X_train, y_train, X_test, y_test, model_name, model, params):
        try:
            with mlflow.start_run(run_name=f"{model_name}_{self.run_name}") as run:
                logging.info(f"Started Training {model_name}...")

                mlflow.log_params(params)

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=5,
                    n_jobs=-1,
                    verbose=2,
                    scoring="accuracy",
                )

                grid_search.fit(X_train, y_train)

                best_params = {
                    f"best_{k}": v for k, v in grid_search.best_params_.items()
                }
                mlflow.log_params(best_params)
                logging.info(f"Best parameters for {model_name}: {best_params}")

                y_train_pred = grid_search.predict(X_train)
                y_test_pred = grid_search.predict(X_test)

                train_metrics = self.log_model_metrics(
                    y_train, y_train_pred, prefix="train_"
                )
                test_metrics = self.log_model_metrics(
                    y_test, y_test_pred, prefix="test_"
                )

                mlflow.log_metric("cv_mean_score", grid_search.best_score_)
                mlflow.log_metric(
                    "cv_std_score",
                    np.std(
                        grid_search.cv_results_["std_test_score"][
                            grid_search.best_index_
                        ]
                    ),
                )

                if hasattr(grid_search.best_estimator_, "feature_importances_"):
                    feature_importances = pd.DataFrame(
                        {
                            "feature": [
                                f"feature_{i}" for i in range(X_train.shape[1])
                            ],
                            "importance": grid_search.best_estimator_.feature_importances_,
                        }
                    )

                    plt.figure(figsize=(10, 6))
                    plt.barh(
                        feature_importances["feature"], feature_importances["importance"]
                    )
                    plt.xticks(rotation=45)
                    plt.title(f"Feature Importances for {model_name}")
                    plt.tight_layout()

                    plot_path = (
                        f"artifacts/model_trainer/{model_name}_feature_importances.png"
                    )
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    plt.close()
                    os.remove(plot_path)

                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    f"{model_name}_model",
                    registered_model_name=model_name,
                )

            logging.info(f"Completed Training {model_name}.")
            return grid_search.best_estimator_, test_metrics["test_accuracy"]

        except Exception as e:
            logging.error(f"Error occurred while training {model_name}.")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer started")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        "n_estimators": [20, 50, 30],
                        "max_depth": [10, 8, 5],
                        "min_samples_split": [2, 5, 10],
                    },
                },
                "DecisionTree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        "criterion": ["gini", "entropy", "log_loss"],
                        "max_depth": [3, 4, 5, 6],
                        "min_samples_split": [2, 3, 4, 5],
                    },
                },
                "LogisticRegression": {
                    "model": LogisticRegression(),
                    "params": {
                        "class_weight": ["balanced"],
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "solver": ["liblinear", "saga"],
                    },
                },
            }

            model_results = {}

            for model_name, config in models.items():
                logging.info(f"Training model: {model_name}")
                model, accuracy = self.train_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_name,
                    config["model"],
                    config["params"],
                )

                model_results[model_name] = {
                    "model": model,
                    "accuracy": accuracy,
                }

            best_model_name = max(
                model_results, key=lambda name: model_results[name]["accuracy"]
            )

            best_model_info = model_results[best_model_name]
            best_model = best_model_info["model"]
            best_accuracy = best_model_info["accuracy"]

            logging.info(
                f"Best model found: {best_model_name} with accuracy: {best_accuracy}"
            )

            with mlflow.start_run(run_name=f"Best_Model_Summary{self.run_name}"):
                mlflow.sklearn.log_model(
                    best_model,
                    "best_model",
                    registered_model_name="Best_Classification_Model",
                )

                comparison_metrics = {
                    f"{name}_accuracy": info["accuracy"]
                    for name, info in model_results.items()
                }
                mlflow.log_metrics(comparison_metrics)

                plt.figure(figsize=(10, 6))
                plt.bar(comparison_metrics.keys(), comparison_metrics.values())

                plt.xticks(rotation=45)
                plt.title("Model Comparison based on Accuracy")
                plt.tight_layout()
                plt.savefig("artifacts/model_trainer/model_comparison.png")
                mlflow.log_artifact("artifacts/model_trainer/model_comparison.png")
                plt.close()
                os.remove("artifacts/model_trainer/model_comparison.png")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True,
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )
            logging.info(
                f"Trained model saved at {self.model_trainer_config.trained_model_path}"
            )
            return best_accuracy
        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)


def main():
    try:
        # Import here to avoid circular import
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        
        with mlflow.start_run(run_name="Complete_Model_Training_Pipeline"):
            logging.info("Starting complete model training pipeline...")

            data_ingestion = DataIngestion()
            train_data_path, test_data_path = (
                data_ingestion.initiate_data_ingestion()
            )
            mlflow.log_param("train_data_path", train_data_path)
            mlflow.log_param("test_data_path", test_data_path)

            logging.info("Starting data transformation...")
            data_transformation = DataTransformation()
            train_array, test_array, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_data_path, test_data_path
                )
            )
            mlflow.log_param("preprocessor_path", preprocessor_path)
            logging.info("Data transformation completed.")

            logging.info("Starting model training...")
            model_trainer = ModelTrainer()
            accuracy = model_trainer.initiate_model_trainer(train_array, test_array)
            logging.info(
                f"Model training completed with best model accuracy: {accuracy}"
            )
            return accuracy
    except Exception as e:
        logging.error("Error occurred in the complete model training pipeline.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

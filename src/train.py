from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import (
    build_preprocessor,
    load_data,
    split_features_target,
)


CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    """Load project configuration from YAML."""

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config


def create_model(model_config):
    """Create a regression model from its configuration."""

    model_type = model_config["type"]

    if model_type == "LinearRegression":
        return LinearRegression()

    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"],
        )

    if model_type == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=model_config["n_estimators"],
            learning_rate=model_config["learning_rate"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"],
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained regression model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def main():
    config = load_config()

    # Configure MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load data
    wine = load_data(
        config["data"]["red_path"],
        config["data"]["white_path"],
    )

    X, y = split_features_target(wine)

    # Create held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print()

    for model_name, model_config in config["models"].items():
        print(f"Training: {model_name}")

        preprocessor = build_preprocessor(X_train)
        regressor = create_model(model_config)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", regressor),
            ]
        )

        with mlflow.start_run(run_name=model_name):

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test,
            )

            # Log model configuration
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_config["type"])

            for parameter_name, parameter_value in model_config.items():
                if parameter_name != "type":
                    mlflow.log_param(
                        parameter_name,
                        parameter_value,
                    )

            # Log data information
            mlflow.log_param("dataset", "UCI Wine Quality")
            mlflow.log_param("total_samples", len(wine))
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param(
                "test_size",
                config["data"]["test_size"],
            )

            # Log evaluation metrics
            mlflow.log_metrics(metrics)

            # Save trained pipeline as an MLflow artifact
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            print(f"MAE:  {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R2:   {metrics['r2']:.4f}")
            print("-" * 40)


if __name__ == "__main__":
    main()
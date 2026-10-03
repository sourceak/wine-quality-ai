from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml


CONFIG_PATH = Path("configs/config.yaml")
MODEL_PATH = Path("models/best_model.pkl")


def load_config():
    """Load project configuration."""

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Connect to MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment = mlflow.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )

    if experiment is None:
        raise ValueError("MLflow experiment not found.")

    # Find all successful runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )

    if runs.empty:
        raise ValueError("No successful MLflow runs found.")

    # Keep runs that contain all three evaluation metrics
    runs = runs.dropna(
        subset=[
            "metrics.mae",
            "metrics.rmse",
            "metrics.r2",
        ]
    )

    if runs.empty:
        raise ValueError("No runs with evaluation metrics found.")

    # Lower RMSE is better
    ranked_runs = runs.sort_values(
        by="metrics.rmse",
        ascending=True,
    )

    columns = [
        "tags.mlflow.runName",
        "metrics.mae",
        "metrics.rmse",
        "metrics.r2",
        "run_id",
    ]

    print("MODEL COMPARISON")
    print("=" * 80)
    print(ranked_runs[columns].to_string(index=False))

    # Select best run
    best_run = ranked_runs.iloc[0]
    best_run_id = best_run["run_id"]

    print("\nBEST RUN")
    print("=" * 80)
    print("Model:", best_run["tags.mlflow.runName"])
    print(f"MAE:  {best_run['metrics.mae']:.4f}")
    print(f"RMSE: {best_run['metrics.rmse']:.4f}")
    print(f"R2:   {best_run['metrics.r2']:.4f}")
    print("Run ID:", best_run_id)

    # Load the winning pipeline from MLflow
    model_uri = f"runs:/{best_run_id}/model"

    print("\nLoading best model from MLflow...")

    best_model = mlflow.sklearn.load_model(model_uri)

    # Save a local copy for the application
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    print("\nMODEL SAVED")
    print("=" * 80)
    print("Path:", MODEL_PATH)


if __name__ == "__main__":
    main()
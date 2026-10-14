
from pathlib import Path
import json

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_numeric_metrics(data, prefix=""):
    """
    Flatten nested dictionaries and retain numeric metrics.
    Excludes booleans, strings, and null values.
    """
    flattened = {}

    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(
                flatten_numeric_metrics(value, name)
            )
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            flattened[name[:250]] = float(value)

    return flattened


def log_model_run(
    experiment_name,
    run_name,
    metrics_files,
    artifacts,
    params,
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        for metrics_file in metrics_files:
            path = REPORT_DIR / metrics_file

            if not path.exists():
                print(f"Skipping missing metrics: {path}")
                continue

            data = read_json(path)
            metrics = flatten_numeric_metrics(data)

            if metrics:
                mlflow.log_metrics(metrics)

            # Preserve the exact JSON used for the run.
            mlflow.log_artifact(str(path), artifact_path="metrics_json")

        for relative_path in artifacts:
            path = PROJECT_ROOT / relative_path

            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="outputs")

        print(f"Logged MLflow run: {run_name}")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    common_params = {
        "category": "bottle",
        "image_size": 256,
        "seed": 42,
    }

    log_model_run(
        experiment_name="industrial-anomaly-detection",
        run_name="cnn-autoencoder",
        metrics_files=[
            "test_metrics.json",
            "localization_metrics.json",
        ],
        artifacts=[
            "artifacts/autoencoder_best.pth",
            "reports/training_loss.png",
            "reports/test_predictions.png",
            "reports/localization_results.png",
        ],
        params={
            **common_params,
            "model": "cnn_autoencoder",
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
    )

    log_model_run(
        experiment_name="industrial-anomaly-detection",
        run_name="patchcore-style",
        metrics_files=["patchcore_metrics.json"],
        artifacts=[
            "artifacts/patchcore_memory_bank.pth",
            "reports/patchcore_localization.png",
        ],
        params={
            **common_params,
            "model": "patchcore_style",
            "backbone": "resnet18",
            "batch_size": 4,
            "patches_per_image": 64,
            "max_memory_bank_size": 5000,
        },
    )

    print("\nMLflow logging completed.")


if __name__ == "__main__":
    main()

"""Download the current model from the DagsHub MLflow Model Registry.

Runs once, during `docker build`, so the model is baked into the image and the
API never needs DagsHub credentials at runtime.

Configuration (environment variables, set as build args in the Dockerfile):
    MLFLOW_TRACKING_URI  DagsHub MLflow server of the repo
    DAGSHUB_USERNAME     DagsHub user owning the token
    MODEL_NAME           registered model name
    MODEL_ALIAS          alias pointing to the version to ship (moved after each retraining)
    MODEL_DIR            where to write the model inside the image
Credentials:
    the DagsHub token is read from the Docker build secret file
    /run/secrets/dagshub_token, or from DAGSHUB_TOKEN when run outside Docker.
"""

import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

SECRET_FILE = Path("/run/secrets/dagshub_token")


def read_token() -> str:
    if SECRET_FILE.exists():
        token = SECRET_FILE.read_text().strip()
    else:
        token = os.environ.get("DAGSHUB_TOKEN", "").strip()
    if not token:
        sys.exit(
            "ERROR: no DagsHub token. Pass it as a build secret "
            "(--secret id=dagshub_token,...) or set DAGSHUB_TOKEN."
        )
    return token


def main() -> None:
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "https://dagshub.com/Phaztox/dvc-lab_efrei.mlflow"
    )
    model_name = os.environ.get("MODEL_NAME", "RandomForest_Digits")
    model_alias = os.environ.get("MODEL_ALIAS", "champion")
    model_dir = Path(os.environ.get("MODEL_DIR", "/app/model"))

    # MLflow authenticates to DagsHub with HTTP basic auth: username + token.
    # These env vars only live in this process, they are not written to the image.
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", "Phaztox")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = read_token()
    mlflow.set_tracking_uri(tracking_uri)

    # Resolve the alias to a concrete version first, so the build log says exactly what is shipped.
    version = MlflowClient().get_model_version_by_alias(model_name, model_alias)
    print(
        f"Registry: {model_name}@{model_alias} -> version {version.version} "
        f"(run {version.run_id}, source {version.source})"
    )

    # Download that exact version (not the alias again: it could move in between).
    model_dir.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{version.version}", dst_path=str(model_dir)
    )

    # Traceability: the API reports which model version answered each request.
    info = {
        "model_name": model_name,
        "model_alias": model_alias,
        "model_version": version.version,
        "run_id": version.run_id,
        "source": version.source,
    }
    (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Model downloaded to {model_dir}: {sorted(p.name for p in model_dir.iterdir())}")


if __name__ == "__main__":
    main()

# syntax=docker/dockerfile:1
# Digits classifier API. The model is downloaded from the DagsHub MLflow registry at build time.
#
# Build (the DagsHub token is passed as a secret, it is NOT stored in the image):
#   DAGSHUB_TOKEN=<token> docker build --secret id=dagshub_token,env=DAGSHUB_TOKEN \
#       --build-arg MODEL_CACHE_BUST=$(date +%s) -t digits-api .
# Run:
#   docker run -p 8000:8000 digits-api

# Same Python minor version as the training environment (model python_env.yaml: 3.12.3)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# 1) Dependencies first: this layer is cached as long as requirements.txt doesn't change.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Fetch the model from DagsHub through MLflow.
ARG MLFLOW_TRACKING_URI=https://dagshub.com/Phaztox/dvc-lab_efrei.mlflow
ARG DAGSHUB_USERNAME=Phaztox
ARG MODEL_NAME=RandomForest_Digits
ARG MODEL_ALIAS=champion
# Changing this value invalidates the cache from here on, forcing a fresh model download.
# The CI passes a unique value on every run so each build ships the current champion model.
ARG MODEL_CACHE_BUST=none
COPY app/download_model.py app/download_model.py
RUN --mount=type=secret,id=dagshub_token,required=true \
    echo "cache bust: ${MODEL_CACHE_BUST}" && python app/download_model.py

# 3) Application code.
COPY app/main.py app/main.py

# Don't run the API as root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

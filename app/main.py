"""FastAPI app serving the digits classifier with a single endpoint: POST /predict.

The model is loaded from MODEL_DIR, where `download_model.py` put it during the
Docker build. No connection to DagsHub is needed at runtime.

Request:  {"inputs": [[64 pixel values], ...]}     (8x8 digit images, values 0-16)
Response: {"predictions": [digit, ...], "model_version": "<registry version>"}
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
N_FEATURES = 64  # load_digits: 8x8 images flattened

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not on every request.
    state["model"] = mlflow.pyfunc.load_model(str(MODEL_DIR))
    info_file = MODEL_DIR / "model_info.json"
    state["info"] = json.loads(info_file.read_text()) if info_file.exists() else {}
    print(f"Loaded model: {state['info'] or MODEL_DIR}")
    yield
    state.clear()


app = FastAPI(title="Digits classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    inputs: list[list[float]] = Field(
        ..., min_length=1, description=f"One row of {N_FEATURES} pixel values per image"
    )


class PredictResponse(BaseModel):
    predictions: list[int]
    model_version: str | None = None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    bad_rows = [i for i, row in enumerate(request.inputs) if len(row) != N_FEATURES]
    if bad_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Each input row must have {N_FEATURES} values; wrong length at rows {bad_rows}",
        )
    # float64 matches the model signature logged in MLflow (tensor float64, shape [-1, 64]).
    X = np.asarray(request.inputs, dtype=np.float64)
    predictions = state["model"].predict(X)
    return PredictResponse(
        predictions=[int(p) for p in predictions],
        model_version=state["info"].get("model_version"),
    )

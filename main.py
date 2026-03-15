from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from utils.model_utils import load_model, predict_text


class PredictionRequest(BaseModel):
    text: str
    source: Optional[str] = None


class PredictionResponse(BaseModel):
    label: str
    score: float
    details: dict


MODEL_PATH = "saved_models/phishing_model.joblib"

app = FastAPI(
    title="Phishing Detection API",
    version="0.1.0",
    description="A simple FastAPI service for phishing detection using a scikit-learn model.",
)


@app.on_event("startup")
def startup_event() -> None:
    """Load the model when the server starts."""

    try:
        app.state.model = load_model(MODEL_PATH)
    except Exception as exc:
        # If the model isn't available on startup, we keep the app running and fail at prediction time.
        app.state.model = None
        app.state.model_load_error = str(exc)


@app.get("/health")
def health() -> dict:
    """Health endpoint to confirm the service is running."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict whether text is phishing or legitimate."""

    model = getattr(app.state, "model", None)
    if model is None:
        error_detail = getattr(app.state, "model_load_error", "Model not loaded.")
        raise HTTPException(status_code=500, detail=error_detail)

    label, score = predict_text(model, request.text)
    return PredictionResponse(label=label, score=score, details={"source": request.source})

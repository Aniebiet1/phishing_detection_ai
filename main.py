import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from utils.app_state import (
    authenticate_user,
    create_user,
    get_public_user_summaries,
    get_user_by_token,
    get_user_history,
    record_prediction,
)
from utils.model_utils import load_model, predict_text_with_metadata
from utils.db import init_db


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "phishing_model.joblib"
REPORT_PATH = BASE_DIR / "saved_models" / "training_report.json"
FRONTEND_DIR = BASE_DIR / "Frontend"


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text or URL to classify. If it contains control characters, use /predict-raw instead.",
        min_length=1,
    )
    source: Optional[str] = Field(default=None, description="Optional hint such as email or url.")


class PredictionResponse(BaseModel):
    label: str
    score: float
    details: dict


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: dict


app = FastAPI(
    title="Phishing Detection Platform",
    version="1.0.0",
    description="A FastAPI service for phishing detection, model reporting, and user activity tracking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_loaded_model():
    """Return loaded model or raise a clear HTTP error if unavailable."""

    model = getattr(app.state, "model", None)
    if model is None and MODEL_PATH.exists():
        # Allow recovery without restarting the API if model file is created after startup.
        try:
            app.state.model = load_model(str(MODEL_PATH))
            app.state.model_load_error = None
            model = app.state.model
        except Exception as exc:
            app.state.model_load_error = str(exc)

    if model is None:
        error_detail = getattr(app.state, "model_load_error", "Model not loaded.")
        raise HTTPException(status_code=500, detail=error_detail)
    return model


def _load_training_report() -> dict:
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training report not found at {REPORT_PATH}. Run training first.",
        )

    return getattr(app.state, "training_report", {})


def _resolve_user(authorization: Optional[str]) -> Optional[dict]:
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header.")

    user = get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return user


def _predict_and_log(
    text: str,
    source: Optional[str],
    authorization: Optional[str],
    mode: str,
) -> PredictionResponse:
    model = _get_loaded_model()
    label, score, prediction_meta = predict_text_with_metadata(model, text=text, source=source)
    user = None
    if authorization:
        try:
            user = _resolve_user(authorization)
        except HTTPException as exc:
            # Classification should still work for unauthenticated users.
            # If the token is missing/expired/invalid, continue as guest.
            if exc.status_code != 401:
                raise
    is_authenticated = user is not None
    guest_id = None if is_authenticated else f"guest_{uuid.uuid4().hex[:12]}"

    usage_entry = record_prediction(
        email=user["email"] if is_authenticated else None,
        text=text,
        label=label,
        score=score,
        source=source or mode,
    )

    details = {
        "source": source or mode,
        "input_length": len(text),
        "mode": mode,
        "user_type": "authenticated" if is_authenticated else "guest",
        "guest_id": guest_id,
        "usage_entry": usage_entry,
        **prediction_meta,
    }
    if user is not None:
        details["user"] = user["email"]

    return PredictionResponse(label=label, score=score, details=details)


@app.on_event("startup")
def startup_event() -> None:
    """Load persisted assets on application startup."""

    init_db()

    try:
        app.state.model = load_model(str(MODEL_PATH))
        app.state.model_load_error = None
    except Exception as exc:
        app.state.model = None
        app.state.model_load_error = str(exc)

    if REPORT_PATH.exists():
        app.state.training_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        app.state.training_report = {}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "report_available": REPORT_PATH.exists(),
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: AuthRequest) -> AuthResponse:
    try:
        token, user = create_user(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest) -> AuthResponse:
    try:
        token, user = authenticate_user(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@app.get("/auth/me")
def get_me(authorization: Optional[str] = Header(default=None)) -> dict:
    user = _resolve_user(authorization)
    history = get_user_history(user["email"])
    return {
        "user": user,
        "history_count": len(history),
    }


@app.get("/users/me/history")
def user_history(authorization: Optional[str] = Header(default=None)) -> dict:
    user = _resolve_user(authorization)
    history = get_user_history(user["email"])
    phishing_count = sum(1 for item in history if item["label"] == "phishing")
    legitimate_count = len(history) - phishing_count
    return {
        "user": user,
        "history": history,
        "summary": {
            "total_predictions": len(history),
            "phishing_predictions": phishing_count,
            "legitimate_predictions": legitimate_count,
        },
    }


@app.get("/users/summary")
def users_summary() -> dict:
    return {"users": get_public_user_summaries()}


@app.get("/training-report")
def training_report() -> dict:
    return _load_training_report()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    authorization: Optional[str] = Header(default=None),
) -> PredictionResponse:
    return _predict_and_log(
        text=request.text,
        source=request.source,
        authorization=authorization,
        mode="json",
    )


@app.post("/predict-raw", response_model=PredictionResponse)
def predict_raw(
    text: str = Body(
        ...,
        media_type="text/plain",
        description="Raw text body for large pasted content. No JSON escaping needed.",
    ),
    authorization: Optional[str] = Header(default=None),
) -> PredictionResponse:
    return _predict_and_log(
        text=text,
        source="raw_text",
        authorization=authorization,
        mode="plain_text",
    )


@app.get("/")
def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/insights")
def frontend_insights() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "insights.html")


@app.get("/dashboard")
def frontend_dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "dashboard.html")


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

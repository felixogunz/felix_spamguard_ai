from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import ROOT_DIR
from src.inference import (
    load_metrics,
    load_model,
    predict_message,
)

app = FastAPI(
    title="SpamGuard AI",
    description="Explainable spam classification API",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=ROOT_DIR / "static"),
    name="static",
)


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=3,
        max_length=10000,
        description="Message to classify",
    )


@app.get("/", response_class=HTMLResponse)
def home():
    page = ROOT_DIR / "templates" / "index.html"
    return page.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    try:
        load_model()
        return {
            "status": "ok",
            "model_loaded": True,
        }
    except FileNotFoundError:
        return {
            "status": "not_ready",
            "model_loaded": False,
        }


@app.get("/api/metrics")
def metrics():
    report = load_metrics()

    if report is None:
        raise HTTPException(
            status_code=503,
            detail="Train the model first.",
        )

    return report


@app.post("/api/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    try:
        return predict_message(request.text.strip())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

"""Serve the frontend and the already-trained model in one Python process."""
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sentiment.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    model_path = ROOT / "artifacts" / "sentiment_pipeline.joblib"
    try:
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metrics["model_sha256"]:
            raise RuntimeError("Model and metrics do not match. Run train.py successfully, then restart.")
        # Only load our local artifact. joblib files from unknown sources are unsafe.
        app.state.model = joblib.load(model_path)
        app.state.metrics = metrics
        app.state.report = json.loads((ROOT / "results" / "classification_report.json").read_text())
    except FileNotFoundError as error:
        raise RuntimeError("Training artifacts missing. Run: python train.py") from error
    yield


app = FastAPI(title="IMDb Sentiment Lab", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class ReviewInput(BaseModel):
    review: str = Field(min_length=1, max_length=20000, strict=True)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ready", "model_loaded": hasattr(app.state, "model")}


@app.get("/api/experiment")
def experiment():
    result = dict(app.state.metrics)
    result["dataset"] = {k:v for k,v in result["dataset"].items() if k != "first_rows"}
    result["classification_report"] = app.state.report
    return result


@app.post("/api/predict")
def predict(payload: ReviewInput):
    cleaned = clean_text(payload.review)
    if not cleaned or not any(c.isalpha() for c in cleaned):
        raise HTTPException(422, "Enter a movie review containing words, not just symbols or links.")
    vector = app.state.model["tfidf"].transform([payload.review])
    if vector.nnz == 0:
        raise HTTPException(422, "No familiar vocabulary found. Try a longer English movie review.")
    prediction = str(app.state.model.predict([payload.review])[0])
    return {"sentiment": prediction, "label": f"{prediction.title()} Review", "note": "Predicted class only. This model does not provide a calibrated probability."}


@app.get("/api/download/{name}")
def download(name: str):
    allowed = {"metrics.json", "metrics.csv", "confusion_matrix.png", "classification_report.txt", "dataset_audit.json"}
    if name not in allowed:
        raise HTTPException(404, "Unknown result file.")
    return FileResponse(ROOT / "results" / name, filename=name)


@app.get("/api/confusion-matrix")
def matrix_image():
    return FileResponse(ROOT / "results" / "confusion_matrix.png", media_type="image/png")

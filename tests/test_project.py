"""Independent checks of data isolation, evaluation arithmetic and API behavior."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from app import app
from sentiment.dataset import inspect_and_clean
from sentiment.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def experiment():
    return json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "artifacts/sentiment_pipeline.joblib")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_preprocessing_preserves_negation():
    raw = "<br />I DIDN'T like it &amp; I will never recommend it! https://example.org/x"
    assert clean_text(raw) == "i did not like it i will never recommend it"
    assert clean_text("good") != clean_text("not good")
    assert clean_text("No, I won’t. I can’t!") == "no i will not i can not"
    assert clean_text(clean_text(raw)) == clean_text(raw)


def test_original_data_and_artifact_integrity(experiment):
    path = ROOT / "data/IMDB_Sampled_Balanced.csv"
    data, audit = inspect_and_clean(path)
    assert len(data) == 2000
    assert audit["class_distribution"] == {"negative": 1000, "positive": 1000}
    assert audit["removed_count"] == 0
    assert audit["sha256"] == experiment["dataset"]["sha256"]
    assert hashlib.sha256((ROOT / "artifacts/sentiment_pipeline.joblib").read_bytes()).hexdigest() == experiment["model_sha256"]
    original = Path("D:/Downloads/IMDB_Sampled_Balanced.csv")
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest() == audit["sha256"]


def test_split_and_training_only_tfidf(model, experiment):
    data, _ = inspect_and_clean(ROOT / "data/IMDB_Sampled_Balanced.csv")
    train, test = train_test_split(data, test_size=.2, random_state=42, stratify=data.sentiment)
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    assert set(manifest.query("split == 'train'").source_row) == set(train.source_row)
    assert set(manifest.query("split == 'test'").source_row) == set(test.source_row)
    assert not set(train.normalized) & set(test.normalized)
    assert len(train) == 1600 and len(test) == 400
    assert train.sentiment.value_counts().to_dict() == {"negative": 800, "positive": 800}
    assert test.sentiment.value_counts().to_dict() == {"negative": 200, "positive": 200}
    # Reconstruct vocabulary and IDF independently from the recorded training rows.
    vectorizer = clone(model["tfidf"]).fit(train.review)
    assert model["tfidf"].vocabulary_ == vectorizer.vocabulary_
    np.testing.assert_allclose(model["tfidf"].idf_, vectorizer.idf_, rtol=0, atol=0)
    assert isinstance(model["svm"], LinearSVC)
    assert model["tfidf"].preprocessor is clean_text
    analyzer = model["tfidf"].build_analyzer()
    assert "not good" in analyzer("not good")
    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    lookup = data.set_index("source_row")
    assert predictions.sentiment.tolist() == lookup.loc[predictions.source_row].sentiment.tolist()
    np.testing.assert_array_equal(model.predict(lookup.loc[predictions.source_row].review), predictions.prediction)


def test_metrics_independent_arithmetic(experiment):
    saved = pd.read_csv(ROOT / "results/test_predictions.csv")
    tn = int(((saved.sentiment == "negative") & (saved.prediction == "negative")).sum())
    fp = int(((saved.sentiment == "negative") & (saved.prediction == "positive")).sum())
    fn = int(((saved.sentiment == "positive") & (saved.prediction == "negative")).sum())
    tp = int(((saved.sentiment == "positive") & (saved.prediction == "positive")).sum())
    expected = {"accuracy": (tp+tn)/len(saved), "precision": tp/(tp+fp), "recall": tp/(tp+fn), "f1": 2*tp/(2*tp+fp+fn)}
    assert experiment["confusion_matrix"] == [[tn, fp], [fn, tp]]
    for key, value in expected.items():
        assert experiment["metrics"][key] == pytest.approx(value)
        assert pd.read_csv(ROOT / "results/metrics.csv").iloc[0][key] == pytest.approx(value)
    assert experiment["positive_class"] == "positive"


@pytest.mark.parametrize("review, expected", [
    ("This movie was fantastic. The acting and story were excellent.", "positive"),
    ("This movie was extremely boring and I regret watching it.", "negative"),
])
def test_predictions_from_loaded_artifact(client, model, review, expected):
    response = client.post("/api/predict", json={"review": review})
    assert response.status_code == 200
    assert response.json()["sentiment"] == expected == model.predict([review])[0]
    assert response.json()["label"] == f"{expected.title()} Review"
    assert "probability" not in response.json()


@pytest.mark.parametrize("payload", [{"review":""}, {"review":"   "}, {"review":"<br/> https://example.com"}, {"review":"!!! 123"}, {"review":"qzxwvvbnnqqrr"}, {"review":"a"*20001}, {"review":42}, {}])
def test_invalid_inputs(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_api_artifacts(client, experiment):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status":"ready", "model_loaded":True}
    assert client.get("/api/experiment").json()["metrics"] == experiment["metrics"]
    matrix = client.get("/api/confusion-matrix")
    assert matrix.status_code == 200
    assert matrix.headers["content-type"] == "image/png"
    assert matrix.content.startswith(b"\x89PNG")
    for name in ["metrics.json", "metrics.csv", "confusion_matrix.png", "classification_report.txt", "dataset_audit.json"]:
        response = client.get(f"/api/download/{name}")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
    assert client.get("/api/download/sentiment_pipeline.joblib").status_code == 404


def test_documentation_is_generated_from_metrics(experiment):
    for filename in ["README.md", "PRESENTATION.md"]:
        text = (ROOT / filename).read_text(encoding="utf-8")
        for value in experiment["metrics"].values():
            assert f"{value:.2%}" in text


def test_cleaning_reports_removals(tmp_path):
    # Exercise cleanup on an isolated fixture, never modifying the real dataset.
    rows = [{"text":f"Unique review number {i}", "label":"positive" if i%2 else "negative"} for i in range(20)]
    rows += [{"text":"Unique review number 1", "label":"positive"}, {"text":"<br />", "label":"negative"}, {"text":"Conflicting review", "label":"positive"}, {"text":"Conflicting review", "label":"negative"}, {"text":"An invalid label", "label":"neutral"}]
    path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cleaned, audit = inspect_and_clean(path)
    assert len(cleaned) == 20
    assert audit["removed_count"] == 5
    assert len({entry["source_row"] for entry in audit["removed_rows"]}) == 5

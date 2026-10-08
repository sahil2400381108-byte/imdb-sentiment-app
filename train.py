"""Run with the virtual environment's Python and train.py; only this command trains."""
import argparse
import hashlib
import json
import logging
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment.dataset import inspect_and_clean
from sentiment.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL = ROOT / "artifacts" / "sentiment_pipeline.joblib"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_matrix(matrix):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(7.6, 6.1), facecolor="white")
    ax.imshow(matrix, cmap=ListedColormap(["#f0f4fa", "#dae5f5", "#b0c6e5", "#759cd4", "#315f9c"]), vmin=0, vmax=max(matrix.sum(axis=1)))
    ax.set_xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    ax.set_yticks([0, 1], ["Actual Negative", "Actual Positive"])
    ax.tick_params(length=0, pad=13)
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", fontsize=32, fontweight="bold", color="white" if count > matrix.max() / 2 else "#1c304a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix", loc="left", fontsize=21, fontweight="bold", pad=25)
    fig.text(.5, .035, f"Held-out test set · {matrix.sum():,} reviews · Counts, not percentages", ha="center", color="#546479", fontsize=11)
    fig.tight_layout(rect=[0, .07, 1, 1])
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "IMDB_Sampled_Balanced.csv")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    MODEL.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS / "training.log", mode="w", encoding="utf-8")])
    data, audit = inspect_and_clean(args.data)
    write_json(RESULTS / "dataset_audit.json", audit)
    logging.info("Dataset validation: %s", {k:v for k,v in audit.items() if k != "first_rows"})
    logging.info("First rows (abbreviated): %s", [{k: str(v)[:160] for k,v in r.items()} for r in audit["first_rows"]])
    logging.info("Preprocessing validation complete. Removed %s rows; negations retained.", audit["removed_count"])
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data.sentiment)
    assert not set(train.normalized).intersection(test.normalized)
    split = {"training_records": len(train), "testing_records": len(test), "training_classes": train.sentiment.value_counts().to_dict(), "testing_classes": test.sentiment.value_counts().to_dict(), "random_state": 42, "test_size": .2, "stratified": True}
    logging.info("Split: %s", split)
    pd.concat([train.assign(split="train"), test.assign(split="test")])[["source_row", "sentiment", "split"]].sort_values("source_row").to_csv(RESULTS / "split_manifest.csv", index=False)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
        ("svm", LinearSVC(C=1.0, random_state=42, dual=True, max_iter=10000)),
    ])
    logging.info("Training started: TF-IDF vocabulary and IDF fit ONLY on training records.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(train.review, train.sentiment)
    logging.info("Training complete: %s learned features.", len(pipeline["tfidf"].vocabulary_))
    predictions = pipeline.predict(test.review)
    matrix = confusion_matrix(test.sentiment, predictions, labels=["negative", "positive"])
    metrics = {"accuracy": accuracy_score(test.sentiment, predictions), "precision": precision_score(test.sentiment, predictions, pos_label="positive"), "recall": recall_score(test.sentiment, predictions, pos_label="positive"), "f1": f1_score(test.sentiment, predictions, pos_label="positive")}
    report = classification_report(test.sentiment, predictions, labels=["negative", "positive"], output_dict=True)
    logging.info("Evaluation complete: %s", metrics)
    report_text = classification_report(test.sentiment, predictions, digits=4)
    logging.info("Classification report:\n%s", report_text)
    (RESULTS / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_json(RESULTS / "classification_report.json", report)
    pd.DataFrame(report).T.to_csv(RESULTS / "classification_report.csv")
    test.assign(prediction=predictions)[["source_row", "sentiment", "prediction"]].to_csv(RESULTS / "test_predictions.csv", index=False)
    plot_matrix(matrix)
    joblib.dump(pipeline, MODEL, compress=3)
    loaded = joblib.load(MODEL)
    assert np.array_equal(predictions, loaded.predict(test.review)), "Reloaded predictions changed."
    examples = [
        ("This movie was fantastic. The acting and story were excellent.", "positive"),
        ("This movie was extremely boring and I regret watching it.", "negative"),
        ("I expected very little, but the characters won me over. A thoughtful and moving film.", "positive"),
        ("The cast is talented, yet the story is dull and never gives them anything interesting to do.", "negative"),
        ("Not a perfect film, but I enjoyed its warmth and wonderful performances.", "positive"),
        ("Beautiful photography cannot rescue this tedious, empty story.", "negative"),
    ]
    assert not set(map(lambda x: clean_text(x[0]), examples)).intersection(data.normalized)
    manual = [{"review": review, "intended_sentiment": intended, "prediction": str(loaded.predict([review])[0])} for review, intended in examples]
    write_json(RESULTS / "manual_predictions.json", manual)
    for item in manual:
        logging.info("Reloaded model: %s | %s", item["prediction"], item["review"])
    experiment = {
        "metrics": metrics, "positive_class": "positive", "dataset": audit, "split": split,
        "algorithm": "Linear SVM (Support Vector Machine)", "estimator": "sklearn.svm.LinearSVC",
        "feature_extraction": "TF-IDF (Term Frequency-Inverse Document Frequency)",
        "parameters": {"ngram_range": [1,2], "min_df": 2, "sublinear_tf": True, "stop_words": None, "max_features": None, "C": 1.0, "dual": True, "max_iter": 10000},
        "feature_count": len(pipeline["tfidf"].vocabulary_), "confusion_matrix": matrix.tolist(), "class_order": ["negative", "positive"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "methodology": "One fixed stratified holdout; no hyperparameter search or test-based model selection. Saved model remains trained on the 80% partition.",
    }
    write_json(RESULTS / "metrics.json", experiment)
    pd.DataFrame([{**metrics, "dataset_size": len(data), "positive_reviews": audit["class_distribution"]["positive"], "negative_reviews": audit["class_distribution"]["negative"], "training_records": len(train), "testing_records": len(test), "algorithm": experiment["algorithm"], "feature_extraction": experiment["feature_extraction"]}]).to_csv(RESULTS / "metrics.csv", index=False)
    from sentiment.documentation import generate_docs
    generate_docs(experiment)
    logging.info("Saved artifacts: %s; results and generated documentation in %s", MODEL, RESULTS)


if __name__ == "__main__":
    main()

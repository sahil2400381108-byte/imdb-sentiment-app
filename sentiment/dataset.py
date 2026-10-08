"""Inspect source data and keep an explicit audit trail of exclusions."""
import hashlib
from pathlib import Path

import pandas as pd

from sentiment.preprocessing import clean_text


def inspect_and_clean(path: Path):
    frame = pd.read_csv(path)
    # Infer labels by their domain and reviews by text length; fail on ambiguity.
    label_candidates = []
    for column in frame:
        values = frame[column].dropna().astype(str).str.strip().str.lower()
        known = values.isin(["positive", "negative"])
        if set(values[known]) == {"positive", "negative"} and known.mean() > .5:
            label_candidates.append(column)
    if len(label_candidates) != 1:
        raise ValueError("Expected one unambiguous positive/negative sentiment column.")
    label_col = label_candidates[0]
    text_candidates = [c for c in frame if c != label_col and (pd.api.types.is_string_dtype(frame[c]) or frame[c].dtype == object)]
    if len(text_candidates) != 1:
        raise ValueError("Expected one unambiguous review text column.")
    text_col = text_candidates[0]
    audit = {
        "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": list(frame.columns), "shape": list(frame.shape), "original_records": len(frame),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_review_texts": int(frame[text_col].duplicated().sum()),
        "original_class_distribution": frame[label_col].value_counts().to_dict(),
        "first_rows": frame.head(5).fillna("").to_dict(orient="records"),
        "review_column": text_col, "label_column": label_col,
    }
    data = frame[[text_col, label_col]].rename(columns={text_col: "review", label_col: "sentiment"}).copy()
    data["source_row"] = data.index + 2  # CSV line number including header.
    data["sentiment"] = data.sentiment.astype("string").str.strip().str.lower()
    data["normalized"] = data.review.fillna("").astype(str).map(clean_text)
    removed = []

    def exclude(mask, reason):
        nonlocal data
        removed.extend({"source_row": int(row), "reason": reason} for row in data.loc[mask, "source_row"])
        data = data.loc[~mask].copy()

    exclude(data.review.isna() | ~data.sentiment.isin(["positive", "negative"]) | ~data.normalized.str.contains(r"\w\w", regex=True), "missing, invalid, or empty normalized text")
    conflicts = data.groupby("normalized").sentiment.nunique()
    conflict_keys = conflicts[conflicts > 1].index
    exclude(data.normalized.isin(conflict_keys), "conflicting labels for identical normalized review")
    audit["normalized_duplicates_same_label"] = int(data.normalized.duplicated().sum())
    exclude(data.normalized.duplicated(), "duplicate normalized review; retained first occurrence")
    audit.update({"removed_rows": removed, "removed_count": len(removed), "final_records": len(data), "class_distribution": data.sentiment.value_counts().to_dict()})
    if len(data) < 10 or data.sentiment.nunique() != 2:
        raise ValueError("Insufficient valid data for a stratified binary experiment.")
    return data, audit

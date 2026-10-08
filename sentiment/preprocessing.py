"""Stateless normalization shared by the serialized model and application."""
import html
import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("’", "'")
    # Expand common negative contractions so the token 'not' survives tokenization.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

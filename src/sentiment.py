"""Sentiment analysis – VADER (default), FinBERT, Twitter-RoBERTa.

Modelle:
  'vader'           – regelbasiert, schnell, kein Download
  'finbert'         – BERT auf Finanztexten, ~440 MB (einmalig)
  'twitter-roberta' – RoBERTa auf Social-Media-Texten, ~500 MB (einmalig)

Install für Transformer-Modelle:
  pip install transformers torch
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_VADER   = "vader"
MODEL_FINBERT = "finbert"
MODEL_ROBERTA = "twitter-roberta"

_TRANSFORMER_IDS = {
    MODEL_FINBERT: "ProsusAI/finbert",
    MODEL_ROBERTA: "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

_vader    = SentimentIntensityAnalyzer()
_pipe_cache: dict = {}   # lazy: Modell nur beim ersten Aufruf laden


# ── interne Hilfsfunktionen ───────────────────────────────────────────────────

def _label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _load_pipe(model: str):
    """Transformer-Pipeline lazy laden und cachen."""
    if model in _pipe_cache:
        return _pipe_cache[model]
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Transformer-Modelle benötigen: pip install transformers torch"
        )
    model_id = _TRANSFORMER_IDS[model]
    print(f"[sentiment] Lade '{model_id}' (einmalig, ~500 MB) …")
    pipe = pipeline(
        "text-classification",
        model=model_id,
        device=-1,          # CPU; für GPU: device=0
        truncation=True,
        max_length=512,
    )
    _pipe_cache[model] = pipe
    print(f"[sentiment] '{model_id}' geladen.")
    return pipe


def _transformer_scores(texts: list[str], model: str, batch_size: int = 16) -> list[float]:
    """Batch-Inferenz → Compound-Score [-1, 1] pro Text."""
    pipe   = _load_pipe(model)
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:512] if t.strip() else " " for t in texts[i : i + batch_size]]
        try:
            results = pipe(batch, truncation=True, batch_size=batch_size)
            for r in results:
                lbl = r["label"].lower()
                s   = float(r["score"])
                scores.append(s if "pos" in lbl else -s if "neg" in lbl else 0.0)
        except Exception:
            scores.extend([0.0] * len(batch))
    return scores


# ── öffentliche API ───────────────────────────────────────────────────────────

def analyze(posts: pd.DataFrame, model: str = MODEL_VADER) -> pd.DataFrame:
    """Sentiment-Spalten zu einem Posts-DataFrame hinzufügen.

    Parameters
    ----------
    posts : DataFrame mit Spalten 'title' und/oder 'text'
    model : 'vader' (schnell) | 'finbert' | 'twitter-roberta'

    Returns
    -------
    DataFrame mit neuen Spalten 'compound' und 'sentiment_label'
    """
    if posts.empty:
        out = posts.copy()
        out["compound"]        = pd.Series(dtype=float)
        out["sentiment_label"] = pd.Series(dtype=str)
        return out

    posts  = posts.copy()
    titles = posts.get("title", pd.Series([""] * len(posts))).fillna("")
    texts  = posts.get("text",  pd.Series([""] * len(posts))).fillna("")
    combined = (titles + " " + texts).str.strip()

    if model == MODEL_VADER:
        posts["compound"] = combined.apply(
            lambda t: _vader.polarity_scores(t)["compound"]
        )
    else:
        posts["compound"] = _transformer_scores(combined.tolist(), model)

    posts["sentiment_label"] = posts["compound"].map(_label)
    return posts


def compare_models(
    posts: pd.DataFrame,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Sentiment aller angegebenen Modelle nebeneinander vergleichen.

    Returns
    -------
    DataFrame mit Spalten compound_vader, compound_finbert, …
    """
    if models is None:
        models = [MODEL_VADER, MODEL_FINBERT, MODEL_ROBERTA]

    out = posts.copy()
    for m in models:
        try:
            scored = analyze(posts, model=m)
            out[f"compound_{m}"]        = scored["compound"]
            out[f"sentiment_label_{m}"] = scored["sentiment_label"]
        except Exception as e:
            print(f"[compare_models] Modell '{m}' übersprungen: {e}")
    return out


def aggregate(posts: pd.DataFrame) -> dict:
    """Aggregierte Sentiment-Metriken für einen Datensatz."""
    if posts.empty or "compound" not in posts.columns:
        return {"mean_compound": 0.0, "label": "neutral", "counts": {}}

    mean   = posts["compound"].mean()
    counts = posts["sentiment_label"].value_counts().to_dict()
    return {
        "mean_compound": round(mean, 4),
        "label":         _label(mean),
        "counts":        counts,
    }

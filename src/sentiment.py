"""Sentiment analysis – VADER (default), FinBERT, Twitter-RoBERTa.
Stance Detection – Zero-Shot NLI (DeBERTa-v3-small, ~85 MB).

Modelle:
  'vader'           – regelbasiert, schnell, kein Download
  'finbert'         – BERT auf Finanztexten, ~440 MB (einmalig)
  'twitter-roberta' – RoBERTa auf Social-Media-Texten, ~500 MB (einmalig)

Install für Transformer-Modelle:
  pip install transformers torch
"""

import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_VADER   = "vader"
MODEL_FINBERT = "finbert"
MODEL_ROBERTA = "twitter-roberta"
MODEL_NLI     = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

_TRANSFORMER_IDS = {
    MODEL_FINBERT: "ProsusAI/finbert",
    MODEL_ROBERTA: "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

_vader    = SentimentIntensityAnalyzer()
_pipe_cache: dict = {}   # lazy: Modell nur beim ersten Aufruf laden
_nli_pipe  = None         # Zero-Shot NLI Pipeline (lazy)
_st_model  = None         # sentence-transformers Modell (lazy)


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


# ── Semantische Filterung ─────────────────────────────────────────────────────

def semantic_filter(
    posts: pd.DataFrame,
    query: str,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Filtert Posts nach semantischer Ähnlichkeit zur Markt-Frage.

    Verwendet sentence-transformers ('all-MiniLM-L6-v2', ~80 MB).
    Posts mit cosine-Ähnlichkeit < threshold werden verworfen.

    Parameters
    ----------
    posts     : DataFrame mit Spalten 'title' und/oder 'text'
    query     : vollständige Markt-Frage (z.B. "Will Bitcoin reach $150k?")
    threshold : Mindestsimilarität [0..1], Standard 0.20

    Returns
    -------
    Gefilterter DataFrame, absteigend nach 'semantic_score' sortiert.
    Gibt ungefilterten DataFrame zurück falls sentence-transformers fehlt.
    """
    if posts.empty:
        return posts

    global _st_model
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("[semantic_filter] sentence-transformers nicht installiert – kein Filter.")
        return posts

    if _st_model is None:
        print("[semantic_filter] Lade 'all-MiniLM-L6-v2' (~80 MB, einmalig) ...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[semantic_filter] Modell geladen.")

    titles = posts.get("title", pd.Series([""] * len(posts))).fillna("")
    texts  = posts.get("text",  pd.Series([""] * len(posts))).fillna("")
    combined = (titles + " " + texts).str.strip().tolist()

    q_emb = _st_model.encode(query,    convert_to_tensor=True)
    t_emb = _st_model.encode(combined, convert_to_tensor=True, show_progress_bar=False)
    scores = util.cos_sim(q_emb, t_emb)[0].cpu().numpy()

    mask = scores >= threshold
    result = posts[mask].copy()
    result["semantic_score"] = scores[mask]
    result = result.sort_values("semantic_score", ascending=False).reset_index(drop=True)

    print(f"  [semantic] {mask.sum()}/{len(posts)} Posts relevant (threshold={threshold})")
    return result


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


# ── Stance Detection (Zero-Shot NLI) ─────────────────────────────────────────

def question_to_hypothesis(question: str) -> str:
    """Konvertiert Polymarket-Frage in NLI-Hypothese.

    Beispiele:
      'Will Bitcoin reach $150k?'      → 'Bitcoin reach $150k'
      'Will there be a US recession?'  → 'there be a US recession'
    """
    h = question.strip().rstrip("?")
    h = re.sub(r"^[Ww]ill\s+", "", h)
    return h


def detect_stance(
    posts: pd.DataFrame,
    hypothesis: str,
    batch_size: int = 8,
) -> tuple[float, "pd.Series"]:
    """Zero-Shot NLI Stance Detection.

    Misst, ob Reddit-Posts glauben, dass das Ereignis eintritt.

    Parameters
    ----------
    posts      : DataFrame mit Spalten 'title' und/oder 'text'
    hypothesis : Deklarative Aussage (via question_to_hypothesis())
    batch_size : Texte pro NLI-Batch (kleiner = weniger RAM)

    Returns
    -------
    (mean_stance_score: float, per_post_scores: pd.Series)
      +1  = alle Posts glauben, das Ereignis tritt ein
      -1  = alle Posts glauben, das Ereignis tritt NICHT ein
       0  = neutral / unentschieden
    """
    global _nli_pipe

    if posts.empty:
        return 0.0, pd.Series(dtype=float)

    if _nli_pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Stance Detection benötigt: pip install transformers torch")
        print(f"[stance] Lade '{MODEL_NLI}' (~85 MB, einmalig) …")
        _nli_pipe = hf_pipeline(
            "zero-shot-classification",
            model=MODEL_NLI,
            device=-1,
        )
        print("[stance] Modell geladen.")

    titles   = posts.get("title", pd.Series([""] * len(posts))).fillna("")
    texts    = posts.get("text",  pd.Series([""] * len(posts))).fillna("")
    combined = (titles + " " + texts).str.strip().tolist()
    combined = [t[:512] if t.strip() else " " for t in combined]

    pos_label = f"{hypothesis} will happen"
    neg_label = f"{hypothesis} will not happen"

    scores: list[float] = []
    for i in range(0, len(combined), batch_size):
        batch = combined[i : i + batch_size]
        try:
            results = _nli_pipe(batch, candidate_labels=[pos_label, neg_label])
            if isinstance(results, dict):
                results = [results]
            for r in results:
                p = r["scores"][r["labels"].index(pos_label)]
                n = r["scores"][r["labels"].index(neg_label)]
                scores.append(p - n)
        except Exception:
            scores.extend([0.0] * len(batch))

    series = pd.Series(scores, index=posts.index[:len(scores)])
    return float(np.mean(scores)) if scores else 0.0, series

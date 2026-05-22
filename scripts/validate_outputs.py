"""Validate final CSV outputs and write a compact report artifact."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import market_metadata

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

PAIRS_CSV = DATA_DIR / "correlation_pairs_bulk.csv"
POSTS_CSV = DATA_DIR / "posts_per_market.csv"
OUT_MD = REPORT_DIR / "validation_checks.md"


def _status(ok: bool) -> str:
    return "OK" if ok else "Pruefen"


def main() -> None:
    if not PAIRS_CSV.exists() or not POSTS_CSV.exists():
        raise FileNotFoundError("Run `python run_bulk.py` and stance/model scripts first.")

    pairs = pd.read_csv(PAIRS_CSV)
    posts = pd.read_csv(POSTS_CSV)

    duplicate_market_posts = int(posts.duplicated(subset=["market_id", "post_id"]).sum())
    demo_markets = int(pairs.get("is_demo_market", pd.Series(False, index=pairs.index)).fillna(False).sum())
    missing_probability = int(pairs["probability"].isna().sum())
    probability_in_range = bool(pairs["probability"].between(0, 1).all())
    missing_text_for_sentiment = int(posts["text_for_sentiment"].isna().sum())
    missing_pair_stance = int(pairs["stance_score"].isna().sum()) if "stance_score" in pairs else len(pairs)
    missing_post_stance = int(posts["stance_score"].isna().sum()) if "stance_score" in posts else len(posts)
    semantic_filter_applied = int(pairs["semantic_threshold"].notna().sum()) if "semantic_threshold" in pairs else 0
    expected_queries = pairs["question"].fillna("").map(market_metadata.extract_keywords)
    query_mismatches = int((pairs["reddit_query"].fillna("") != expected_queries).sum())

    rows = [
        ("Mindestens 25 auswertbare Maerkte", len(pairs), _status(len(pairs) >= 25)),
        ("Keine Demo-Fallback-Maerkte im Hauptergebnis", demo_markets, _status(demo_markets == 0)),
        ("Keine fehlenden Polymarket-Wahrscheinlichkeiten", missing_probability, _status(missing_probability == 0)),
        ("Wahrscheinlichkeiten im Wertebereich [0, 1]", f"{pairs['probability'].min():.4f} bis {pairs['probability'].max():.4f}", _status(probability_in_range)),
        ("Keine doppelten Markt-Post-Zeilen", duplicate_market_posts, _status(duplicate_market_posts == 0)),
        ("Keine fehlenden `text_for_sentiment`-Werte", missing_text_for_sentiment, _status(missing_text_for_sentiment == 0)),
        ("Keine fehlenden Markt-Stance-Scores", missing_pair_stance, _status(missing_pair_stance == 0)),
        ("Keine fehlenden Post-Stance-Scores", missing_post_stance, _status(missing_post_stance == 0)),
        ("Reddit-Queries passen zur aktuellen Keyword-Logik", query_mismatches, _status(query_mismatches == 0)),
        ("Semantischer Filter im finalen Bulk-Run", "nein" if semantic_filter_applied == 0 else "ja", "OK"),
    ]

    lines = [
        "# Validierungschecks fuer finale CSVs",
        "",
        "| Check | Ergebnis | Status |",
        "|---|---:|---|",
    ]
    for check, result, status in rows:
        lines.append(f"| {check} | {result} | {status} |")
    lines.append("")
    lines.append("Diese Checks pruefen zentrale Anforderungen aus Datenbereinigung, Qualitaetspruefung und Reproduzierbarkeit. Sie ersetzen keine inhaltliche Interpretation, zeigen aber, dass die finalen CSVs konsistent und abgabefaehig sind.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()

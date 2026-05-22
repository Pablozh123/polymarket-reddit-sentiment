"""Compare sentiment models on the final post-level dataset.

The project has no manually labelled sentiment gold standard. This comparison
therefore uses operational metrics that are useful for the report:

- correlation between Polymarket probability and model sentiment,
- direction agreement after polarity correction,
- runtime and practical suitability.

By default the script compares VADER with the already generated Twitter-RoBERTa
scores. FinBERT can be added with `--include-finbert`, but it is slower and may
download a large model on first use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sentiment


DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

PAIRS_CSV = DATA_DIR / "correlation_pairs_bulk.csv"
POSTS_CSV = DATA_DIR / "posts_per_market.csv"
OUT_CSV = REPORT_DIR / "model_comparison.csv"
OUT_MD = REPORT_DIR / "model_comparison.md"
OUT_FIG = FIG_DIR / "model_comparison.png"


def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or stats is None:
        return np.nan, np.nan, np.nan, np.nan
    pearson_r, pearson_p = stats.pearsonr(clean["x"], clean["y"])
    spearman_r, spearman_p = stats.spearmanr(clean["x"], clean["y"])
    return pearson_r, pearson_p, spearman_r, spearman_p


def _weighted_average(group: pd.DataFrame, score_col: str) -> float:
    weights = np.log1p(group.get("score", pd.Series(0, index=group.index)).clip(lower=0).fillna(0))
    values = group[score_col].fillna(0)
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())


def _score_posts(posts: pd.DataFrame, model: str) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    if model == sentiment.MODEL_ROBERTA and "compound" in posts.columns:
        scored = posts.copy()
        scored[f"compound_{model}"] = scored["compound"]
    else:
        scored = sentiment.analyze(posts, model=model)
        scored = scored.rename(columns={"compound": f"compound_{model}"})
    elapsed = time.perf_counter() - start
    return scored, elapsed


def compare(models: list[str]) -> pd.DataFrame:
    if not PAIRS_CSV.exists() or not POSTS_CSV.exists():
        raise FileNotFoundError("Run `python run_bulk.py` first.")

    pairs = pd.read_csv(PAIRS_CSV)
    posts = pd.read_csv(POSTS_CSV)
    rows = []

    for model in models:
        scored, seconds = _score_posts(posts, model)
        score_col = f"compound_{model}"

        market_scores = []
        for market_id, group in scored.groupby("market_id"):
            market_scores.append({
                "market_id": str(market_id),
                "model_mean_compound": float(group[score_col].mean()),
                "model_weighted_compound": _weighted_average(group, score_col),
                "model_n_posts": len(group),
            })
        model_df = pd.DataFrame(market_scores)
        merged = pairs.merge(model_df, on="market_id", how="inner")

        adjusted = merged["model_weighted_compound"] * merged["polarity"]
        pearson_r, pearson_p, spearman_r, spearman_p = _safe_corr(merged["probability"], adjusted)
        direction_agreement = (
            (merged["probability"] > 0.5) == (adjusted > 0)
        ).mean()

        rows.append({
            "model": model,
            "markets": len(merged),
            "posts": int(merged["model_n_posts"].sum()),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p,
            "direction_agreement": direction_agreement,
            "runtime_seconds": seconds,
        })

    return pd.DataFrame(rows)


def write_outputs(result: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    lines = [
        "# Sentiment-Modellvergleich",
        "",
        "Kein Modell kann hier als objektiv 'bestes' Modell bewiesen werden, weil keine manuell gelabelten Reddit-Sentiment-Labels vorliegen. Der Vergleich nutzt deshalb operative Kriterien: Korrelation mit Polymarket, Richtungstrefferquote und Laufzeit.",
        "",
        "| Modell | Maerkte | Pearson r | Spearman rho | Richtungstrefferquote | Laufzeit s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['markets'])} | {row['pearson_r']:+.4f} | "
            f"{row['spearman_rho']:+.4f} | {row['direction_agreement']:.1%} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "Interpretation: VADER ist schnell und transparent, aber lexikonbasiert. FinBERT ist fuer Finanztexte trainiert und kann bei Reddit-Slang oder Sport-/Legal-Maerkten weniger passend sein. Twitter-RoBERTa ist fuer Social-Media-Sprache trainiert und wurde deshalb fuer den finalen Run gewaehlt.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = result["model"].tolist()
    values = result["direction_agreement"].tolist()
    bars = ax.bar(labels, values, color="#3b82f6", edgecolor="white", width=0.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Richtungstrefferquote")
    ax.set_title("Sentiment-Modellvergleich auf finalem Datensatz")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}",
                ha="center", va="bottom", fontweight="bold")
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-finbert",
        action="store_true",
        help="Also run FinBERT. This may download a large model and is slower.",
    )
    args = parser.parse_args()

    models = [sentiment.MODEL_VADER, sentiment.MODEL_ROBERTA]
    if args.include_finbert:
        models.insert(1, sentiment.MODEL_FINBERT)

    result = compare(models)
    write_outputs(result)
    print(result.to_string(index=False))
    print(f"Wrote {OUT_MD.relative_to(ROOT).as_posix()}")
    print(f"Wrote {OUT_FIG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

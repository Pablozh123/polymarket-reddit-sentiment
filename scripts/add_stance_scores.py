"""Compute stance scores from existing post-level data.

Reads `data/posts_per_market.csv`, computes one stance score per market with
the configured NLI model in `src.sentiment`, and updates
`data/correlation_pairs_bulk.csv`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sentiment

PAIRS_CSV = ROOT / "data" / "correlation_pairs_bulk.csv"
POSTS_CSV = ROOT / "data" / "posts_per_market.csv"


def main() -> None:
    if not PAIRS_CSV.exists() or not POSTS_CSV.exists():
        raise FileNotFoundError("Run `python run_bulk.py` before computing stance scores.")

    pairs = pd.read_csv(PAIRS_CSV)
    posts = pd.read_csv(POSTS_CSV)
    if pairs.empty or posts.empty:
        raise ValueError("Input CSVs are empty.")

    stance_by_market: dict[str, float] = {}
    for _, market_row in pairs.iterrows():
        market_id = str(market_row["market_id"])
        question = str(market_row["question"])
        market_posts = posts[posts["market_id"].astype(str) == market_id].copy()
        if market_posts.empty:
            stance_by_market[market_id] = float("nan")
            continue

        hypothesis = sentiment.question_to_hypothesis(question)
        stance_score, per_post = sentiment.detect_stance(market_posts, hypothesis)
        stance_by_market[market_id] = stance_score
        posts.loc[market_posts.index, "stance_score"] = per_post.values
        print(f"{question[:70]:70s} stance={stance_score:+.3f} n={len(market_posts)}")

    pairs["stance_score"] = pairs["market_id"].astype(str).map(stance_by_market)
    pairs.to_csv(PAIRS_CSV, index=False)
    posts.to_csv(POSTS_CSV, index=False)
    print(f"Updated {PAIRS_CSV.relative_to(ROOT).as_posix()}")
    print(f"Updated {POSTS_CSV.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""
Bulk-Analyse: bis zu 30 Märkte mit Twitter-RoBERTa (live Polymarket-Daten).
Speichert:
  - data/correlation_pairs_bulk.csv   (aggregiert, Hauptkorrelation)
  - data/posts_per_market.csv         (Einzelposts mit Datum/Subreddit)

Ausführen:
    .venv/Scripts/python.exe run_bulk.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src import reddit, polymarket, sentiment, market_metadata

# ── Konfiguration ──────────────────────────────────────────────────────────────
MAX_MARKETS      = 30
POSTS_PER_MARKET = 25
INCLUDE_COMMENTS = False          # Schnellmodus: keine Kommentare
INCLUDE_STANCE   = False          # True: +~5-10 Min., fügt stance_score zur CSV hinzu
SENTIMENT_MODEL  = sentiment.MODEL_ROBERTA  # Twitter-RoBERTa: besser für Reddit-Stil
SLEEP_BETWEEN    = 1.5            # s zwischen Märkten

SUBREDDITS = ["politics", "worldnews", "stocks", "investing", "news",
              "Economics", "geopolitics"]

SAMPLE_DATA = [
    {"question": "Will Trump be impeached in 2025?",               "probability": 0.08, "category": "Politics"},
    {"question": "Will there be a US federal shutdown in 2025?",   "probability": 0.35, "category": "Politics"},
    {"question": "Will the US raise the debt ceiling in 2025?",    "probability": 0.82, "category": "Politics"},
    {"question": "Will Biden pardon Trump?",                        "probability": 0.04, "category": "Politics"},
    {"question": "Will a third party win a Senate seat in 2025?",  "probability": 0.12, "category": "Politics"},
    {"question": "Will Kamala Harris run for president again?",    "probability": 0.28, "category": "Politics"},
    {"question": "Will the US enter a recession in 2025?",         "probability": 0.31, "category": "Economy"},
    {"question": "Will the Fed cut rates in 2025?",                "probability": 0.74, "category": "Economy"},
    {"question": "Will inflation stay above 3% in 2025?",          "probability": 0.40, "category": "Economy"},
    {"question": "Will gold exceed $3000 per ounce?",              "probability": 0.68, "category": "Economy"},
    {"question": "Will oil fall below $60 per barrel?",            "probability": 0.22, "category": "Economy"},
    {"question": "Will the S&P 500 reach 6500?",                   "probability": 0.55, "category": "Economy"},
    {"question": "Will Bitcoin reach $150000 in 2025?",            "probability": 0.38, "category": "Crypto"},
    {"question": "Will Ethereum exceed $5000?",                    "probability": 0.29, "category": "Crypto"},
    {"question": "Will a Bitcoin ETF reach $100B AUM?",            "probability": 0.45, "category": "Crypto"},
    {"question": "Will Solana hit $400?",                          "probability": 0.21, "category": "Crypto"},
    {"question": "Will Bitcoin market cap exceed $3 trillion?",    "probability": 0.33, "category": "Crypto"},
    {"question": "Will Apple release AR glasses in 2025?",         "probability": 0.14, "category": "Tech"},
    {"question": "Will OpenAI IPO in 2025?",                       "probability": 0.19, "category": "Tech"},
    {"question": "Will Nvidia stock double from 2024 peak?",       "probability": 0.24, "category": "Tech"},
    {"question": "Will Tesla release a $25000 car?",               "probability": 0.16, "category": "Tech"},
    {"question": "Will a major AI regulation pass in the EU?",     "probability": 0.58, "category": "Tech"},
    {"question": "Will there be a ceasefire in Ukraine in 2025?",  "probability": 0.47, "category": "Geopolitics"},
    {"question": "Will China invade Taiwan by 2026?",              "probability": 0.07, "category": "Geopolitics"},
    {"question": "Will North Korea conduct nuclear test in 2025?", "probability": 0.13, "category": "Geopolitics"},
    {"question": "Will NATO expand further in 2025?",              "probability": 0.35, "category": "Geopolitics"},
    {"question": "Will Iran produce nuclear weapon in 2025?",      "probability": 0.06, "category": "Geopolitics"},
    {"question": "Will US and Russia normalize relations?",         "probability": 0.23, "category": "Geopolitics"},
    {"question": "Will 2025 be the hottest year on record?",       "probability": 0.61, "category": "Climate"},
    {"question": "Will the US rejoin Paris Agreement?",            "probability": 0.11, "category": "Climate"},
]


def main():
    collected_at_utc = pd.Timestamp.utcnow().isoformat()
    try:
        all_markets = polymarket.get_markets(limit=200)
        if all_markets.empty or "probability" not in all_markets.columns:
            raise ValueError("leere API")
        markets = market_metadata.filter_relevant_markets(all_markets, MAX_MARKETS)
        market_source = "polymarket_live"
        print(f"Polymarket live: {len(all_markets)} Maerkte gesamt, {len(markets)} relevant")
    except Exception as exc:
        print(f"Polymarket API nicht erreichbar -> Demo-Datensatz ({exc})")
        markets = pd.DataFrame(SAMPLE_DATA).head(MAX_MARKETS)
        market_source = "demo_fallback"

    print(f"\nBulk-Analyse: {len(markets)} Maerkte, Modell=RoBERTa, Kommentare=Nein")
    print("-" * 60)

    pairs = []
    all_posts = []

    for position, (_, row) in enumerate(markets.iterrows(), start=1):
        market_fields = market_metadata.stable_market_fields(
            row=row,
            position=position,
            api_source=market_source,
            collected_at_utc=collected_at_utc,
        )
        question = market_fields["question"]
        prob = market_fields["probability"]
        category = market_fields["category"]
        keywords = market_metadata.extract_keywords(question)

        try:
            raw = reddit.get_posts(keywords, SUBREDDITS, POSTS_PER_MARKET,
                                   include_comments=False)
            if raw.empty or len(raw) < 3:
                print(f"  SKIP [{keywords}] keine Posts")
                continue

            raw_n = len(raw)
            scored = sentiment.analyze(raw, model=SENTIMENT_MODEL)
            mean_s = scored["compound"].mean()
            w = np.log1p(scored["score"].clip(lower=0).fillna(0).values)
            wtd_s = np.average(scored["compound"].values, weights=w) if w.sum() > 0 else mean_s

            polarity = market_metadata.question_polarity(question)

            if INCLUDE_STANCE:
                hyp = sentiment.question_to_hypothesis(question)
                stance_s, _ = sentiment.detect_stance(scored, hyp)
            else:
                stance_s = float("nan")

            pairs.append({
                **market_fields,
                "reddit_query": keywords,
                "subreddits": ",".join(SUBREDDITS),
                "sentiment_model": SENTIMENT_MODEL,
                "semantic_threshold": float("nan"),
                "mean_compound": mean_s,
                "weighted_compound": wtd_s,
                "adjusted_compound": mean_s * polarity,
                "adjusted_weighted": wtd_s * polarity,
                "polarity": polarity,
                "stance_score": stance_s,
                "n_posts": len(scored),
                "n_comments": 0,
                "n_total": len(scored),
                "n_raw_posts": raw_n,
                "n_after_semantic_filter": len(scored),
                "keywords": keywords,
            })

            post_export = scored.rename(columns={"id": "post_id"}).copy()
            post_export["market_id"] = market_fields["market_id"]
            post_export["clob_token_id"] = market_fields["clob_token_id"]
            post_export["market_url"] = market_fields["market_url"]
            post_export["market_question"] = question
            post_export["probability"] = prob
            post_export["category"] = category
            post_export["api_source"] = market_source
            post_export["collected_at_utc"] = collected_at_utc
            post_export["reddit_query"] = keywords
            post_export["sentiment_model"] = SENTIMENT_MODEL
            post_export["semantic_threshold"] = float("nan")
            post_export["n_raw_posts_for_market"] = raw_n
            post_export["n_after_filter_for_market"] = len(scored)
            titles = post_export.get("title", pd.Series([""] * len(post_export))).fillna("")
            texts = post_export.get("text", pd.Series([""] * len(post_export))).fillna("")
            post_export["text_for_sentiment"] = (
                titles + " " + texts
            ).str.replace(r"\s+", " ", regex=True).str.strip()
            post_cols = [
                "market_id", "clob_token_id", "market_question", "market_url",
                "probability", "category", "api_source", "collected_at_utc",
                "reddit_query", "sentiment_model", "semantic_threshold",
                "n_raw_posts_for_market", "n_after_filter_for_market",
                "post_id", "content_type", "title", "text", "text_for_sentiment",
                "subreddit", "score", "num_comments", "compound",
                "sentiment_label", "created_utc", "url",
            ]
            for col in post_cols:
                if col not in post_export.columns:
                    post_export[col] = ""
            all_posts.append(post_export[post_cols].copy())

            print(f"  [{len(pairs):>2}] p={prob:.2f}  mean={mean_s:+.3f}  "
                  f"posts={len(scored):>3}  [{keywords}]")

        except Exception as e:
            print(f"  Fehler '{question[:40]}': {e}")

        time.sleep(SLEEP_BETWEEN)

    os.makedirs("data", exist_ok=True)

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv("data/correlation_pairs_bulk.csv", index=False)
    print(f"\nGespeichert: data/correlation_pairs_bulk.csv  ({len(pairs_df)} Maerkte)")

    if all_posts:
        posts_df = pd.concat(all_posts, ignore_index=True)
        posts_df.to_csv("data/posts_per_market.csv", index=False)
        print(f"Gespeichert: data/posts_per_market.csv  ({len(posts_df)} Posts)")

    if pairs_df.empty:
        print("  Keine auswertbaren Maerkte gespeichert.")
        return

    print(f"  Durchschnitt Posts/Markt: {pairs_df['n_posts'].mean():.0f}")
    print(f"  Pearson r (quick): ", end="")
    try:
        from scipy import stats
        r, p = stats.pearsonr(pairs_df["probability"], pairs_df["mean_compound"])
        print(f"{r:+.3f}  (p={p:.3f})")
    except Exception:
        print("n/a")


if __name__ == "__main__":
    main()

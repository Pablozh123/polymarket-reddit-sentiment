"""
Bulk-Analyse: 30 Märkte mit Twitter-RoBERTa (live Polymarket-Daten).
Speichert:
  - data/correlation_pairs_bulk.csv   (aggregiert, für Korrelation n=30+)
  - data/posts_per_market.csv         (Einzelposts mit Datum, für Lag-Analyse)

Ausführen:
    .venv/Scripts/python.exe run_bulk.py
"""

import sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src import reddit, polymarket, sentiment

# ── Konfiguration ──────────────────────────────────────────────────────────────
MAX_MARKETS      = 30
POSTS_PER_MARKET = 25
INCLUDE_COMMENTS = False          # Schnellmodus: keine Kommentare
SENTIMENT_MODEL  = sentiment.MODEL_ROBERTA  # Twitter-RoBERTa: besser für Reddit-Stil
SLEEP_BETWEEN    = 1.5            # s zwischen Märkten

SUBREDDITS = ["politics", "worldnews", "stocks", "investing", "news",
              "Economics", "geopolitics"]

NEGATIVE_QUESTION_WORDS = {
    "recession", "shutdown", "impeach", "invade", "invasion",
    "nuclear", "ban", "crash", "default", "hurricane", "attack",
    "sanction", "collapse", "convict", "indict", "conflict",
    "crisis", "fail", "war",
}

def question_polarity(question: str) -> int:
    """Returns +1 for positive-framed questions, -1 for negative-framed.

    Negative-framed: positive Reddit sentiment means the bad outcome is
    *unlikely*, so we flip the sign before correlating with probability.
    """
    words = set(re.findall(r'[a-z]+', question.lower()))
    return -1 if words & NEGATIVE_QUESTION_WORDS else +1


STOPWORDS = {
    "will", "would", "could", "should", "has", "have", "been", "the",
    "and", "are", "for", "was", "not", "with", "this", "that", "from",
    "its", "which", "when", "who", "how", "what", "why", "does", "did",
    "any", "all", "into", "over", "about", "than", "more", "first",
    "2024", "2025", "2026", "year", "end", "hit", "win", "get", "per",
}

RELEVANT_KEYWORDS = [
    "trump", "fed", "rate", "tariff", "recession", "ukraine", "russia",
    "china", "taiwan", "election", "senate", "congress", "gdp", "inflation",
    "bitcoin", "crypto", "war", "trade", "stock", "nasdaq", "dollar",
    "gold", "oil", "iran", "nato", "ceasefire", "president", "democrat",
    "republican", "economy", "market", "bank", "debt", "deficit",
    "ethereum", "solana", "nvidia", "tesla", "openai", "regulation",
]

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


def _filter_relevant(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Filtert Märkte nach thematischer Relevanz, wortgenaues Matching."""
    if df.empty:
        return df
    mask = df["question"].str.lower().apply(
        lambda q: any(kw in set(re.findall(r'[a-z]+', q)) for kw in RELEVANT_KEYWORDS)
    )
    relevant = df[mask]
    if len(relevant) >= n:
        return relevant.head(n)
    rest = df[~mask].head(n - len(relevant))
    return pd.concat([relevant, rest]).head(n)


def extract_keywords(question: str, n: int = 4) -> str:
    words = re.findall(r"[A-Za-z]{4,}", question)
    kws = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kws[:n]) if kws else question[:50]


def main():
    try:
        all_markets = polymarket.get_markets(limit=200)
        if all_markets.empty or "probability" not in all_markets.columns:
            raise ValueError("leere API")
        markets = _filter_relevant(all_markets, MAX_MARKETS)
        print(f"Polymarket live: {len(all_markets)} Maerkte gesamt, {len(markets)} relevant")
    except Exception:
        print("Polymarket API nicht erreichbar -> Demo-Datensatz")
        markets = pd.DataFrame(SAMPLE_DATA).head(MAX_MARKETS)

    print(f"\nBulk-Analyse: {len(markets)} Maerkte, Modell=RoBERTa, Kommentare=Nein")
    print("-" * 60)

    pairs      = []
    all_posts  = []

    for i, row in markets.iterrows():
        question = row["question"]
        prob     = row["probability"]
        category = row.get("category", "Unknown")
        keywords = extract_keywords(question)

        try:
            raw = reddit.get_posts(keywords, SUBREDDITS, POSTS_PER_MARKET,
                                   include_comments=False)
            if raw.empty or len(raw) < 3:
                print(f"  SKIP [{keywords}] keine Posts")
                continue

            scored = sentiment.analyze(raw, model=SENTIMENT_MODEL)
            mean_s = scored["compound"].mean()
            w      = np.log1p(scored["score"].clip(lower=0).fillna(0).values)
            wtd_s  = np.average(scored["compound"].values, weights=w) if w.sum() > 0 else mean_s

            polarity = question_polarity(question)
            pairs.append({
                "question":          question,
                "probability":       prob,
                "mean_compound":     mean_s,
                "weighted_compound": wtd_s,
                "adjusted_compound": mean_s * polarity,
                "adjusted_weighted": wtd_s  * polarity,
                "polarity":          polarity,
                "n_posts":           len(scored),
                "n_comments":        0,
                "n_total":           len(scored),
                "keywords":          keywords,
                "category":          category,
                "model":             "roberta",
            })

            # Einzelposts für Lag-Analyse speichern
            scored["market_question"] = question
            scored["probability"]     = prob
            scored["category"]        = category
            all_posts.append(scored[["market_question", "probability", "category",
                                      "title", "subreddit", "score", "compound",
                                      "sentiment_label", "created_utc"]].copy())

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

    print(f"  Oe Posts/Markt: {pairs_df['n_posts'].mean():.0f}")
    print(f"  Pearson r (quick): ", end="")
    try:
        from scipy import stats
        r, p = stats.pearsonr(pairs_df["probability"], pairs_df["mean_compound"])
        print(f"{r:+.3f}  (p={p:.3f})")
    except Exception:
        print("n/a")


if __name__ == "__main__":
    main()

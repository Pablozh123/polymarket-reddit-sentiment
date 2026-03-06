"""
Standalone-Skript: Markt-Analyse ohne Jupyter.
Verbesserungen v2:
  - Semantische Post-Filterung (sentence-transformers, all-MiniLM-L6-v2)
  - FinBERT statt VADER (domänenspezifisch für Finanztexte)
  - 10 Märkte mit je 30 Posts + 10 Kommentaren

Ausführen:
    .venv/Scripts/python.exe run_analysis.py
"""

import sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src import reddit, polymarket, sentiment

# ── Konfiguration ─────────────────────────────────────────────────────────────
MAX_MARKETS        = 20     # Mehr Märkte für bessere Statistik
POSTS_PER_MARKET   = 30     # Mehr Posts für bessere Stichprobe
COMMENT_LIMIT      = 10     # Kommentare pro Post mitholen
INCLUDE_COMMENTS   = True   # Kommentare aktiviert
INCLUDE_STANCE     = False  # True: +~10-20 Min., fügt stance_score zur CSV hinzu
SENTIMENT_MODEL    = sentiment.MODEL_ROBERTA   # Twitter-RoBERTa: besser für Reddit-Stil
SEMANTIC_THRESHOLD = 0.20   # Mindest-Ähnlichkeit zur Markt-Frage
SUBREDDITS = ["politics", "worldnews", "stocks", "investing", "news",
              "Economics", "geopolitics", "collapse", "Futurology"]

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

SAMPLE_DATA = [
    {"question": "Will Trump be impeached in 2025?",              "probability": 0.08,  "category": "Politics"},
    {"question": "Will there be a US federal shutdown in 2025?",  "probability": 0.35,  "category": "Politics"},
    {"question": "Will the US raise the debt ceiling in 2025?",   "probability": 0.82,  "category": "Politics"},
    {"question": "Will Biden pardon Trump?",                       "probability": 0.04,  "category": "Politics"},
    {"question": "Will a third party win a Senate seat in 2025?", "probability": 0.12,  "category": "Politics"},
    {"question": "Will Kamala Harris run for president again?",   "probability": 0.28,  "category": "Politics"},
    {"question": "Will the US enter a recession in 2025?",        "probability": 0.31,  "category": "Economy"},
    {"question": "Will the Fed cut rates in 2025?",               "probability": 0.74,  "category": "Economy"},
    {"question": "Will inflation stay above 3% in 2025?",         "probability": 0.40,  "category": "Economy"},
    {"question": "Will gold exceed $3000 per ounce?",             "probability": 0.68,  "category": "Economy"},
    {"question": "Will oil fall below $60 per barrel?",           "probability": 0.22,  "category": "Economy"},
    {"question": "Will the S&P 500 reach 6500?",                  "probability": 0.55,  "category": "Economy"},
    {"question": "Will Bitcoin reach $150000 in 2025?",           "probability": 0.38,  "category": "Crypto"},
    {"question": "Will Ethereum exceed $5000?",                   "probability": 0.29,  "category": "Crypto"},
    {"question": "Will a Bitcoin ETF reach $100B AUM?",           "probability": 0.45,  "category": "Crypto"},
    {"question": "Will Solana hit $400?",                         "probability": 0.21,  "category": "Crypto"},
    {"question": "Will Bitcoin market cap exceed $3 trillion?",   "probability": 0.33,  "category": "Crypto"},
    {"question": "Will Apple release AR glasses in 2025?",        "probability": 0.14,  "category": "Tech"},
    {"question": "Will OpenAI IPO in 2025?",                      "probability": 0.19,  "category": "Tech"},
    {"question": "Will Nvidia stock double from 2024 peak?",      "probability": 0.24,  "category": "Tech"},
    {"question": "Will Tesla release a $25000 car?",              "probability": 0.16,  "category": "Tech"},
    {"question": "Will a major AI regulation pass in the EU?",    "probability": 0.58,  "category": "Tech"},
    {"question": "Will there be a ceasefire in Ukraine in 2025?", "probability": 0.47,  "category": "Geopolitics"},
    {"question": "Will China invade Taiwan by 2026?",             "probability": 0.07,  "category": "Geopolitics"},
    {"question": "Will North Korea conduct nuclear test in 2025?","probability": 0.13,  "category": "Geopolitics"},
    {"question": "Will NATO expand further in 2025?",             "probability": 0.35,  "category": "Geopolitics"},
    {"question": "Will Iran produce nuclear weapon in 2025?",     "probability": 0.06,  "category": "Geopolitics"},
    {"question": "Will US and Russia normalize relations?",        "probability": 0.23,  "category": "Geopolitics"},
    {"question": "Will 2025 be the hottest year on record?",      "probability": 0.61,  "category": "Climate"},
    {"question": "Will the US rejoin Paris Agreement?",           "probability": 0.11,  "category": "Climate"},
    {"question": "Will solar reach 20% of US energy mix?",        "probability": 0.42,  "category": "Climate"},
    {"question": "Will a major hurricane hit New York in 2025?",  "probability": 0.09,  "category": "Climate"},
    {"question": "Will Taylor Swift win Album of the Year 2025?", "probability": 0.32,  "category": "Culture"},
    {"question": "Will the NFL expand to 18 games?",              "probability": 0.27,  "category": "Culture"},
    {"question": "Will a new Lord of the Rings film be announced?","probability": 0.18, "category": "Culture"},
    {"question": "Will a new COVID variant cause WHO alert?",      "probability": 0.24,  "category": "Health"},
    {"question": "Will weight-loss drugs exceed $50B in sales?",  "probability": 0.63,  "category": "Health"},
    {"question": "Will the US ban TikTok?",                       "probability": 0.41,  "category": "Politics"},
    {"question": "Will Elon Musk remain Twitter/X CEO in 2025?",  "probability": 0.71,  "category": "Tech"},
    {"question": "Will SpaceX land humans on Mars by 2030?",      "probability": 0.09,  "category": "Tech"},
]


def extract_keywords(question: str, n: int = 4) -> str:
    words = re.findall(r"[A-Za-z]{4,}", question)
    keywords = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(keywords[:n]) if keywords else question[:50]


RELEVANT_KEYWORDS = [
    "trump", "fed", "rate", "tariff", "recession", "ukraine", "russia",
    "china", "taiwan", "election", "senate", "congress", "gdp", "inflation",
    "bitcoin", "crypto", "war", "trade", "stock", "nasdaq", "dollar",
    "gold", "oil", "iran", "nato", "ceasefire", "president", "democrat",
    "republican", "economy", "market", "bank", "debt", "deficit",
]


def _filter_relevant(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Filtert Märkte nach thematischer Relevanz (Politik/Wirtschaft/Geo)."""
    if df.empty:
        return df
    mask = df["question"].str.lower().apply(
        lambda q: any(kw in set(re.findall(r'[a-z]+', q)) for kw in RELEVANT_KEYWORDS)
    )
    relevant = df[mask]
    if len(relevant) >= n:
        return relevant.head(n)
    # Auffüllen mit verbleibenden wenn zu wenige relevante
    rest = df[~mask].head(n - len(relevant))
    return pd.concat([relevant, rest]).head(n)


def main():
    # Märkte laden – mehr holen, dann relevante filtern
    try:
        all_markets = polymarket.get_markets(limit=200)
        if all_markets.empty or 'probability' not in all_markets.columns:
            raise ValueError("Leere API")
        sample_markets = _filter_relevant(all_markets, MAX_MARKETS)
        print(f"Polymarket live: {len(all_markets)} Märkte gesamt, {len(sample_markets)} relevant")
    except Exception as e:
        print(f"Polymarket API nicht erreichbar -> Demo-Datensatz")
        sample_markets = pd.DataFrame(SAMPLE_DATA).head(MAX_MARKETS)

    print(f"\nStarte Analyse: {len(sample_markets)} Märkte, Modell={SENTIMENT_MODEL}")
    print(f"Posts/Markt={POSTS_PER_MARKET}, Kommentare={INCLUDE_COMMENTS} (max {COMMENT_LIMIT})")
    print("-" * 60)

    pairs = []
    for i, row in sample_markets.iterrows():
        question = row['question']
        prob     = row['probability']
        category = row.get('category', 'Unknown')
        keywords = extract_keywords(question, n=4)

        try:
            raw = reddit.get_posts(
                keywords, SUBREDDITS, POSTS_PER_MARKET,
                include_comments=INCLUDE_COMMENTS,
                comment_limit=COMMENT_LIMIT,
            )
            if raw.empty:
                print(f"  SKIP [{keywords}] keine Posts")
                continue

            # Semantische Filterung: nur Posts die zur Frage passen
            raw = sentiment.semantic_filter(raw, question, threshold=SEMANTIC_THRESHOLD)
            if len(raw) < 3:
                print(f"  SKIP [{keywords}] zu wenige relevante Posts nach Filterung")
                continue

            n_posts    = (raw['content_type'] == 'post').sum()    if 'content_type' in raw.columns else len(raw)
            n_comments = (raw['content_type'] == 'comment').sum() if 'content_type' in raw.columns else 0

            scored     = sentiment.analyze(raw, model=SENTIMENT_MODEL)
            mean_s     = scored['compound'].mean()
            weights    = np.log1p(scored['score'].clip(lower=0).fillna(0).values)
            weighted_s = np.average(scored['compound'].values, weights=weights) if weights.sum() > 0 else mean_s

            polarity = question_polarity(question)

            if INCLUDE_STANCE:
                hyp = sentiment.question_to_hypothesis(question)
                stance_s, _ = sentiment.detect_stance(scored, hyp)
            else:
                stance_s = float("nan")

            pairs.append({
                'question':          question,
                'probability':       prob,
                'mean_compound':     mean_s,
                'weighted_compound': weighted_s,
                'adjusted_compound': mean_s    * polarity,
                'adjusted_weighted': weighted_s * polarity,
                'polarity':          polarity,
                'stance_score':      stance_s,
                'n_posts':           n_posts,
                'n_comments':        n_comments,
                'n_total':           len(scored),
                'keywords':          keywords,
                'category':          category,
            })
            print(f"  [{len(pairs):>2}] p={prob:.2f}  mean={mean_s:+.3f}  "
                  f"posts={n_posts}  comments={n_comments}  [{keywords}]")

        except Exception as e:
            print(f"  Fehler bei '{question[:40]}': {e}")

        time.sleep(3.0)   # Mehr Pause wegen INCLUDE_COMMENTS

    # Speichern
    pairs_df = pd.DataFrame(pairs)
    os.makedirs("data", exist_ok=True)
    pairs_df.to_csv("data/correlation_pairs.csv", index=False)
    print(f"\nGespeichert: data/correlation_pairs.csv  ({len(pairs_df)} Maerkte)")
    print(f"  Ø Posts/Markt:      {pairs_df['n_posts'].mean():.0f}")
    print(f"  Ø Kommentare/Markt: {pairs_df['n_comments'].mean():.0f}")


if __name__ == "__main__":
    main()

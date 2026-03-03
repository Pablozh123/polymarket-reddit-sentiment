"""Polymarket Reddit Sentiment Dashboard."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

from src import polymarket, reddit, sentiment

st.set_page_config(
    page_title="Polymarket Reddit Sentiment",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Polymarket Reddit Sentiment")
st.caption("Analysiert Reddit-Sentiment zu Polymarket-Prediction-Märkten")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Einstellungen")

    @st.cache_data(ttl=60)
    def load_markets() -> pd.DataFrame:
        return polymarket.get_markets(limit=100)

    markets_df = pd.DataFrame()
    with st.spinner("Lade Märkte…"):
        try:
            markets_df = load_markets()
        except Exception as exc:
            st.warning(f"Polymarket API nicht erreichbar — manuelle Eingabe aktiv.")

    if not markets_df.empty:
        questions = markets_df["question"].tolist()
        selected_question = st.selectbox("Polymarket-Markt", questions)
        selected_market = markets_df[markets_df["question"] == selected_question].iloc[0].to_dict()
    else:
        selected_question = st.text_input(
            "Suchbegriff / Thema",
            placeholder="z.B. US Election, Bitcoin, Trump",
        )
        selected_market = {"question": selected_question, "probability": None, "category": "—"}

    st.divider()

    all_subs = ["politics", "worldnews", "stocks", "investing", "news", "economy", "geopolitics"]
    selected_subs = st.multiselect(
        "Subreddits",
        all_subs,
        default=["politics", "worldnews", "news"],
    )

    post_limit = st.slider("Max. Posts", min_value=10, max_value=200, value=50, step=10)

    run = st.button("Analysieren", type="primary", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────

if not selected_question:
    st.info("Gib einen Suchbegriff ein und klicke **Analysieren**.")
    st.stop()

prob = selected_market.get("probability")
col1, col2, col3 = st.columns(3)
col1.metric("Markt", selected_question[:60] + ("…" if len(selected_question) > 60 else ""))
col2.metric(
    "Polymarket-Wahrscheinlichkeit",
    f"{prob * 100:.1f}%" if prob is not None else "—",
)
col3.metric("Kategorie", selected_market.get("category") or "—")

st.divider()

if not run:
    st.info("Wähle einen Markt und klicke **Analysieren**.")
    st.stop()

if not selected_subs:
    st.warning("Bitte mindestens ein Subreddit auswählen.")
    st.stop()

# ── Fetch & analyse ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_and_analyse(query: str, subs: tuple, limit: int) -> pd.DataFrame:
    posts = reddit.get_posts(query, list(subs), limit)
    return sentiment.analyze(posts)


with st.spinner("Lade Reddit-Posts und analysiere Sentiment…"):
    try:
        posts_df = fetch_and_analyse(selected_question, tuple(selected_subs), post_limit)
    except requests.HTTPError as exc:
        st.error(f"Reddit API Fehler: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Fehler beim Laden der Posts: {exc}")
        st.stop()

if posts_df.empty:
    st.warning("Keine Posts gefunden. Versuche einen anderen Markt oder andere Subreddits.")
    st.stop()

agg = sentiment.aggregate(posts_df)

# ── Metrics ────────────────────────────────────────────────────────────────────

LABEL_EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
emoji = LABEL_EMOJI.get(agg["label"], "")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Posts analysiert", len(posts_df))
m2.metric("Ø Sentiment-Score", f"{agg['mean_compound']:+.3f}")
m3.metric("Gesamt-Sentiment", f"{emoji} {agg['label'].capitalize()}")
m4.metric(
    "Positiv / Neutral / Negativ",
    f"{agg['counts'].get('positive', 0)} / {agg['counts'].get('neutral', 0)} / {agg['counts'].get('negative', 0)}",
)

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────

chart_col, dist_col = st.columns([2, 1])

with chart_col:
    st.subheader("Sentiment-Verteilung nach Subreddit")
    fig_box = px.box(
        posts_df,
        x="subreddit",
        y="compound",
        color="subreddit",
        points="all",
        labels={"compound": "Compound Score", "subreddit": "Subreddit"},
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_box, use_container_width=True)

with dist_col:
    st.subheader("Anteil Sentiment")
    counts = agg["counts"]
    fig_pie = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        color=list(counts.keys()),
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"},
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Timeline ───────────────────────────────────────────────────────────────────

if "created_utc" in posts_df.columns:
    st.subheader("Sentiment über Zeit")
    timeline_df = posts_df.sort_values("created_utc")
    fig_line = px.scatter(
        timeline_df,
        x="created_utc",
        y="compound",
        color="sentiment_label",
        hover_data=["title", "subreddit", "score"],
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"},
        labels={"compound": "Compound Score", "created_utc": "Datum"},
    )
    fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_line, use_container_width=True)

# ── Posts table ────────────────────────────────────────────────────────────────

st.subheader("Posts")

COLOR_MAP = {"positive": "#d4edda", "negative": "#f8d7da", "neutral": "#fff3cd"}


def highlight_sentiment(row: pd.Series) -> list[str]:
    color = COLOR_MAP.get(row.get("sentiment_label", ""), "")
    return [f"background-color: {color}"] * len(row)


display_cols = ["title", "subreddit", "score", "num_comments", "compound", "sentiment_label", "url"]
existing_cols = [c for c in display_cols if c in posts_df.columns]

styled = (
    posts_df[existing_cols]
    .style.apply(highlight_sentiment, axis=1)
    .format({"compound": "{:+.3f}", "score": "{:,}", "num_comments": "{:,}"})
)
st.dataframe(styled, use_container_width=True, height=400)

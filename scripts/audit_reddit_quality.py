"""Audit Reddit hit quality and semantic filtering for the final dataset.

This script is intentionally separate from the main analysis. It documents a
data-wrangling quality check: keyword search can collect noisy Reddit hits, so
we inspect a reproducible sample and compare it with a semantic filter.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

POSTS_CSV = DATA_DIR / "posts_per_market.csv"
PAIRS_CSV = DATA_DIR / "correlation_pairs_bulk.csv"

AUDIT_CSV = REPORT_DIR / "relevance_audit_sample.csv"
SEMANTIC_CSV = REPORT_DIR / "semantic_filter_comparison.csv"
SUMMARY_MD = REPORT_DIR / "reddit_quality_audit.md"
AUDIT_FIG = FIG_DIR / "relevance_audit.png"

AUDIT_SAMPLE_SIZE = 50
AUDIT_RANDOM_STATE = 26
SEMANTIC_THRESHOLD = 0.20


MANUAL_LABELS = {
    1: ("irrelevant", "Finanz-/AI-Aktie, kein Bezug zu Argentinien oder WM-Sieg."),
    2: ("irrelevant", "Falscher Wettbewerb: FIFA World Cup statt NHL Stanley Cup."),
    3: ("irrelevant", "Makro-/Aktienbeitrag, kein Bezug zu Brasilien oder World Cup."),
    4: ("irrelevant", "ETF-Frage, kein Bezug zu Harvey Weinstein."),
    5: ("teilweise relevant", "World-Cup-Kontext, aber Iran statt Argentinien."),
    6: ("teilweise relevant", "World-Cup-Kontext, aber Hotel/Ort statt Deutschland-Sieg."),
    7: ("irrelevant", "US-Börsenstruktur, kein Bitcoin-1m-Bezug."),
    8: ("teilweise relevant", "World-Cup-Kontext, aber Steuerkosten statt Deutschland-Sieg."),
    9: ("irrelevant", "WIX-Aktienanalyse, kein NBA-Bezug."),
    10: ("teilweise relevant", "Trump/Präsident-Kontext, aber nicht Rücktritt/Amtsverlust."),
    11: ("teilweise relevant", "Jesus-Christ-Kontext, aber nicht Ereignis 'return'."),
    12: ("irrelevant", "Sportwetten-App, kein Colorado-Avalanche-Stanley-Cup-Bezug."),
    13: ("irrelevant", "Portfoliofrage, kein Harvey-Weinstein-Bezug."),
    14: ("teilweise relevant", "World-Cup-Kontext, aber Iran statt Frankreich."),
    15: ("irrelevant", "20 Jahre Gefängnis, aber falsche Person/Fall."),
    16: ("irrelevant", "NFT/Investing, kein Playboi-Carti-Album."),
    17: ("irrelevant", "Gefängnisstrafe, aber falsche Person/Fall."),
    18: ("irrelevant", "Investing-Beitrag, kein Bitcoin-1m-Bezug."),
    19: ("irrelevant", "Titanium-/Defense-DD, kein Cavaliers/NBA-Bezug."),
    20: ("irrelevant", "30-jährige Strafe, aber falsche Person/Fall."),
    21: ("irrelevant", "Geothermal-Energy-Beitrag, kein OKC/NBA-Bezug."),
    22: ("irrelevant", "LLY/Pharma-These, kein Knicks/NBA-Bezug."),
    23: ("irrelevant", "HIMS-Aktienanalyse, kein Bitcoin-1m-Bezug."),
    24: ("irrelevant", "Börsenstruktur, kein Harvey-Weinstein-Bezug."),
    25: ("irrelevant", "Defense-Aktienanalyse, kein Argentinien/WM-Bezug."),
    26: ("irrelevant", "Iran/Energie, aber kein FIFA-World-Cup-Sieg."),
    27: ("irrelevant", "30-jährige Strafe, aber falsche Person/Fall."),
    28: ("teilweise relevant", "World-Cup-Kontext, aber Arbeitskampf statt England-Sieg."),
    29: ("irrelevant", "Neue Alben allgemein, aber kein Rihanna-Album."),
    30: ("irrelevant", "NVO/WHO-Aktienbeitrag, kein Harvey-Weinstein-Bezug."),
    31: ("irrelevant", "AMS-Osram-Aktienanalyse, kein Frankreich/WM-Bezug."),
    32: ("irrelevant", "Nike-Strategie, kein Cavaliers/NBA-Finals-Bezug."),
    33: ("irrelevant", "Iran/Ölpreis, aber kein FIFA-World-Cup-Sieg."),
    34: ("irrelevant", "Qualitätsaktien, kein Harvey-Weinstein-Bezug."),
    35: ("irrelevant", "Software/AI-Trading, kein Harvey-Weinstein-Bezug."),
    36: ("irrelevant", "Bullishe Aktienpositionen, kein Bitcoin-1m-Bezug."),
    37: ("irrelevant", "FuboTV-DD, kein Colorado-Avalanche-Stanley-Cup-Bezug."),
    38: ("irrelevant", "Arbeitsmarkt/Recession, kein Trump-Amtsverlust."),
    39: ("irrelevant", "Sweetgreen-Aktie, kein Knicks/NBA-Bezug."),
    40: ("teilweise relevant", "Trump-politischer Kontext, aber nicht die Marktfrage."),
    41: ("teilweise relevant", "Jesus-Christ-Kontext, aber nicht Ereignis 'return'."),
    42: ("teilweise relevant", "World-Cup/FIFA-Kontext, aber nicht Spanien-Sieg."),
    43: ("irrelevant", "Sony/Musikgeschäft, kein Playboi-Carti-Album."),
    44: ("teilweise relevant", "Stanley-Cup-Kontext, aber falsches Team/alte Saison."),
    45: ("irrelevant", "Rivian-Aktienanalyse, kein MegaETH-Airdrop."),
    46: ("teilweise relevant", "OKC-Thunder/NBA-Champion-Kontext, aber nicht 2026-Finals-Prognose."),
    47: ("teilweise relevant", "Jesus-Christ-Kontext, aber nicht Ereignis 'return'."),
    48: ("irrelevant", "Hantavirus/Stocks, kein England/World-Cup-Bezug."),
    49: ("irrelevant", "Defense-Aktienanalyse, kein Spanien/WM-Bezug."),
    50: ("irrelevant", "Rivian-Aktienanalyse, kein Knicks/NBA-Bezug."),
}


def _clean_excerpt(value: object, max_len: int = 260) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text[: max_len - 3] + "..." if len(text) > max_len else text


def build_relevance_audit(posts: pd.DataFrame) -> pd.DataFrame:
    sample = posts.sample(n=AUDIT_SAMPLE_SIZE, random_state=AUDIT_RANDOM_STATE).reset_index(drop=True)
    rows = []
    for idx, row in sample.iterrows():
        sample_id = idx + 1
        label, note = MANUAL_LABELS[sample_id]
        rows.append({
            "sample_id": sample_id,
            "audit_label": label,
            "audit_note": note,
            "market_id": row.get("market_id", ""),
            "market_question": row.get("market_question", ""),
            "reddit_query": row.get("reddit_query", ""),
            "post_id": row.get("post_id", ""),
            "subreddit": row.get("subreddit", ""),
            "title": row.get("title", ""),
            "excerpt": _clean_excerpt(row.get("text_for_sentiment", "")),
            "url": row.get("url", ""),
        })
    return pd.DataFrame(rows)


def semantic_filter_comparison(posts: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise ImportError("Install sentence-transformers before running the semantic audit.") from exc

    selected_questions = [
        "Trump out as President before GTA VI?",
        "Will China invades Taiwan before GTA VI?",
        "Will bitcoin hit $1m before GTA VI?",
        "Will Harvey Weinstein be sentenced to no prison time?",
        "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    ]
    selected = pairs[pairs["question"].isin(selected_questions)].copy()
    if selected.empty:
        selected = pairs.head(5).copy()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = []
    for _, market in selected.iterrows():
        market_posts = posts[posts["market_id"].astype(str) == str(market["market_id"])].copy()
        if market_posts.empty:
            continue
        texts = market_posts["text_for_sentiment"].fillna("").astype(str).str.slice(0, 1200).tolist()
        q_emb = model.encode(str(market["question"]), convert_to_tensor=True)
        t_emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, t_emb)[0].cpu().numpy()
        market_posts["semantic_score"] = scores
        retained = market_posts[market_posts["semantic_score"] >= SEMANTIC_THRESHOLD]
        removed = market_posts[market_posts["semantic_score"] < SEMANTIC_THRESHOLD].sort_values("semantic_score")
        example_removed = removed.iloc[0] if not removed.empty else market_posts.sort_values("semantic_score").iloc[0]

        rows.append({
            "market_question": market["question"],
            "reddit_query": market["reddit_query"],
            "raw_posts": len(market_posts),
            "retained_posts_threshold_0_20": len(retained),
            "retention_pct": len(retained) / len(market_posts) if len(market_posts) else 0,
            "mean_semantic_score": float(market_posts["semantic_score"].mean()),
            "median_semantic_score": float(market_posts["semantic_score"].median()),
            "example_removed_title": example_removed.get("title", ""),
            "example_removed_score": float(example_removed.get("semantic_score", 0)),
        })
    return pd.DataFrame(rows)


def write_figure(audit: pd.DataFrame) -> None:
    counts = audit["audit_label"].value_counts().reindex(["relevant", "teilweise relevant", "irrelevant"], fill_value=0)
    colors = ["#16a34a", "#f59e0b", "#ef4444"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.58)
    ax.set_ylabel("Anzahl Posts")
    ax.set_title(f"Relevanz-Audit der Reddit-Treffer (n={len(audit)})")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value}\n({value / len(audit):.0%})",
                ha="center", va="bottom", fontweight="bold")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUDIT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(audit: pd.DataFrame, semantic: pd.DataFrame) -> None:
    counts = audit["audit_label"].value_counts().reindex(["relevant", "teilweise relevant", "irrelevant"], fill_value=0)
    lines = [
        "# Reddit-Trefferqualitaet: Relevanz-Audit und Semantic-Filter-Vergleich",
        "",
        f"Die Stichprobe umfasst {len(audit)} reproduzierbar gezogene Reddit-Treffer aus dem finalen Post-Level-Datensatz (`random_state={AUDIT_RANDOM_STATE}`). Jeder Treffer wurde anhand der Marktfrage in drei Kategorien codiert.",
        "",
        "| Kategorie | Anzahl | Anteil | Bedeutung |",
        "|---|---:|---:|---|",
        f"| relevant | {counts['relevant']} | {counts['relevant'] / len(audit):.1%} | Post passt direkt zur Marktfrage. |",
        f"| teilweise relevant | {counts['teilweise relevant']} | {counts['teilweise relevant'] / len(audit):.1%} | Post passt zum Thema, aber nicht genau zur Frage. |",
        f"| irrelevant | {counts['irrelevant']} | {counts['irrelevant'] / len(audit):.1%} | Post enthält Query-Wörter, aber falschen Kontext. |",
        "",
        "Interpretation: Die Stichprobe zeigt, dass die Keyword-Suche zwar Abdeckung erzeugt, aber viele Treffer nur oberflächlich passen. Das ist ein klassisches Text-Wrangling-Problem und spricht für zusätzliche Relevanzprüfung.",
        "",
        "## Semantic-Filter-Vergleich",
        "",
        f"Für fünf Beispielmärkte wurde ein `sentence-transformers`-Filter mit Schwelle {SEMANTIC_THRESHOLD:.2f} simuliert. Der finale Bulk-Run bleibt unverändert; der Vergleich zeigt nur, wie stark ein semantischer Filter die Trefferzahl reduzieren würde.",
        "",
        "| Marktfrage | Ohne Filter | Mit Filter | Retention | Ø Similarity | Beispiel entfernter Treffer |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in semantic.iterrows():
        lines.append(
            f"| {row['market_question']} | {int(row['raw_posts'])} | "
            f"{int(row['retained_posts_threshold_0_20'])} | {row['retention_pct']:.1%} | "
            f"{row['mean_semantic_score']:.3f} | {str(row['example_removed_title'])[:90]} |"
        )
    lines.extend([
        "",
        "## Text-Cleaning-Notizen",
        "",
        "- `title` und `text` werden zu `text_for_sentiment` kombiniert, damit Link-Posts ohne Body nicht verloren gehen.",
        "- Leere Bodies werden als leerer String behandelt; der Titel bleibt als auswertbarer Text erhalten.",
        "- `[deleted]` und `[removed]` werden bei Kommentaren verworfen; sehr kurze Kommentare unter 10 Zeichen werden nicht aufgenommen.",
        "- URLs und Markdown bleiben im finalen Text weitgehend erhalten. Das ist transparent, kann aber Modell-Scores beeinflussen und wird als Limitation dokumentiert.",
        "- Die eingesetzten Modelle sind englischsprachig bzw. Social-Media-orientiert. Nicht-englische oder stark gemischte Posts können daher ungenauer bewertet werden.",
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not POSTS_CSV.exists() or not PAIRS_CSV.exists():
        raise FileNotFoundError("Run the final pipeline before auditing Reddit quality.")
    posts = pd.read_csv(POSTS_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    audit = build_relevance_audit(posts)
    semantic = semantic_filter_comparison(posts, pairs)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    semantic.to_csv(SEMANTIC_CSV, index=False)
    write_figure(audit)
    write_summary(audit, semantic)

    print(f"Wrote {AUDIT_CSV.relative_to(ROOT).as_posix()}")
    print(f"Wrote {SEMANTIC_CSV.relative_to(ROOT).as_posix()}")
    print(f"Wrote {SUMMARY_MD.relative_to(ROOT).as_posix()}")
    print(f"Wrote {AUDIT_FIG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""Generate report-ready figures from the CSV outputs.

This script does not call Reddit or Polymarket. It only reads generated files
from data/ and writes figures plus a compact metric summary to reports/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - summary still works without scipy
    stats = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

PAIRS_CSV = DATA_DIR / "correlation_pairs_bulk.csv"
POSTS_CSV = DATA_DIR / "posts_per_market.csv"
DETAIL_POSTS_CSV = DATA_DIR / "posts_per_market_detail.csv"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or stats is None:
        return np.nan, np.nan, np.nan, np.nan
    pearson_r, pearson_p = stats.pearsonr(clean["x"], clean["y"])
    spearman_r, spearman_p = stats.spearmanr(clean["x"], clean["y"])
    return pearson_r, pearson_p, spearman_r, spearman_p


def pipeline_diagram() -> Path:
    path = FIG_DIR / "pipeline_diagram.png"
    steps = [
        "Polymarket\nGamma/CLOB API",
        "Market\nselection",
        "Reddit\nsearch",
        "Text cleaning\n+ semantic filter",
        "Sentiment\n+ polarity",
        "Correlation\n+ report",
    ]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    box_width = 0.13
    xs = np.linspace(0.08, 0.90, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        color = "#dbeafe" if i < 2 else "#dcfce7" if i < 4 else "#fee2e2"
        box = plt.Rectangle((x - box_width / 2, 0.42), box_width, 0.26, facecolor=color,
                            edgecolor="#334155", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - box_width / 2 - 0.012, 0.55),
                        xytext=(x + box_width / 2 + 0.012, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
    ax.set_title("Datenpipeline: Reddit-Sentiment zu Polymarket-Maerkten", fontsize=12, weight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def data_quality_figure(pairs: pd.DataFrame, posts: pd.DataFrame) -> Path | None:
    if pairs.empty and posts.empty:
        return None
    path = FIG_DIR / "data_quality_summary.png"
    metrics = []
    if not pairs.empty:
        metrics.extend([
            ("Markets", len(pairs)),
            ("Demo markets", int(pairs.get("is_demo_market", pd.Series(False, index=pairs.index)).fillna(False).sum())),
            ("Missing probability", int(pairs.get("probability", pd.Series(index=pairs.index)).isna().sum())),
            ("Median posts/market", int(pairs.get("n_total", pd.Series([0])).median())),
        ])
        semantic_applied = (
            "semantic_threshold" in pairs.columns
            and pairs["semantic_threshold"].notna().any()
        )
        if semantic_applied and {"n_after_semantic_filter", "n_raw_posts"}.issubset(pairs.columns):
            raw_total = pairs["n_raw_posts"].sum()
            after_total = pairs["n_after_semantic_filter"].sum()
            retention = after_total / raw_total if raw_total else np.nan
            metrics.append(("Semantic retention", retention))
        else:
            metrics.append(("Semantic filter applied", "No (bulk run)"))
    if not posts.empty:
        post_id_col = "post_id" if "post_id" in posts.columns else "id"
        duplicate_subset = ["market_id", post_id_col] if {"market_id", post_id_col}.issubset(posts.columns) else [post_id_col]
        metrics.extend([
            ("Rows posts CSV", len(posts)),
            ("Duplicate market-post rows", int(posts.duplicated(subset=duplicate_subset).sum()) if post_id_col in posts else 0),
            ("Subreddits", int(posts.get("subreddit", pd.Series(dtype=str)).nunique())),
        ])

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    display_values = [f"{v:.1%}" if isinstance(v, float) and 0 <= v <= 1 else str(v) for v in values]

    fig, ax = plt.subplots(figsize=(10, max(3.8, len(metrics) * 0.42)))
    ax.axis("off")
    table = ax.table(
        cellText=[[label, value] for label, value in zip(labels, display_values)],
        colLabels=["Qualitaetscheck", "Wert"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title("Datenqualitaet und Stichprobe", fontsize=12, weight="bold", pad=14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_scatter(pairs: pd.DataFrame) -> Path | None:
    if pairs.empty or "probability" not in pairs:
        return None
    y_col = "adjusted_weighted" if "adjusted_weighted" in pairs else "mean_compound"
    clean = pairs.dropna(subset=["probability", y_col]).copy()
    if len(clean) < 3:
        return None

    path = FIG_DIR / "correlation_scatter.png"
    pearson_r, pearson_p, spearman_r, spearman_p = _safe_corr(clean["probability"], clean[y_col])
    category_series = clean.get("category", pd.Series(["Other"] * len(clean), index=clean.index)).fillna("Other")
    categories = sorted(category_series.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))
    color_map = dict(zip(categories, colors))

    fig, ax = plt.subplots(figsize=(9, 6))
    for category in categories:
        subset = clean[category_series == category]
        size = subset.get("n_total", pd.Series([20] * len(subset))).fillna(10) * 2 + 30
        ax.scatter(subset["probability"], subset[y_col], s=size,
                   alpha=0.75, edgecolor="white", linewidth=0.8,
                   label=f"{category} (n={len(subset)})", color=color_map[category])

    slope, intercept = np.polyfit(clean["probability"], clean[y_col], 1)
    xs = np.linspace(clean["probability"].min(), clean["probability"].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.4)
    ax.axhline(0, color="#64748b", lw=0.8, ls=":")
    ax.axvline(0.5, color="#64748b", lw=0.8, ls=":")
    ax.set_xlabel("Polymarket-Wahrscheinlichkeit")
    ax.set_ylabel("Reddit Sentiment (adjusted weighted)")
    ax.set_title(
        f"F1: Sentiment vs. Polymarket | n={len(clean)} | "
        f"Pearson r={pearson_r:+.3f}, p={pearson_p:.3f}; "
        f"Spearman rho={spearman_r:+.3f}, p={spearman_p:.3f}"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def category_correlation(pairs: pd.DataFrame) -> Path | None:
    if pairs.empty or "category" not in pairs:
        return None
    y_col = "adjusted_weighted" if "adjusted_weighted" in pairs else "mean_compound"
    rows = []
    for category, group in pairs.dropna(subset=["probability", y_col]).groupby("category"):
        if len(group) < 3:
            continue
        pearson_r, pearson_p, _, _ = _safe_corr(group["probability"], group[y_col])
        rows.append({"category": category, "r": pearson_r, "p": pearson_p, "n": len(group)})
    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("r")
    path = FIG_DIR / "category_correlation.png"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#ef4444" if value < 0 else "#22c55e" for value in result["r"]]
    ax.barh(result["category"], result["r"], color=colors, edgecolor="white")
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r")
    ax.set_title("F1b: Korrelation nach Markt-Kategorie")
    for i, row in result.reset_index(drop=True).iterrows():
        ax.text(row["r"] + (0.03 if row["r"] >= 0 else -0.03), i,
                f"n={int(row['n'])}, p={row['p']:.2f}",
                va="center", ha="left" if row["r"] >= 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def stance_comparison(pairs: pd.DataFrame) -> Path | None:
    required = {"probability", "adjusted_weighted", "stance_score"}
    if pairs.empty or required.difference(pairs.columns):
        return None
    clean = pairs.dropna(subset=["probability", "adjusted_weighted", "stance_score"]).copy()
    if len(clean) < 3:
        return None

    r_sent, p_sent, _, _ = _safe_corr(clean["probability"], clean["adjusted_weighted"])
    r_stance, p_stance, _, _ = _safe_corr(clean["probability"], clean["stance_score"])
    path = FIG_DIR / "stance_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].scatter(clean["probability"], clean["stance_score"],
                    color="#16a34a", alpha=0.75, edgecolor="white", s=60)
    slope, intercept = np.polyfit(clean["probability"], clean["stance_score"], 1)
    xs = np.linspace(clean["probability"].min(), clean["probability"].max(), 100)
    axes[0].plot(xs, slope * xs + intercept, "k--", lw=1.2)
    axes[0].axhline(0, color="#64748b", lw=0.8, ls=":")
    axes[0].axvline(0.5, color="#64748b", lw=0.8, ls=":")
    axes[0].set_xlabel("Polymarket-Wahrscheinlichkeit")
    axes[0].set_ylabel("Stance Score")
    axes[0].set_title(f"F4: Stance vs. Markt\nr={r_stance:+.3f}, p={p_stance:.3f}, n={len(clean)}")

    labels = ["Sentiment\n(adjusted)", "Stance\n(NLI)"]
    values = [r_sent, r_stance]
    colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in values]
    bars = axes[1].bar(labels, values, color=colors, edgecolor="white", width=0.55)
    axes[1].axhline(0, color="#111827", lw=0.8)
    axes[1].set_ylim(-1, 1)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Vergleich der Korrelationsstaerke")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     value + (0.04 if value >= 0 else -0.08),
                     f"{value:+.3f}", ha="center", va="center", fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def subreddit_boxplot(posts: pd.DataFrame) -> Path | None:
    if posts.empty or {"subreddit", "compound"}.difference(posts.columns):
        return None
    groups = [
        group["compound"].dropna().values
        for _, group in posts.groupby("subreddit")
        if len(group["compound"].dropna()) >= 3
    ]
    labels = [
        subreddit
        for subreddit, group in posts.groupby("subreddit")
        if len(group["compound"].dropna()) >= 3
    ]
    if len(groups) < 2:
        return None

    path = FIG_DIR / "subreddit_boxplot.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(groups, tick_labels=labels, patch_artist=True)
    ax.axhline(0, color="#64748b", lw=0.8, ls=":")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("Compound Score")
    ax.set_title("F2: Sentiment-Verteilung nach Subreddit")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def sentiment_timeline(posts: pd.DataFrame) -> Path | None:
    if posts.empty or {"created_utc", "compound"}.difference(posts.columns):
        return None
    timeline = posts.copy()
    timeline["created_utc"] = pd.to_datetime(timeline["created_utc"], errors="coerce", utc=True)
    timeline = timeline.dropna(subset=["created_utc", "compound"])
    timeline = timeline[timeline["created_utc"] >= pd.Timestamp("2025-01-01", tz="UTC")]
    if timeline.empty:
        return None
    if "category" not in timeline:
        timeline["category"] = "Other"

    timeline["week"] = (
        timeline["created_utc"]
        .dt.tz_convert(None)
        .dt.to_period("W")
        .apply(lambda period: period.start_time)
    )
    weekly = timeline.groupby(["week", "category"]).agg(
        mean_compound=("compound", "mean"),
        n=("compound", "count"),
    ).reset_index()

    path = FIG_DIR / "sentiment_timeline.png"
    categories = (
        timeline.groupby("category")["compound"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )
    ncols = 2
    nrows = int(np.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(5.5, nrows * 2.5)), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))

    for ax, category, color in zip(axes, categories, colors):
        group = weekly[weekly["category"] == category].sort_values("week")
        ax.plot(group["week"], group["mean_compound"], marker="o", lw=1.6,
                color=color, label=category)
        ax.axhline(0, color="#64748b", lw=0.8, ls=":")
        ax.set_title(f"{category} (n={int(group['n'].sum())})", fontsize=10)
        ax.grid(alpha=0.18)
        ax.set_ylim(-1, 1)

    for ax in axes[len(categories):]:
        ax.axis("off")

    fig.suptitle(
        "F3: Reddit-Sentiment nach Kategorie seit 2025 (woechentlicher Durchschnitt)",
        fontsize=12,
        weight="bold",
    )
    for ax in axes[-ncols:]:
        ax.set_xlabel("Woche")
    for ax in axes[::ncols]:
        ax.set_ylabel("Compound Score")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(pairs: pd.DataFrame, posts: pd.DataFrame, figures: list[Path]) -> Path:
    path = REPORT_DIR / "report_summary.md"
    lines = [
        "# Automatisch generierte Report-Zusammenfassung",
        "",
        "Diese Datei wurde aus den CSV-Outputs in `data/` erstellt.",
        "",
        "## Stichprobe",
    ]
    if pairs.empty:
        lines.append("- Keine `data/correlation_pairs_bulk.csv` gefunden. Bitte zuerst `python run_bulk.py` ausfuehren.")
    else:
        demo_count = int(pairs.get("is_demo_market", pd.Series(False, index=pairs.index)).fillna(False).sum())
        lines.extend([
            f"- Maerkte: {len(pairs)}",
            f"- Demo-Fallback-Maerkte: {demo_count}",
            f"- Kategorien: {pairs.get('category', pd.Series(dtype=str)).nunique()}",
            f"- Durchschnitt Posts/Markt: {pairs.get('n_total', pd.Series([0])).mean():.1f}",
        ])
        if demo_count:
            lines.append("- Achtung: Demo-Fallback im finalen Bericht klar als nicht-live kennzeichnen.")
    if not posts.empty:
        lines.extend([
            f"- Post-/Kommentar-Zeilen: {len(posts)}",
            f"- Subreddits: {posts.get('subreddit', pd.Series(dtype=str)).nunique()}",
        ])

    if not pairs.empty and {"probability", "adjusted_weighted"}.issubset(pairs.columns):
        pearson_r, pearson_p, spearman_r, spearman_p = _safe_corr(pairs["probability"], pairs["adjusted_weighted"])
        lines.extend([
            "",
            "## F1 Korrelation",
            f"- Pearson r: {pearson_r:+.4f}, p={pearson_p:.4f}",
            f"- Spearman rho: {spearman_r:+.4f}, p={spearman_p:.4f}",
            "- Interpretation: explorative Korrelation, keine Kausalitaet behaupten.",
        ])

    if not pairs.empty and {"probability", "stance_score"}.issubset(pairs.columns):
        valid_stance = pairs.dropna(subset=["probability", "stance_score"])
        if len(valid_stance) >= 3:
            stance_r, stance_p, stance_s, stance_sp = _safe_corr(valid_stance["probability"], valid_stance["stance_score"])
            lines.extend([
                "",
                "## F4 Stance Detection",
                f"- Pearson r: {stance_r:+.4f}, p={stance_p:.4f}",
                f"- Spearman rho: {stance_s:+.4f}, p={stance_sp:.4f}",
                "- Interpretation: Stance misst Ereigniszustimmung direkter als allgemeines Sentiment.",
            ])

    lines.extend(["", "## Generierte Figuren"])
    if figures:
        for figure in figures:
            lines.append(f"- `{figure.relative_to(ROOT).as_posix()}`")
    else:
        lines.append("- Keine Figuren generiert, weil noch keine passenden CSVs vorhanden sind.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    _ensure_dirs()
    pairs = _read_csv(PAIRS_CSV)
    posts = _read_csv(POSTS_CSV)
    if posts.empty:
        posts = _read_csv(DETAIL_POSTS_CSV)

    figures = [pipeline_diagram()]
    for maybe_path in [
        data_quality_figure(pairs, posts),
        correlation_scatter(pairs),
        category_correlation(pairs),
        stance_comparison(pairs),
        subreddit_boxplot(posts),
        sentiment_timeline(posts),
    ]:
        if maybe_path is not None:
            figures.append(maybe_path)

    summary = write_summary(pairs, posts, figures)
    print(f"Summary: {summary.relative_to(ROOT).as_posix()}")
    for figure in figures:
        print(f"Figure:  {figure.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

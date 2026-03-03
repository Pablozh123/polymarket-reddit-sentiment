"""Reddit scraper using public JSON API (no credentials needed)."""

import time
import requests
import pandas as pd

DEFAULT_SUBREDDITS = ["politics", "worldnews", "stocks", "investing", "news"]

HEADERS = {"User-Agent": "polymarket-sentiment-bot/1.0"}


# ── Posts ─────────────────────────────────────────────────────────────────────

def get_posts(
    query: str,
    subreddits: list[str] | None = None,
    limit: int = 50,
    include_comments: bool = False,
    comment_limit: int = 10,
) -> pd.DataFrame:
    """Search Reddit for posts matching a query across given subreddits.

    Parameters
    ----------
    query            : Suchbegriff
    subreddits       : Liste von Subreddits (Standard: DEFAULT_SUBREDDITS)
    limit            : Max. Anzahl Posts
    include_comments : Kommentare der gefundenen Posts mitholen
    comment_limit    : Max. Kommentare pro Post (nur wenn include_comments=True)

    Returns
    -------
    DataFrame mit Spalten: id, title, text, subreddit, score,
                           num_comments, created_utc, url, content_type
    content_type = 'post' | 'comment'
    """
    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    subreddit_str = "+".join(subreddits)
    url    = f"https://www.reddit.com/r/{subreddit_str}/search.json"
    params = {
        "q":           query,
        "sort":        "new",
        "limit":       min(limit, 100),
        "restrict_sr": "1",
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    rows = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        rows.append({
            "id":           post.get("id"),
            "title":        post.get("title", ""),
            "text":         post.get("selftext", ""),
            "subreddit":    post.get("subreddit", ""),
            "score":        post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "created_utc":  pd.to_datetime(post.get("created_utc", 0), unit="s", utc=True),
            "url":          f"https://reddit.com{post.get('permalink', '')}",
            "content_type": "post",
        })

    posts_df = pd.DataFrame(rows)

    if include_comments and not posts_df.empty:
        comment_rows = []
        for _, post_row in posts_df.iterrows():
            comments = get_comments(post_row["id"], post_row["subreddit"], comment_limit)
            for body in comments:
                comment_rows.append({
                    "id":           f"{post_row['id']}_c{len(comment_rows)}",
                    "title":        "",
                    "text":         body,
                    "subreddit":    post_row["subreddit"],
                    "score":        0,           # Kommentar-Score nicht verfügbar
                    "num_comments": 0,
                    "created_utc":  post_row["created_utc"],
                    "url":          post_row["url"],
                    "content_type": "comment",
                })
            time.sleep(0.1)   # Rate-Limiting

        if comment_rows:
            posts_df = pd.concat(
                [posts_df, pd.DataFrame(comment_rows)],
                ignore_index=True,
            )

    return posts_df


# ── Kommentare ────────────────────────────────────────────────────────────────

def get_comments(post_id: str, subreddit: str, limit: int = 20) -> list[str]:
    """Fetch top-level comment texts for a single post.

    Returns
    -------
    Liste von Comment-Strings (bereinigt, ohne [deleted]/[removed])
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    try:
        response = requests.get(
            url, headers=HEADERS, params={"limit": limit}, timeout=8
        )
        response.raise_for_status()
        data = response.json()

        # data[0] = Post, data[1] = Kommentare
        comments = []
        for child in data[1].get("data", {}).get("children", []):
            body = child.get("data", {}).get("body", "")
            if body and body not in ("[deleted]", "[removed]") and len(body.strip()) > 10:
                comments.append(body.strip())

        return comments[:limit]

    except Exception:
        return []

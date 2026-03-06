"""Reddit scraper — PRAW (authentifiziert) mit Public-JSON-Fallback.

Mit .env-Credentials (höhere Rate Limits, kein 429):
    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...
    REDDIT_USER_AGENT=polymarket-sentiment-bot/1.0

Ohne .env: automatischer Fallback auf öffentliche JSON-API.
"""

import os
import time
import requests
import pandas as pd

DEFAULT_SUBREDDITS = ["politics", "worldnews", "stocks", "investing", "news"]
HEADERS = {"User-Agent": "polymarket-sentiment-bot/1.0"}


# ── PRAW-Loader ───────────────────────────────────────────────────────────────

def _load_praw():
    """Versucht praw.Reddit mit Credentials aus .env zu laden.

    Returns praw.Reddit-Instanz oder None (Fallback auf Public JSON).
    """
    try:
        import praw
        from dotenv import load_dotenv
        load_dotenv()

        client_id     = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        user_agent    = os.getenv("REDDIT_USER_AGENT", "polymarket-sentiment-bot/1.0")

        if client_id and client_secret and "your_client_id" not in client_id:
            r = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            # Verbindung testen (liest nur Metadaten)
            _ = r.subreddit("politics").id
            print("[reddit] PRAW authentifiziert – höhere Rate Limits aktiv.")
            return r
    except Exception:
        pass
    return None


# ── PRAW-Pfad ─────────────────────────────────────────────────────────────────

def _get_posts_praw(
    reddit,
    query: str,
    subreddits: list[str],
    limit: int,
    include_comments: bool,
    comment_limit: int,
) -> pd.DataFrame:
    """Posts über PRAW holen (authentifiziert, höhere Rate Limits)."""
    sub_str = "+".join(subreddits)
    rows = []

    for submission in reddit.subreddit(sub_str).search(query, sort="new", limit=limit):
        rows.append({
            "id":           submission.id,
            "title":        submission.title,
            "text":         submission.selftext or "",
            "subreddit":    str(submission.subreddit),
            "score":        submission.score,
            "num_comments": submission.num_comments,
            "created_utc":  pd.to_datetime(submission.created_utc, unit="s", utc=True),
            "url":          f"https://reddit.com{submission.permalink}",
            "content_type": "post",
        })

    posts_df = pd.DataFrame(rows)

    if include_comments and not posts_df.empty:
        comment_rows = []
        for _, post_row in posts_df.iterrows():
            try:
                submission = reddit.submission(id=post_row["id"])
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:comment_limit]:
                    body = getattr(comment, "body", "")
                    if body and body not in ("[deleted]", "[removed]") and len(body.strip()) > 10:
                        comment_rows.append({
                            "id":           f"{post_row['id']}_c{len(comment_rows)}",
                            "title":        "",
                            "text":         body.strip(),
                            "subreddit":    post_row["subreddit"],
                            "score":        getattr(comment, "score", 0),
                            "num_comments": 0,
                            "created_utc":  post_row["created_utc"],
                            "url":          post_row["url"],
                            "content_type": "comment",
                        })
            except Exception:
                pass

        if comment_rows:
            posts_df = pd.concat(
                [posts_df, pd.DataFrame(comment_rows)],
                ignore_index=True,
            )

    return posts_df


# ── Public-JSON-Pfad ──────────────────────────────────────────────────────────

def _get_with_retry(url: str, params: dict, timeout: int = 10, max_retries: int = 4) -> requests.Response:
    """GET mit exponentiellem Backoff bei 429 (30s → 60s → 120s → 240s)."""
    wait = 30
    for attempt in range(max_retries + 1):
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        if response.status_code == 429:
            if attempt < max_retries:
                print(f"  [reddit] 429 Rate Limit – warte {wait}s … (Versuch {attempt+1}/{max_retries})")
                time.sleep(wait)
                wait *= 2
            else:
                response.raise_for_status()
        else:
            response.raise_for_status()
            return response
    return response


def _get_posts_public(
    query: str,
    subreddits: list[str],
    limit: int,
    include_comments: bool,
    comment_limit: int,
) -> pd.DataFrame:
    """Posts über öffentliche Reddit JSON-API (kein Login, niedrigere Rate Limits)."""
    subreddit_str = "+".join(subreddits)
    url    = f"https://www.reddit.com/r/{subreddit_str}/search.json"
    params = {"q": query, "sort": "new", "limit": min(limit, 100), "restrict_sr": "1"}

    response = _get_with_retry(url, params, timeout=10)
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
            comments = _get_comments_public(post_row["id"], post_row["subreddit"], comment_limit)
            for body in comments:
                comment_rows.append({
                    "id":           f"{post_row['id']}_c{len(comment_rows)}",
                    "title":        "",
                    "text":         body,
                    "subreddit":    post_row["subreddit"],
                    "score":        0,
                    "num_comments": 0,
                    "created_utc":  post_row["created_utc"],
                    "url":          post_row["url"],
                    "content_type": "comment",
                })
            time.sleep(1.5)

        if comment_rows:
            posts_df = pd.concat(
                [posts_df, pd.DataFrame(comment_rows)],
                ignore_index=True,
            )

    return posts_df


def _get_comments_public(post_id: str, subreddit: str, limit: int = 20) -> list[str]:
    """Kommentare eines Posts über öffentliche JSON-API."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    try:
        response = _get_with_retry(url, {"limit": limit}, timeout=8)
        data = response.json()
        comments = []
        for child in data[1].get("data", {}).get("children", []):
            body = child.get("data", {}).get("body", "")
            if body and body not in ("[deleted]", "[removed]") and len(body.strip()) > 10:
                comments.append(body.strip())
        return comments[:limit]
    except Exception:
        return []


# ── Öffentliche API ───────────────────────────────────────────────────────────

_praw_instance = None
_praw_checked  = False


def get_posts(
    query: str,
    subreddits: list[str] | None = None,
    limit: int = 50,
    include_comments: bool = False,
    comment_limit: int = 10,
) -> pd.DataFrame:
    """Holt Reddit-Posts via PRAW (wenn .env gesetzt) oder Public JSON (Fallback).

    Parameters
    ----------
    query            : Suchbegriff
    subreddits       : Liste von Subreddits (Standard: DEFAULT_SUBREDDITS)
    limit            : Max. Anzahl Posts
    include_comments : Kommentare der Posts mitholen
    comment_limit    : Max. Kommentare pro Post

    Returns
    -------
    DataFrame mit Spalten: id, title, text, subreddit, score,
                           num_comments, created_utc, url, content_type
    """
    global _praw_instance, _praw_checked

    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    # PRAW lazy initialisieren (einmalig pro Session)
    if not _praw_checked:
        _praw_instance = _load_praw()
        _praw_checked  = True

    if _praw_instance is not None:
        try:
            return _get_posts_praw(_praw_instance, query, subreddits, limit,
                                   include_comments, comment_limit)
        except Exception as e:
            print(f"  [reddit] PRAW Fehler, Fallback auf Public API: {e}")

    return _get_posts_public(query, subreddits, limit, include_comments, comment_limit)


# Backwards-compat alias
def get_comments(post_id: str, subreddit: str, limit: int = 20) -> list[str]:
    return _get_comments_public(post_id, subreddit, limit)

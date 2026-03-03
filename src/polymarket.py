"""Polymarket API client with multiple endpoint fallbacks."""

import requests
import pandas as pd

ENDPOINTS = [
    "https://gamma-api.polymarket.com",
    "https://clob.polymarket.com",
]


def get_markets(limit: int = 50, active_only: bool = True) -> pd.DataFrame:
    """Fetch open markets from Polymarket API, trying multiple endpoints."""
    params = {
        "limit": limit,
        "active": str(active_only).lower(),
        "closed": "false",
        "archived": "false",
    }

    last_exc = None
    for base in ENDPOINTS:
        try:
            response = requests.get(f"{base}/markets", params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                data = data.get("results", data.get("data", []))

            rows = []
            for market in data:
                best_ask = market.get("bestAsk")
                best_bid = market.get("bestBid")
                if best_ask is not None and best_bid is not None:
                    try:
                        probability = (float(best_ask) + float(best_bid)) / 2
                    except (ValueError, TypeError):
                        probability = None
                else:
                    probability = None

                rows.append({
                    "id": market.get("id"),
                    "question": market.get("question", ""),
                    "category": market.get("category", ""),
                    "probability": probability,
                    "volume": market.get("volume"),
                    "end_date": market.get("endDate"),
                    "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                })

            return pd.DataFrame(rows)
        except Exception as exc:
            last_exc = exc
            continue

    raise ConnectionError(f"Polymarket API nicht erreichbar: {last_exc}")

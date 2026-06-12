"""Optional enrichment: light analyst-sentiment scraping + free-API fallbacks.

This module is *strictly optional colour* on top of the yfinance numbers, which
remain the source of truth. Everything here is:

  * **wrapped** - any failure returns ``None``; it never raises;
  * **cached** - HTTP responses go through a ``requests-cache`` session (TTL),
    so we hit external sites at most once per ticker per day;
  * **non-blocking** - the app calls this only for the *selected* ticker in the
    detail view (and the top-N during a refresh), never for the whole universe.

Sources:
  * Finviz quote page  - free, no key (analyst recommendation + price target).
  * Finnhub            - free key, recommendation trend (buy/hold/sell counts).
  * FMP                - free key, piotroski/altman-style financial score.
  * Yahoo news + VADER - free, no key (recent-headline sentiment tone).
"""

from __future__ import annotations

from typing import Optional

import config

_SESSION = None


def _session():
    """A cached requests session (1-day TTL). Falls back to plain requests."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        import requests_cache

        _SESSION = requests_cache.CachedSession(
            cache_name=str(config.HTTP_CACHE_PATH),
            backend="sqlite",
            expire_after=config.FUNDAMENTALS_TTL,
        )
    except Exception:
        import requests

        _SESSION = requests.Session()
    return _SESSION


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# --------------------------------------------------------------------------- #
# Finviz (no key)
# --------------------------------------------------------------------------- #
def get_finviz_snapshot(ticker: str) -> Optional[dict]:
    """Scrape Finviz's snapshot table. Returns {label: value} or None.

    Finviz lays the snapshot out as a flat table of label/value cells; we pull
    out the analyst-relevant ones (Recom, Target Price, plus a few extras).
    """
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    try:
        resp = _session().get(url, headers=_HEADERS, timeout=config.SCRAPE_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return None
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "lxml")
        cells = soup.select("table.snapshot-table2 td")
        if not cells:
            return None
        texts = [c.get_text(strip=True) for c in cells]
        # Cells alternate label, value, label, value, ...
        table = dict(zip(texts[0::2], texts[1::2]))
        wanted = ("Recom", "Target Price", "Price", "Analyst Recom",
                  "Perf Half Y", "Perf Year", "Insider Trans", "Inst Trans")
        out = {k: table[k] for k in wanted if k in table and table[k] not in ("-", "")}
        return out or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Finnhub (free key)
# --------------------------------------------------------------------------- #
def get_finnhub_recommendation(ticker: str) -> Optional[dict]:
    """Latest analyst recommendation trend from Finnhub (needs FINNHUB_API_KEY)."""
    if not config.FINNHUB_API_KEY:
        return None
    url = "https://finnhub.io/api/v1/stock/recommendation"
    try:
        resp = _session().get(
            url,
            params={"symbol": ticker.upper(), "token": config.FINNHUB_API_KEY},
            timeout=config.SCRAPE_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        latest = data[0]  # most recent month first
        return {
            "period": latest.get("period"),
            "strongBuy": latest.get("strongBuy"),
            "buy": latest.get("buy"),
            "hold": latest.get("hold"),
            "sell": latest.get("sell"),
            "strongSell": latest.get("strongSell"),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# FMP (free key)
# --------------------------------------------------------------------------- #
def get_fmp_score(ticker: str) -> Optional[dict]:
    """Piotroski / Altman-Z style financial score from FMP (needs FMP_API_KEY)."""
    if not config.FMP_API_KEY:
        return None
    url = f"https://financialmodelingprep.com/api/v4/score"
    try:
        resp = _session().get(
            url,
            params={"symbol": ticker.upper(), "apikey": config.FMP_API_KEY},
            timeout=config.SCRAPE_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        row = data[0] if isinstance(data, list) else data
        return {
            "altmanZScore": row.get("altmanZScore"),
            "piotroskiScore": row.get("piotroskiScore"),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Yahoo news headlines + VADER sentiment (no key)
# --------------------------------------------------------------------------- #
def get_news_sentiment(ticker: str, max_items: int = 8) -> Optional[dict]:
    """Headline-tone read from recent Yahoo Finance news (free, no key).

    Pulls recent headlines via yfinance and scores each title with VADER's
    compound polarity (-1..+1), then averages. Returns a compact summary::

        {"label": "Bullish|Neutral|Bearish", "score": <-1..1>, "n": <int>,
         "headlines": [{"title", "publisher", "link", "compound"}, ...]}

    or ``None`` if there are no headlines. Never raises.
    """
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).news or []
    except Exception:
        return None
    if not raw:
        return None

    items = []
    for it in raw[:max_items]:
        if not isinstance(it, dict):
            continue
        # yfinance has shipped two shapes: a flat dict, and a nested
        # {"content": {...}} form. Support both so we don't break on upgrade.
        content = it.get("content")
        if isinstance(content, dict):
            title = content.get("title")
            publisher = (content.get("provider") or {}).get("displayName")
            url = ((content.get("canonicalUrl") or {}).get("url")
                   or (content.get("clickThroughUrl") or {}).get("url"))
        else:
            title = it.get("title")
            publisher = it.get("publisher")
            url = it.get("link")
        if title:
            items.append({"title": title, "publisher": publisher, "link": url})
    if not items:
        return None

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        analyzer = None

    if analyzer is not None:
        for it in items:
            it["compound"] = analyzer.polarity_scores(it["title"]).get("compound", 0.0)
        score = sum(it["compound"] for it in items) / len(items)
        if score >= 0.15:
            label = "Bullish"
        elif score <= -0.15:
            label = "Bearish"
        else:
            label = "Neutral"
    else:
        score, label = None, "Unscored"

    return {"label": label, "score": score, "n": len(items), "headlines": items}


def enrich(ticker: str) -> dict:
    """Best-effort merge of all enrichment sources for one ticker.

    Always returns a dict (possibly with None values). Used by the UI detail
    view, so the keys are stable regardless of which sources are configured.
    """
    return {
        "finviz": get_finviz_snapshot(ticker),
        "finnhub": get_finnhub_recommendation(ticker),
        "fmp": get_fmp_score(ticker),
        "news_sentiment": get_news_sentiment(ticker),
    }


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = enrich(sym)
    for source, payload in result.items():
        print(f"{source}: {payload}")

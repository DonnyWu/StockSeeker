"""Data layer: pull raw per-ticker data and cache it on disk.

Design goals (from the plan):
  * **yfinance is the primary source** - no API key, gives prices, fundamentals,
    analyst targets and recommendations.
  * **No database.** Each ticker's raw payload is cached as a small JSON file
    under ``data/cache/tickers/<TICKER>.json`` with a ``fetched_at`` timestamp;
    fundamentals get a 24h TTL.
  * **Throttle + degrade gracefully.** Every network call is wrapped; a failure
    returns the last cached payload if present, otherwise ``None``. Nothing here
    ever raises to the caller, so a flaky source can never crash the app.

The raw payload is deliberately a plain JSON-serializable dict (scalars + small
lists). Deriving analytical metrics from it is :mod:`screener.metrics`'s job.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Optional

import config

# yfinance is imported lazily inside _fetch_from_yfinance so that importing this
# module (e.g. for cache-only/offline use) never fails if yfinance is missing.

# Subset of yfinance ``.info`` keys we keep. Keeping a fixed whitelist makes the
# cached JSON small and stable across yfinance versions.
_INFO_KEYS = (
    "longName", "shortName", "sector", "industry",
    "currentPrice", "regularMarketPrice", "previousClose",
    "marketCap", "sharesOutstanding", "beta",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume", "averageVolume10days",
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "priceToBook",
    "enterpriseToEbitda", "pegRatio", "trailingPegRatio",
    "trailingEps", "forwardEps",
    "totalRevenue", "revenueGrowth", "earningsGrowth",
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnEquity", "returnOnAssets",
    "debtToEquity", "currentRatio", "quickRatio", "totalDebt", "totalCash",
    "freeCashflow", "operatingCashflow",
    "dividendYield", "payoutRatio", "fiveYearAvgDividendYield",
    "targetMeanPrice", "targetMedianPrice", "targetHighPrice", "targetLowPrice",
    "numberOfAnalystOpinions", "recommendationMean", "recommendationKey",
)


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str):
    return config.TICKER_CACHE_DIR / f"{ticker.upper()}.json"


def _read_cache(ticker: str) -> Optional[dict]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_cache(ticker: str, payload: dict) -> None:
    try:
        with _cache_path(ticker).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:  # pragma: no cover - disk issues shouldn't crash us
        print(f"[fetch] cache write failed for {ticker}: {exc}")


def _is_fresh(payload: Optional[dict], ttl: int) -> bool:
    if not payload:
        return False
    ts = payload.get("fetched_at", 0)
    return (time.time() - ts) < ttl


def _clean_number(x: Any) -> Optional[float]:
    """Coerce to a finite float or None (JSON can't hold NaN/inf reliably)."""
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# yfinance pull
# --------------------------------------------------------------------------- #
def _fetch_from_yfinance(ticker: str) -> Optional[dict]:
    """Pull a fresh raw payload from yfinance. Returns None on any failure."""
    try:
        import yfinance as yf
    except Exception as exc:
        print(f"[fetch] yfinance unavailable: {exc}")
        return None

    try:
        tkr = yf.Ticker(ticker)
    except Exception as exc:
        print(f"[fetch] could not construct Ticker({ticker}): {exc}")
        return None

    payload: dict[str, Any] = {"ticker": ticker.upper(), "fetched_at": time.time()}

    # --- info dict (fundamentals + analyst data) ---
    info = {}
    try:
        raw_info = tkr.info or {}
        for k in _INFO_KEYS:
            if k in raw_info:
                val = raw_info[k]
                if isinstance(val, (int, float)):
                    val = _clean_number(val)
                info[k] = val
    except Exception as exc:
        print(f"[fetch] info failed for {ticker}: {exc}")
    payload["info"] = info

    # --- 1-year daily price history (chart + momentum + RSI) ---
    history = {"dates": [], "close": [], "volume": []}
    try:
        hist = tkr.history(period="1y", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            history["dates"] = [d.strftime("%Y-%m-%d") for d in hist.index]
            history["close"] = [_clean_number(c) for c in hist["Close"].tolist()]
            history["volume"] = [_clean_number(v) for v in hist["Volume"].tolist()]
    except Exception as exc:
        print(f"[fetch] history failed for {ticker}: {exc}")
    payload["history"] = history

    # --- annual revenue series (3-yr CAGR / acceleration) ---
    annual_revenue: list[float] = []
    try:
        stmt = tkr.income_stmt  # rows = line items, cols = period-end dates (desc)
        if stmt is not None and not stmt.empty:
            for label in ("Total Revenue", "TotalRevenue", "Operating Revenue"):
                if label in stmt.index:
                    row = stmt.loc[label]
                    # Columns are newest-first; reverse to chronological order.
                    vals = [_clean_number(v) for v in row.tolist()][::-1]
                    annual_revenue = [v for v in vals if v is not None]
                    break
    except Exception as exc:
        print(f"[fetch] income_stmt failed for {ticker}: {exc}")
    payload["annual_revenue"] = annual_revenue

    # Consider the pull a failure if we got essentially nothing useful.
    if not info and not history["close"]:
        return None
    return payload


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_raw(
    ticker: str,
    *,
    force: bool = False,
    ttl: int = config.FUNDAMENTALS_TTL,
    throttle: bool = True,
) -> Optional[dict]:
    """Return the raw payload for ``ticker``.

    Order of preference:
      1. Fresh on-disk cache (unless ``force``).
      2. A fresh yfinance pull (then written to cache).
      3. Stale on-disk cache (better than nothing when the network is down).
    """
    ticker = ticker.upper()
    cached = _read_cache(ticker)
    if not force and _is_fresh(cached, ttl):
        return cached

    if throttle:
        time.sleep(config.FETCH_THROTTLE_SECONDS)

    fresh = _fetch_from_yfinance(ticker)
    if fresh is not None:
        _write_cache(ticker, fresh)
        return fresh

    if cached is not None:
        print(f"[fetch] using stale cache for {ticker} (live fetch failed).")
    return cached


def peek_cache(ticker: str) -> Optional[dict]:
    """Return the on-disk payload for ``ticker`` without any network call.

    Used by the UI detail view to draw the price chart from whatever was cached
    during the last screen, so opening a stock is instant and offline-safe.
    """
    return _read_cache(ticker.upper())


def get_many(
    tickers: list[str],
    *,
    force: bool = False,
    ttl: int = config.FUNDAMENTALS_TTL,
    max_workers: int = config.FETCH_MAX_WORKERS,
    progress=None,
) -> dict[str, dict]:
    """Fetch many tickers concurrently. ``progress(done, total)`` is optional.

    Returns a dict {ticker: payload} including only tickers we got data for.
    Concurrency is modest (``FETCH_MAX_WORKERS``) to stay polite to yfinance;
    throttling is disabled per-call since the pool already limits parallelism.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict] = {}
    total = len(tickers)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(get_raw, t, force=force, ttl=ttl, throttle=False): t
            for t in tickers
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                payload = fut.result()
                if payload is not None:
                    results[t] = payload
            except Exception as exc:
                print(f"[fetch] worker failed for {t}: {exc}")
            done += 1
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:
                    pass
    return results


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    data = get_raw(sym, force=True)
    if data is None:
        print(f"No data for {sym}")
    else:
        info = data["info"]
        print(f"{sym}: {info.get('longName')}")
        print(f"  price      : {info.get('currentPrice')}")
        print(f"  market cap : {info.get('marketCap')}")
        print(f"  trailingPE : {info.get('trailingPE')}")
        print(f"  hist points: {len(data['history']['close'])}")
        print(f"  rev years  : {len(data['annual_revenue'])}")

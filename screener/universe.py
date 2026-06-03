"""Build and load the ticker universe.

The universe is the union of two lists:

1. **S&P 500 constituents** - fetched once from Wikipedia and cached to
   ``data/universe_sp500.csv``. Re-fetched at most weekly (``UNIVERSE_CSV_TTL``).
2. **A curated small/mid-cap growth seed list** - hand-authored and version
   controlled at ``data/universe_growth.csv`` (editable by you).

Everything degrades gracefully: if Wikipedia is unreachable we fall back to the
last cached CSV; if that is missing too, we still return the growth seed so the
app is never empty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

import config


@dataclass(frozen=True)
class UniverseEntry:
    ticker: str
    name: str
    sector: str
    source: str  # "sp500", "growth", or "both"


def _normalize_ticker(t: str) -> str:
    """Normalize a ticker for yfinance (e.g. BRK.B -> BRK-B)."""
    return str(t).strip().upper().replace(".", "-")


def _csv_is_fresh(path, ttl_seconds: int) -> bool:
    try:
        return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds
    except OSError:
        return False


def fetch_sp500(force: bool = False) -> pd.DataFrame:
    """Return the S&P 500 constituents as a DataFrame [ticker, name, sector].

    Uses the cached CSV when fresh; otherwise scrapes Wikipedia and rewrites the
    cache. Never raises - on total failure returns whatever cache exists, else
    an empty frame.
    """
    cache = config.UNIVERSE_SP500_CSV
    if not force and _csv_is_fresh(cache, config.UNIVERSE_CSV_TTL):
        try:
            return pd.read_csv(cache)
        except Exception:
            pass  # fall through to re-fetch

    try:
        # Wikipedia 403s the default urllib User-Agent, so fetch the page
        # ourselves with a browser-like header and hand the HTML to pandas.
        import io

        import requests

        resp = requests.get(
            config.SP500_WIKI_URL,
            headers={"User-Agent": "Mozilla/5.0 (StockSeeker universe builder)"},
            timeout=20,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        raw = tables[0]
        df = pd.DataFrame(
            {
                "ticker": raw["Symbol"].map(_normalize_ticker),
                "name": raw["Security"].astype(str).str.strip(),
                "sector": raw["GICS Sector"].astype(str).str.strip(),
            }
        ).dropna(subset=["ticker"])
        df = df[df["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)
        if not df.empty:
            df.to_csv(cache, index=False)
            return df
    except Exception as exc:  # network down, layout change, lxml missing, ...
        print(f"[universe] S&P 500 fetch failed ({exc}); using cache if available.")

    if cache.exists():
        try:
            return pd.read_csv(cache)
        except Exception:
            pass
    return pd.DataFrame(columns=["ticker", "name", "sector"])


def load_growth_seed() -> pd.DataFrame:
    """Load the curated growth seed list. Empty frame if the file is missing."""
    path = config.UNIVERSE_GROWTH_CSV
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "name", "sector"])
    try:
        df = pd.read_csv(path)
        df["ticker"] = df["ticker"].map(_normalize_ticker)
        for col in ("name", "sector"):
            if col not in df.columns:
                df[col] = ""
        return df[["ticker", "name", "sector"]].dropna(subset=["ticker"])
    except Exception as exc:
        print(f"[universe] growth seed load failed ({exc}).")
        return pd.DataFrame(columns=["ticker", "name", "sector"])


def build_universe(
    include_sp500: bool = True,
    include_growth: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Merge the requested lists into a deduped universe DataFrame.

    Columns: ticker, name, sector, source. ``source`` is "sp500", "growth", or
    "both" so the UI can let the user filter by where a name came from.
    """
    frames = []
    if include_sp500:
        sp = fetch_sp500(force=force).copy()
        sp["source"] = "sp500"
        frames.append(sp)
    if include_growth:
        gr = load_growth_seed().copy()
        gr["source"] = "growth"
        frames.append(gr)

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "sector", "source"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["ticker"].astype(bool)]

    # Collapse duplicates, marking anything in both lists as "both".
    def _merge_group(g: pd.DataFrame) -> pd.Series:
        sources = set(g["source"])
        source = "both" if len(sources) > 1 else next(iter(sources))
        name = next((n for n in g["name"] if isinstance(n, str) and n.strip()), "")
        sector = next((s for s in g["sector"] if isinstance(s, str) and s.strip()), "")
        return pd.Series({"name": name, "sector": sector, "source": source})

    merged = (
        combined.groupby("ticker", sort=True)
        .apply(_merge_group, include_groups=False)
        .reset_index()
    )
    return merged[["ticker", "name", "sector", "source"]]


def get_tickers(**kwargs) -> list[str]:
    """Convenience: just the list of ticker symbols."""
    return build_universe(**kwargs)["ticker"].tolist()


if __name__ == "__main__":
    uni = build_universe()
    print(f"Universe size: {len(uni)}")
    print(uni["source"].value_counts().to_string())
    print(uni.head(10).to_string(index=False))

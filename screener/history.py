"""Pick history: record each snapshot's top picks and how they did since.

Every time a snapshot is built, the top names per category — scored with the
*default* weights, so the log is deterministic and slider-independent — are
appended to ``data/history/picks_history.csv``. The Performance tab joins that
log against the latest snapshot's prices to show the return since each pick.

One row per (date, category, ticker); re-refreshing on the same day overwrites
that day's rows rather than duplicating them. The CSV lives outside
``data/cache/`` on purpose: clearing the cache should not erase the track
record.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

import config

HISTORY_DIR = config.DATA_DIR / "history"
PICKS_CSV = HISTORY_DIR / "picks_history.csv"
TOP_N = 10

_COLUMNS = ["date", "category", "rank", "ticker", "name", "score", "price"]


def record_picks(
    results: dict[str, pd.DataFrame],
    when: Optional[float] = None,
) -> pd.DataFrame:
    """Log the top ``TOP_N`` picks per category; upsert on (date, category).

    ``results`` is the output of :func:`screener.scoring.screen` (ranked
    DataFrames, best first). Returns the full history after writing.
    """
    date = time.strftime("%Y-%m-%d", time.localtime(when or time.time()))
    rows = []
    for cat, ranked in (results or {}).items():
        if ranked is None or ranked.empty:
            continue
        for i, (_, r) in enumerate(ranked.head(TOP_N).iterrows(), start=1):
            rows.append({
                "date": date,
                "category": cat,
                "rank": i,
                "ticker": r.get("ticker"),
                "name": r.get("name"),
                "score": r.get("score"),
                "price": r.get("price"),
            })
    new = pd.DataFrame(rows, columns=_COLUMNS)

    hist = load_history()
    if not hist.empty and not new.empty:
        replaced = (hist["date"] == date) & hist["category"].isin(
            new["category"].unique())
        hist = hist[~replaced]
    out = pd.concat([hist, new], ignore_index=True)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PICKS_CSV, index=False)
    return out


def load_history() -> pd.DataFrame:
    """The full pick log (empty frame with the right columns if none yet)."""
    if not PICKS_CSV.exists():
        return pd.DataFrame(columns=_COLUMNS)
    try:
        return pd.read_csv(PICKS_CSV)
    except Exception as exc:
        print(f"[history] failed to read {PICKS_CSV}: {exc}")
        return pd.DataFrame(columns=_COLUMNS)


def performance_frame(
    history: pd.DataFrame,
    snapshot: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Join the pick log with the latest snapshot prices.

    Adds ``current_price``, ``return_pct`` (since the pick) and ``days_held``.
    Tickers no longer in the snapshot get NaN current price/return — the UI
    shows them as "—" rather than dropping the rows.
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=_COLUMNS + ["current_price", "return_pct",
                                                "days_held"])
    out = history.copy()
    current = pd.Series(dtype=float)
    if snapshot is not None and not snapshot.empty and "ticker" in snapshot.columns:
        latest = snapshot.drop_duplicates("ticker").set_index("ticker")
        current = pd.to_numeric(latest["price"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["current_price"] = out["ticker"].map(current)
    out["return_pct"] = out["current_price"] / out["price"] - 1.0
    out["days_held"] = (
        pd.Timestamp.now().normalize()
        - pd.to_datetime(out["date"], errors="coerce")
    ).dt.days
    return out

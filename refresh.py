"""Build (and persist) the screen snapshot.

The snapshot is a single parquet file of computed metrics for the whole
universe, plus a small JSON sidecar of metadata (when it was built, how many
tickers). This is the project's "no database" persistence: the Streamlit app
loads it instantly and works offline, re-scoring on the fly as the user moves
the weight sliders. ``Refresh data`` in the UI (or running this script) rebuilds
it.

Run standalone to precompute (e.g. via Windows Task Scheduler):

    python refresh.py                # S&P 500 + growth seed
    python refresh.py --growth-only  # just the curated growth list (fast)
    python refresh.py --limit 50     # cap tickers (handy for a quick test)
"""

from __future__ import annotations

import json
import time
from typing import Callable, Optional

import pandas as pd

import config
from screener import fetch, history, metrics as metrics_mod, scoring, universe


def compute_snapshot(
    *,
    include_sp500: bool = True,
    include_growth: bool = True,
    force: bool = True,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Fetch the universe, compute metrics, and return them as a DataFrame.

    ``progress(done, total)`` is called as fetching proceeds (used by the UI's
    progress bar). Does not persist - call :func:`save_snapshot` for that.
    """
    uni = universe.build_universe(
        include_sp500=include_sp500, include_growth=include_growth
    )
    tickers = uni["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]

    run_start = time.time()
    raw_by_ticker = fetch.get_many(tickers, force=force, progress=progress)

    # Carry the universe's name/sector/source through, preferring live metrics.
    uni_idx = uni.set_index("ticker")
    rows = []
    for t in tickers:
        payload = raw_by_ticker.get(t)
        if not payload:
            continue
        m = metrics_mod.compute_metrics(payload)
        if t in uni_idx.index:
            if not m.get("name"):
                m["name"] = uni_idx.loc[t, "name"]
            if not m.get("sector"):
                m["sector"] = uni_idx.loc[t, "sector"]
            m["source"] = uni_idx.loc[t, "source"]
        rows.append(m)

    df = pd.DataFrame(rows)

    # Record how many tickers came from a *fresh* live pull vs. fell back to
    # stale on-disk cache. A live fetch stamps ``fetched_at`` during this run, so
    # anything older than ``run_start`` was served from cache (a silent symptom of
    # rate-limiting). The UI surfaces this so a no-op refresh doesn't look healthy.
    fresh = sum(1 for p in raw_by_ticker.values()
                if p.get("fetched_at", 0) >= run_start)
    df.attrs["fetch_stats"] = {
        "fresh": fresh,
        "stale": len(raw_by_ticker) - fresh,
        "total": len(raw_by_ticker),
    }
    return df


def save_snapshot(df: pd.DataFrame) -> dict:
    """Persist the metrics DataFrame to parquet + write metadata. Returns meta."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SNAPSHOT_PARQUET, index=False)
    meta = {
        "built_at": time.time(),
        "built_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tickers": int(len(df)),
    }
    with config.SNAPSHOT_META_JSON.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return meta


def load_snapshot() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load the persisted snapshot, or (None, None) if it doesn't exist."""
    if not config.SNAPSHOT_PARQUET.exists():
        return None, None
    try:
        df = pd.read_parquet(config.SNAPSHOT_PARQUET)
    except Exception as exc:
        print(f"[refresh] failed to read snapshot: {exc}")
        return None, None
    meta = None
    if config.SNAPSHOT_META_JSON.exists():
        try:
            with config.SNAPSHOT_META_JSON.open(encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = None
    return df, meta


def metrics_records(df: pd.DataFrame) -> list[dict]:
    """Snapshot DataFrame -> list of metric dicts for the scoring engine."""
    return df.to_dict("records") if df is not None and not df.empty else []


def record_default_picks(df: pd.DataFrame) -> None:
    """Score the fresh snapshot with the *default* weights and log the top
    picks to the history CSV (best effort — never blocks a refresh)."""
    try:
        results = scoring.screen(metrics_records(df))
        history.record_picks(results)
    except Exception as exc:
        print(f"[refresh] failed to record pick history: {exc}")


def _main():
    import argparse

    p = argparse.ArgumentParser(description="Build the StockSeeker snapshot.")
    p.add_argument("--growth-only", action="store_true",
                   help="Only the curated growth seed (skip S&P 500).")
    p.add_argument("--sp500-only", action="store_true",
                   help="Only the S&P 500 (skip the growth seed).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of tickers (quick test).")
    args = p.parse_args()

    include_sp500 = not args.growth_only
    include_growth = not args.sp500_only

    start = time.time()
    print("Building snapshot...")

    def _progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"  fetched {done}/{total}")

    df = compute_snapshot(
        include_sp500=include_sp500,
        include_growth=include_growth,
        limit=args.limit,
        progress=_progress,
    )
    meta = save_snapshot(df)
    record_default_picks(df)
    elapsed = time.time() - start
    print(f"Saved {meta['n_tickers']} tickers to {config.SNAPSHOT_PARQUET}")
    print(f"Done in {elapsed:.0f}s")


if __name__ == "__main__":
    _main()

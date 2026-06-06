"""Compute analytical metrics from a raw fetched payload.

:func:`compute_metrics` turns the JSON payload produced by :mod:`screener.fetch`
into a flat dict of named, derived numbers (returns, RSI, margins, valuation
multiples, analyst upside, etc.). Scoring consumes only this flat dict, which
keeps the scoring layer independent of yfinance's quirks.

Everything is defensive: any missing input yields ``None`` for the affected
metric rather than an exception, and ``None`` flows through scoring as "factor
unavailable" (a neutral score) so a partial payload still produces a ranking.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

# Approximate trading days per lookback window.
_WINDOWS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}


def _num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _pct_change(series: list, lookback: int) -> Optional[float]:
    """Return over ``lookback`` trading days, as a fraction (0.1 == +10%)."""
    closes = [c for c in series if c is not None]
    if len(closes) <= lookback or closes[-1 - lookback] in (None, 0):
        # Not enough history for the full window; use the oldest point we have.
        if len(closes) >= 2 and closes[0] not in (None, 0):
            return closes[-1] / closes[0] - 1.0
        return None
    return closes[-1] / closes[-1 - lookback] - 1.0


def _rsi(series: list, period: int = 14) -> Optional[float]:
    """Classic Wilder-style RSI on the last ``period`` daily changes (0-100)."""
    closes = [c for c in series if c is not None]
    if len(closes) < period + 1:
        return None
    diffs = np.diff(np.asarray(closes, dtype=float))
    recent = diffs[-period:]
    gains = recent[recent > 0].sum()
    losses = -recent[recent < 0].sum()
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return float(100.0 - 100.0 / (1.0 + rs))


def _sma(series: list, window: int) -> Optional[float]:
    closes = [c for c in series if c is not None]
    if len(closes) < window:
        return None
    return float(np.mean(closes[-window:]))


def _norm_ratio(x: Optional[float]) -> Optional[float]:
    """yfinance reports debt/equity as a percent (154.0 == 1.54x). Normalize."""
    if x is None:
        return None
    return x / 100.0 if x > 5 else x


def _norm_yield(x: Optional[float]) -> Optional[float]:
    """Normalize dividend yield to a fraction (0.025 == 2.5%).

    Recent yfinance versions report yield as a percent number (0.35 == 0.35%,
    3.0 == 3%); older versions used a fraction (0.0035). A real yield expressed
    as a fraction is almost never above ~0.15, so anything larger is a percent.
    """
    if x is None:
        return None
    return x / 100.0 if x > 0.15 else x


def compute_metrics(payload: dict) -> dict:
    """Return a flat dict of derived metrics for one ticker."""
    info = payload.get("info", {}) or {}
    history = payload.get("history", {}) or {}
    closes = history.get("close", []) or []
    volumes = history.get("volume", []) or []
    annual_rev = payload.get("annual_revenue", []) or []

    m: dict[str, Any] = {"ticker": payload.get("ticker")}
    m["name"] = info.get("longName") or info.get("shortName") or m["ticker"]
    m["sector"] = info.get("sector")
    m["industry"] = info.get("industry")

    # --- price & size ---
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None and closes:
        price = _num(closes[-1])
    m["price"] = price
    m["market_cap"] = _num(info.get("marketCap"))
    m["beta"] = _num(info.get("beta"))

    hi = _num(info.get("fiftyTwoWeekHigh"))
    lo = _num(info.get("fiftyTwoWeekLow"))
    clean_closes = [c for c in closes if c is not None]
    if hi is None and clean_closes:
        hi = max(clean_closes)
    if lo is None and clean_closes:
        lo = min(clean_closes)
    m["high_52w"] = hi
    m["low_52w"] = lo
    # Discount from the 52-wk high (0.30 == 30% below high).
    m["off_high"] = (hi - price) / hi if (hi and price) else None
    # Position within the 52-wk range (0 == at low, 1 == at high) — a cheap
    # "cheap vs its own recent history" proxy used by the value category.
    if hi is not None and lo is not None and (hi - lo) > 0 and price is not None:
        m["range_position"] = (price - lo) / (hi - lo)
    else:
        m["range_position"] = None

    avg_vol = _num(info.get("averageVolume")) or _num(info.get("averageVolume10days"))
    if avg_vol is None and volumes:
        vv = [v for v in volumes if v is not None]
        avg_vol = float(np.mean(vv)) if vv else None
    m["avg_volume"] = avg_vol
    m["avg_dollar_volume"] = avg_vol * price if (avg_vol and price) else None

    # --- growth ---
    m["revenue_ttm"] = _num(info.get("totalRevenue"))
    m["revenue_growth"] = _num(info.get("revenueGrowth"))  # YoY fraction
    m["earnings_growth"] = _num(info.get("earningsGrowth"))
    # 3-yr revenue CAGR from the annual series (chronological order).
    rev = [r for r in annual_rev if r and r > 0]
    if len(rev) >= 2:
        n = len(rev) - 1
        try:
            m["revenue_cagr"] = (rev[-1] / rev[0]) ** (1.0 / n) - 1.0
        except (ValueError, ZeroDivisionError):
            m["revenue_cagr"] = None
        # Acceleration: latest YoY step vs the earliest available YoY step.
        latest_yoy = rev[-1] / rev[-2] - 1.0
        first_yoy = rev[1] / rev[0] - 1.0
        m["revenue_acceleration"] = latest_yoy - first_yoy
    else:
        m["revenue_cagr"] = None
        m["revenue_acceleration"] = None

    # --- margins & profitability ---
    m["gross_margin"] = _num(info.get("grossMargins"))
    m["operating_margin"] = _num(info.get("operatingMargins"))
    m["profit_margin"] = _num(info.get("profitMargins"))
    m["free_cash_flow"] = _num(info.get("freeCashflow"))
    eps = _num(info.get("trailingEps"))
    m["trailing_eps"] = eps
    pm = m["profit_margin"]
    # Profitable if positive net margin or positive trailing EPS.
    m["profitable"] = bool((pm is not None and pm > 0) or (eps is not None and eps > 0))

    # --- valuation multiples ---
    m["pe_trailing"] = _num(info.get("trailingPE"))
    m["pe_forward"] = _num(info.get("forwardPE"))
    m["ps_ratio"] = _num(info.get("priceToSalesTrailing12Months"))
    m["pb_ratio"] = _num(info.get("priceToBook"))
    m["ev_ebitda"] = _num(info.get("enterpriseToEbitda"))
    m["peg_ratio"] = _num(info.get("trailingPegRatio")) or _num(info.get("pegRatio"))

    # --- quality ---
    m["roe"] = _num(info.get("returnOnEquity"))
    m["roa"] = _num(info.get("returnOnAssets"))
    m["debt_to_equity"] = _norm_ratio(_num(info.get("debtToEquity")))
    m["current_ratio"] = _num(info.get("currentRatio"))

    # --- dividend ---
    m["dividend_yield"] = _norm_yield(_num(info.get("dividendYield")))
    m["payout_ratio"] = _num(info.get("payoutRatio"))

    # --- momentum ---
    for key, lb in _WINDOWS.items():
        m[key] = _pct_change(closes, lb)
    m["rsi"] = _rsi(closes)
    ma50 = _sma(closes, 50)
    ma200 = _sma(closes, 200)
    m["ma50"] = ma50
    m["ma200"] = ma200
    m["golden_cross"] = bool(ma50 and ma200 and ma50 > ma200)

    # --- recent drop / "dip" signals ---
    # Latest single-day move and short-window declines (fractions; -0.10 == -10%).
    m["ret_1d"] = _pct_change(closes, 1)
    m["ret_1w"] = _pct_change(closes, 5)    # ~1 trading week
    m["ret_2w"] = _pct_change(closes, 10)   # ~2 trading weeks
    # Daily-return volatility and how many sigmas the latest move was. A big
    # negative drop_sigma flags an unusual one-day shock relative to the stock's
    # own noise (e.g. an earnings/guidance reaction), as opposed to routine wiggle.
    clean = [c for c in closes if c is not None]
    if len(clean) >= 21:
        arr = np.asarray(clean, dtype=float)
        rets = arr[1:] / arr[:-1] - 1.0
        vol = float(np.std(rets[-252:]))
        m["daily_vol"] = vol if vol > 0 else None
        m["drop_sigma"] = (
            m["ret_1d"] / vol if (m["ret_1d"] is not None and vol > 0) else None
        )
    else:
        m["daily_vol"] = None
        m["drop_sigma"] = None
    # Distance below the moving-average trend (negative == price below the average).
    m["pct_vs_ma50"] = (price / ma50 - 1.0) if (price and ma50) else None
    m["pct_vs_ma200"] = (price / ma200 - 1.0) if (price and ma200) else None

    # --- analyst ---
    target = _num(info.get("targetMeanPrice"))
    m["target_mean"] = target
    m["num_analysts"] = _num(info.get("numberOfAnalystOpinions"))
    m["recommendation_mean"] = _num(info.get("recommendationMean"))  # 1=buy..5=sell
    m["recommendation_key"] = info.get("recommendationKey")
    m["analyst_upside"] = (target / price - 1.0) if (target and price) else None

    return m


def metrics_for(ticker: str, force: bool = False) -> Optional[dict]:
    """Convenience: fetch + compute for a single ticker."""
    from screener import fetch

    payload = fetch.get_raw(ticker, force=force)
    return compute_metrics(payload) if payload else None


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:,.1f}%"
    if abs(v) >= 1e9:
        return f"{v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.2f}M"
    return f"{v:,.2f}"


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    met = metrics_for(sym, force=False)
    if not met:
        print(f"No metrics for {sym}")
        raise SystemExit(1)

    print(f"=== {sym}  {met['name']} ({met['sector']}) ===")
    rows = [
        ("Price", _fmt(met["price"])),
        ("Market cap", _fmt(met["market_cap"])),
        ("Off 52w high", _fmt(met["off_high"], pct=True)),
        ("Revenue (TTM)", _fmt(met["revenue_ttm"])),
        ("Revenue growth YoY", _fmt(met["revenue_growth"], pct=True)),
        ("3y revenue CAGR", _fmt(met["revenue_cagr"], pct=True)),
        ("Gross margin", _fmt(met["gross_margin"], pct=True)),
        ("Profit margin", _fmt(met["profit_margin"], pct=True)),
        ("Profitable", met["profitable"]),
        ("P/E trailing", _fmt(met["pe_trailing"])),
        ("P/E forward", _fmt(met["pe_forward"])),
        ("P/S", _fmt(met["ps_ratio"])),
        ("ROE", _fmt(met["roe"], pct=True)),
        ("Debt/Equity", _fmt(met["debt_to_equity"])),
        ("Dividend yield", _fmt(met["dividend_yield"], pct=True)),
        ("6m return", _fmt(met["ret_6m"], pct=True)),
        ("12m return", _fmt(met["ret_12m"], pct=True)),
        ("RSI(14)", _fmt(met["rsi"])),
        ("Analyst upside", _fmt(met["analyst_upside"], pct=True)),
        ("# analysts", _fmt(met["num_analysts"])),
        ("Rec mean (1=buy)", _fmt(met["recommendation_mean"])),
    ]
    for label, val in rows:
        print(f"  {label:<22}: {val}")

"""The scoring engine: turn metrics into three explainable, ranked buckets.

For each category we use the two-step method from the plan:

1. **Eligibility gate** - a stock only enters a category if it meets that
   category's minimums (keeps the buckets clean and distinct).
2. **Weighted factor score** - each factor is normalized to 0-100 via a
   *cross-sectional percentile rank within the eligible set* (robust to
   outliers), then weighted-summed using the weights from :mod:`config` (which
   the UI exposes as sliders).

Every pick also gets **reason chips** derived from its actual metric values, so
the output explains *why* each name surfaced. Transparency is the product.

Public entry point: :func:`screen`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #
def _pct(series: pd.Series, direction: str) -> pd.Series:
    """Normalize a raw factor series to 0-100.

    direction:
      "high"  -> bigger raw value is better (percentile rank)
      "low"   -> smaller raw value is better (inverted percentile rank)
      "score" -> series is already a 0-100 sub-score; just clip
    Missing values become a neutral 50 ("factor unavailable").
    """
    s = pd.to_numeric(series, errors="coerce")
    if direction == "score":
        return s.clip(0, 100).fillna(config.NEUTRAL_SCORE)
    if s.notna().sum() < 2:
        # Not enough comparables to rank meaningfully.
        return pd.Series(config.NEUTRAL_SCORE, index=series.index)
    ranks = s.rank(pct=True) * 100.0
    if direction == "low":
        ranks = 100.0 - ranks
    return ranks.fillna(config.NEUTRAL_SCORE)


def _pct_sector(df: pd.DataFrame, series: pd.Series, direction: str) -> pd.Series:
    """Percentile rank against sector peers, blended with the universe rank.

    Margins, ROE, leverage and valuation multiples mean different things in
    different industries (software always "wins" gross margin universe-wide;
    banks always "lose" debt/equity), so sector-sensitive factors are ranked
    within their sector and blended 50/50 with the universe-wide rank to damp
    small-group artifacts. Sectors with fewer than ``MIN_SECTOR_PEERS`` known
    values fall back to the universe rank. Disabled via ``SECTOR_RELATIVE``.
    """
    universe = _pct(series, direction)
    if not config.SECTOR_RELATIVE or "sector" not in df.columns:
        return universe
    s = pd.to_numeric(series, errors="coerce")
    sectors = df["sector"].fillna("Unknown").astype(str)
    out = universe.copy()
    for _, idx in s.groupby(sectors).groups.items():
        if s.loc[idx].notna().sum() >= config.MIN_SECTOR_PEERS:
            sector_rank = _pct(s.loc[idx], direction)
            out.loc[idx] = 0.5 * sector_rank + 0.5 * universe.loc[idx]
    return out


def _mean_of_available(parts: list[tuple[pd.Series, pd.Series]],
                       index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """Average (score, available-mask) pairs over the *known* components only.

    Returns (mean, n_known); rows with no known component are NaN in the mean.
    Unlike filling missing components with a neutral 50, this stops sparse data
    from dragging a well-covered name toward the middle — coverage rules are
    applied by the callers instead.
    """
    total = pd.Series(0.0, index=index)
    known = pd.Series(0, index=index)
    for score, avail in parts:
        total = total + score.where(avail, 0.0)
        known = known + avail.astype(int)
    return total / known.replace(0, np.nan), known


def _quality_subscore(df: pd.DataFrame) -> pd.Series:
    """Composite 0-100 quality score: ROE, margin, low debt, positive FCF.

    Sector-relative where it matters, and averaged over the components actually
    known per name; names with fewer than ``QUALITY_MIN_KNOWN`` known inputs get
    a flat neutral score so sparse data can't masquerade as quality.
    """
    roe = pd.to_numeric(df["roe"], errors="coerce")
    margin = pd.to_numeric(df["profit_margin"], errors="coerce")
    de = pd.to_numeric(df["debt_to_equity"], errors="coerce")
    fcf = pd.to_numeric(df["free_cash_flow"], errors="coerce")
    fcf_score = pd.Series(
        np.where(fcf > 0, config.FCF_POSITIVE_SCORE, config.FCF_NEGATIVE_SCORE),
        index=df.index,
    )
    parts = [
        (_pct_sector(df, roe, "high"), roe.notna()),
        (_pct_sector(df, margin, "high"), margin.notna()),
        (_pct_sector(df, de, "low"), de.notna()),
        (fcf_score, fcf.notna()),
    ]
    mean, known = _mean_of_available(parts, df.index)
    mean = mean.where(known >= config.QUALITY_MIN_KNOWN, config.NEUTRAL_SCORE)
    return mean.fillna(config.NEUTRAL_SCORE)


def _momentum_with_rsi_damping(df: pd.DataFrame) -> pd.Series:
    """6-12mo relative strength, with the reward capped when overbought.

    We want to catch names *on the way up*, not at a blow-off peak, so once RSI
    crosses ``RSI_HALVE_REWARD_AT`` the momentum score is pulled back toward the
    neutral 50, and harder still above ``RSI_OVERBOUGHT``.
    """
    momentum_raw = pd.concat(
        [pd.to_numeric(df["ret_6m"], errors="coerce"),
         pd.to_numeric(df["ret_12m"], errors="coerce")],
        axis=1,
    ).mean(axis=1)
    base = _pct(momentum_raw, "high")

    rsi = pd.to_numeric(df["rsi"], errors="coerce")

    def _mult(r):
        if pd.isna(r):
            return 1.0
        if r <= config.RSI_HALVE_REWARD_AT:
            return 1.0
        if r >= config.RSI_OVERBOUGHT:
            return 0.4
        # Linear taper between the two thresholds (1.0 -> 0.5).
        span = config.RSI_OVERBOUGHT - config.RSI_HALVE_REWARD_AT
        return 1.0 - 0.5 * (r - config.RSI_HALVE_REWARD_AT) / span

    mult = rsi.map(_mult)
    return 50.0 + (base - 50.0) * mult


def _drawdown_quality(df: pd.DataFrame) -> pd.Series:
    """Discount from the 52-wk high, rewarded up to a point then penalized.

    A 25-40% pullback in a quality name is the sweet spot; a >60% collapse more
    often signals a broken thesis than a bargain, so we fold deep drops back.
    """
    off = pd.to_numeric(df["off_high"], errors="coerce")
    cap = config.DRAWDOWN_BROKEN_THESIS
    adjusted = off.where(off <= cap, cap - (off - cap))
    return _pct(adjusted, "high")


def _value_vs_growth(df: pd.DataFrame) -> pd.Series:
    """Growth per unit of sales valuation: revenue_growth / P/S (higher better).

    Rewards reasonable valuation *relative to* growth so we favour names that are
    growing fast without already being priced for perfection.
    """
    g = pd.to_numeric(df["revenue_growth"], errors="coerce")
    ps = pd.to_numeric(df["ps_ratio"], errors="coerce")
    raw = g / ps.where(ps > 0)
    return _pct(raw, "high")


def _valuation_multiples(df: pd.DataFrame) -> pd.Series:
    """Cheapness on real multiples: P/E (forward fallback), EV/EBITDA, P/S.

    Each multiple is ranked low-is-better against sector peers; non-positive
    P/E and EV/EBITDA are treated as missing (negative earnings are not
    "cheap"). Averages whatever components a name actually has. Replaces the
    old 52-week range position, which mostly re-measured the drawdown factor
    and never looked at an actual multiple.
    """
    pe = pd.to_numeric(df["pe_trailing"], errors="coerce")
    pe = pe.fillna(pd.to_numeric(df["pe_forward"], errors="coerce"))
    pe = pe.where(pe > 0)
    ev = pd.to_numeric(df["ev_ebitda"], errors="coerce")
    ev = ev.where(ev > 0)
    ps = pd.to_numeric(df["ps_ratio"], errors="coerce")
    ps = ps.where(ps > 0)
    parts = [(_pct_sector(df, s, "low"), s.notna()) for s in (pe, ev, ps)]
    mean, _ = _mean_of_available(parts, df.index)
    return mean.fillna(config.NEUTRAL_SCORE)


def _reasonable_valuation(df: pd.DataFrame) -> pd.Series:
    """Compounder: forward P/E vs sector peers (non-positive treated missing)."""
    pe = pd.to_numeric(df["pe_forward"], errors="coerce")
    return _pct_sector(df, pe.where(pe > 0), "low")


def _dividend_quality(df: pd.DataFrame) -> pd.Series:
    """Dividend yield, sustainability-aware, weighted down when there is none."""
    dy = pd.to_numeric(df["dividend_yield"], errors="coerce")
    payout = pd.to_numeric(df["payout_ratio"], errors="coerce")
    base = _pct(dy, "high")
    # No / negligible dividend -> a low (not zero) score, since a compounder can
    # still compound via buybacks; we just don't reward it on this factor.
    base = base.where(dy.fillna(0) > config.DIVIDEND_YIELD_FLOOR,
                      config.NO_DIVIDEND_SCORE)
    # Penalize unsustainable payout ratios.
    base = base.where(~(payout > config.PAYOUT_UNSUSTAINABLE),
                      base * config.PAYOUT_PENALTY_MULT)
    return base


def _shock_subscore(df: pd.DataFrame) -> pd.Series:
    """Size of the latest one-day shock: a bigger sigma drop scores higher."""
    sigma = pd.to_numeric(df["drop_sigma"], errors="coerce")
    # Negate so a large *negative* sigma (a sharp drop) ranks at the top.
    return _pct(-sigma, "high")


def _short_drawdown_subscore(df: pd.DataFrame) -> pd.Series:
    """Recent 1-2 week decline: a deeper short-window drop scores higher."""
    short = pd.concat(
        [pd.to_numeric(df["ret_1w"], errors="coerce"),
         pd.to_numeric(df["ret_2w"], errors="coerce")],
        axis=1,
    ).mean(axis=1)
    return _pct(-short, "high")


def _below_trend_subscore(df: pd.DataFrame) -> pd.Series:
    """Distance below the 200d MA, rewarded up to a point then folded back.

    Being below trend is the oversold signal we want; but a name that has fallen
    *far* below its 200d (a collapse, not a dip) is more often broken than cheap,
    so beyond ``DIP_DEEP_DROP`` the reward is folded back like the value drawdown.
    """
    below = -pd.to_numeric(df["pct_vs_ma200"], errors="coerce")  # positive == below
    cap = config.DIP_DEEP_DROP
    adjusted = below.where(below <= cap, cap - (below - cap))
    return _pct(adjusted, "high")


def _room_to_grow_subscore(df: pd.DataFrame) -> pd.Series:
    """Moonshot: smaller companies have more room to multiply, so rank smaller
    market caps higher (an inverted percentile within the eligible set)."""
    return _pct(df["market_cap"], "low")


def _analyst_conviction_subscore(df: pd.DataFrame) -> pd.Series:
    """Moonshot: reward upside that is *corroborated by coverage breadth*.

    A blend of the analyst-upside rank and the analyst-count rank, so a high
    target backed by many analysts beats the same target from a lone optimist.
    """
    upside = _pct(df["analyst_upside"], "high")
    coverage = _pct(df["num_analysts"], "high")
    return 0.6 * upside + 0.4 * coverage


# --------------------------------------------------------------------------- #
# Factor specs per category
# Each entry: factor_key -> (callable(df) -> raw Series, direction)
# --------------------------------------------------------------------------- #
def _factor_specs():
    return {
        "growth": {
            "revenue_growth": (lambda d: d["revenue_growth"], "high"),
            "revenue_cagr": (lambda d: d["revenue_cagr"], "high"),
            "gross_margin": (
                lambda d: _pct_sector(d, d["gross_margin"], "high"), "score"),
            "value_vs_growth": (_value_vs_growth, "score"),
            "analyst_upside": (lambda d: d["analyst_upside"], "high"),
            "relative_strength": (_momentum_with_rsi_damping, "score"),
            "earnings_momentum": (lambda d: d["earnings_growth"], "high"),
        },
        "value": {
            "drawdown": (_drawdown_quality, "score"),
            "valuation_vs_history": (_valuation_multiples, "score"),
            "quality": (_quality_subscore, "score"),
            "fundamental_stability": (lambda d: d["revenue_growth"], "high"),
            "analyst_upside": (lambda d: d["analyst_upside"], "high"),
        },
        "compounder": {
            "quality_consistency": (_quality_subscore, "score"),
            "reasonable_valuation": (_reasonable_valuation, "score"),
            "durable_growth": (lambda d: d["revenue_cagr"], "high"),
            "dividend_quality": (_dividend_quality, "score"),
            "low_volatility": (lambda d: d["beta"], "low"),
            "analyst_consensus": (lambda d: d["recommendation_mean"], "low"),
        },
        "dip": {
            "shock": (_shock_subscore, "score"),
            "short_drawdown": (_short_drawdown_subscore, "score"),
            "oversold_rsi": (lambda d: d["rsi"], "low"),
            "below_trend": (_below_trend_subscore, "score"),
            "analyst_upside": (lambda d: d["analyst_upside"], "high"),
            "fundamental_health": (_quality_subscore, "score"),
        },
        "moonshot": {
            "analyst_upside": (lambda d: d["analyst_upside"], "high"),
            "revenue_growth": (lambda d: d["revenue_growth"], "high"),
            "revenue_cagr": (lambda d: d["revenue_cagr"], "high"),
            "room_to_grow": (_room_to_grow_subscore, "score"),
            "relative_strength": (_momentum_with_rsi_damping, "score"),
            "analyst_conviction": (_analyst_conviction_subscore, "score"),
        },
    }


# --------------------------------------------------------------------------- #
# Eligibility gates
# --------------------------------------------------------------------------- #
def _eligible_growth(df: pd.DataFrame) -> pd.Series:
    g = config.GATE_GROWTH
    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    rg = pd.to_numeric(df["revenue_growth"], errors="coerce")
    adv = pd.to_numeric(df["avg_dollar_volume"], errors="coerce")
    return (
        (mc >= g["market_cap_min"]) & (mc <= g["market_cap_max"])
        & (price >= g["price_min"])
        & (rg > g["min_revenue_growth"])
        & (adv >= g["min_avg_dollar_volume"])
    ).fillna(False)


def _eligible_value(df: pd.DataFrame) -> pd.Series:
    g = config.GATE_VALUE
    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    off = pd.to_numeric(df["off_high"], errors="coerce")
    de = pd.to_numeric(df["debt_to_equity"], errors="coerce")
    profitable = df["profitable"].fillna(False).astype(bool)
    cond = (
        (mc >= g["market_cap_min"])
        & (price >= g["price_min"])
        & (off >= g["min_off_high"]) & (off <= g["max_off_high"])
        & profitable
    )
    # Debt gate: only excludes when we actually know debt is extreme.
    cond = cond & ~(de > g["max_debt_to_equity"])
    return cond.fillna(False)


def _eligible_compounder(df: pd.DataFrame) -> pd.Series:
    g = config.GATE_COMPOUNDER
    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    beta = pd.to_numeric(df["beta"], errors="coerce")
    profitable = df["profitable"].fillna(False).astype(bool)
    cond = (
        (mc >= g["market_cap_min"])
        & (price >= g["price_min"])
        & profitable
    )
    # Beta gate: only excludes when beta is known and too high.
    cond = cond & ~(beta > g["max_beta"])
    return cond.fillna(False)


def _eligible_dip(df: pd.DataFrame) -> pd.Series:
    g = config.GATE_DIP
    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    adv = pd.to_numeric(df["avg_dollar_volume"], errors="coerce")
    sigma = pd.to_numeric(df["drop_sigma"], errors="coerce")
    r1w = pd.to_numeric(df["ret_1w"], errors="coerce")
    r2w = pd.to_numeric(df["ret_2w"], errors="coerce")
    rsi = pd.to_numeric(df["rsi"], errors="coerce")
    base = (
        (mc >= g["market_cap_min"])
        & (price >= g["price_min"])
        & (adv >= g["min_avg_dollar_volume"])
    )
    # Deliberately permissive on quality (risks are surfaced, not gated), but the
    # name must actually be dipping/oversold on at least one signal to belong here.
    dipping = (
        (sigma <= g["shock_sigma"])
        | (r1w <= g["short_drop"])
        | (r2w <= g["short_drop"])
        | (rsi <= g["oversold_rsi"])
    )
    return (base & dipping).fillna(False)


def _eligible_moonshot(df: pd.DataFrame) -> pd.Series:
    g = config.GATE_MOONSHOT
    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    adv = pd.to_numeric(df["avg_dollar_volume"], errors="coerce")
    upside = pd.to_numeric(df["analyst_upside"], errors="coerce")
    n_analysts = pd.to_numeric(df["num_analysts"], errors="coerce")
    rg = pd.to_numeric(df["revenue_growth"], errors="coerce")
    # Small/cheap and tradable. Price is the literal "low share price" filter.
    base = (
        (mc >= g["market_cap_min"]) & (mc <= g["market_cap_max"])
        & (price <= g["price_max"])
        & (adv >= g["min_avg_dollar_volume"])
    )
    # "High potential" — must clear at least one real signal. Analyst upside only
    # counts when enough analysts cover it (so one optimist can't game the bucket).
    potential = (
        ((upside >= g["upside_min"]) & (n_analysts >= g["min_analysts"]))
        | (rg >= g["revenue_growth_min"])
    )
    return (base & potential).fillna(False)


_GATES = {
    "growth": _eligible_growth,
    "value": _eligible_value,
    "compounder": _eligible_compounder,
    "dip": _eligible_dip,
    "moonshot": _eligible_moonshot,
}


def gate_report(metrics: dict) -> dict[str, list[str]]:
    """Explain which eligibility conditions a single name fails, per category.

    Returns ``{category: [human-readable failure, ...]}`` — an empty list means
    the name clears that category's gate. Mirrors the vectorized ``_GATES``
    logic including its missing-data behavior: a required-but-unknown metric
    fails (with a "unknown" message), while "only exclude when known" checks
    (value debt, compounder beta) pass when the metric is missing.
    """
    def v(key):
        x = metrics.get(key)
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) else None

    mc, price, adv = v("market_cap"), v("price"), v("avg_dollar_volume")
    report: dict[str, list[str]] = {c: [] for c in config.CATEGORIES}

    def need(category, ok, msg_fail, msg_unknown=None):
        if ok is None:
            report[category].append(msg_unknown or msg_fail)
        elif not ok:
            report[category].append(msg_fail)

    def _known(x, cond):
        return None if x is None else cond

    # --- growth ---
    g = config.GATE_GROWTH
    rg = v("revenue_growth")
    need("growth",
         _known(mc, mc is not None and g["market_cap_min"] <= mc <= g["market_cap_max"]),
         f"market cap {_fmt_cap(mc) if mc else ''} outside "
         f"{_fmt_cap(g['market_cap_min'])}–{_fmt_cap(g['market_cap_max'])}",
         "market cap unknown")
    need("growth", _known(price, price is not None and price >= g["price_min"]),
         f"price below ${g['price_min']:.0f}", "price unknown")
    need("growth", _known(rg, rg is not None and rg > g["min_revenue_growth"]),
         f"revenue not growing ({rg * 100:+.0f}% YoY)" if rg is not None
         else "revenue growth unknown", "revenue growth unknown")
    need("growth", _known(adv, adv is not None and adv >= g["min_avg_dollar_volume"]),
         f"avg dollar volume below {_fmt_cap(g['min_avg_dollar_volume'])}/day",
         "avg dollar volume unknown")

    # --- value ---
    g = config.GATE_VALUE
    off, de = v("off_high"), v("debt_to_equity")
    profitable = bool(metrics.get("profitable"))
    need("value", _known(mc, mc is not None and mc >= g["market_cap_min"]),
         f"market cap below {_fmt_cap(g['market_cap_min'])}", "market cap unknown")
    need("value", _known(price, price is not None and price >= g["price_min"]),
         f"price below ${g['price_min']:.0f}", "price unknown")
    if off is None:
        report["value"].append("distance from 52-wk high unknown")
    elif off < g["min_off_high"]:
        report["value"].append(
            f"only {off * 100:.0f}% below 52-wk high (needs ≥ {g['min_off_high'] * 100:.0f}%)")
    elif off > g["max_off_high"]:
        report["value"].append(
            f"{off * 100:.0f}% below 52-wk high — beyond the "
            f"{g['max_off_high'] * 100:.0f}% broken-thesis cap")
    if not profitable:
        report["value"].append("not profitable")
    if de is not None and de > g["max_debt_to_equity"]:
        report["value"].append(
            f"debt/equity {de:.1f} above {g['max_debt_to_equity']:.1f}")

    # --- compounder ---
    g = config.GATE_COMPOUNDER
    beta = v("beta")
    need("compounder", _known(mc, mc is not None and mc >= g["market_cap_min"]),
         f"market cap below {_fmt_cap(g['market_cap_min'])}", "market cap unknown")
    need("compounder", _known(price, price is not None and price >= g["price_min"]),
         f"price below ${g['price_min']:.0f}", "price unknown")
    if not profitable:
        report["compounder"].append("not profitable")
    if beta is not None and beta > g["max_beta"]:
        report["compounder"].append(f"beta {beta:.2f} above {g['max_beta']:.1f}")

    # --- dip ---
    g = config.GATE_DIP
    sigma, r1w, r2w, rsi = v("drop_sigma"), v("ret_1w"), v("ret_2w"), v("rsi")
    need("dip", _known(mc, mc is not None and mc >= g["market_cap_min"]),
         f"market cap below {_fmt_cap(g['market_cap_min'])}", "market cap unknown")
    need("dip", _known(price, price is not None and price >= g["price_min"]),
         f"price below ${g['price_min']:.0f}", "price unknown")
    need("dip", _known(adv, adv is not None and adv >= g["min_avg_dollar_volume"]),
         f"avg dollar volume below {_fmt_cap(g['min_avg_dollar_volume'])}/day",
         "avg dollar volume unknown")
    dipping = (
        (sigma is not None and sigma <= g["shock_sigma"])
        or (r1w is not None and r1w <= g["short_drop"])
        or (r2w is not None and r2w <= g["short_drop"])
        or (rsi is not None and rsi <= g["oversold_rsi"])
    )
    if not dipping:
        report["dip"].append(
            f"no dip signal (needs a {abs(g['shock_sigma']):.0f}σ down day, "
            f"{g['short_drop'] * 100:.0f}% over 1–2 weeks, or RSI ≤ "
            f"{g['oversold_rsi']:.0f})")

    # --- moonshot ---
    g = config.GATE_MOONSHOT
    upside, n_an = v("analyst_upside"), v("num_analysts")
    need("moonshot",
         _known(mc, mc is not None and g["market_cap_min"] <= mc <= g["market_cap_max"]),
         f"market cap {_fmt_cap(mc) if mc else ''} outside "
         f"{_fmt_cap(g['market_cap_min'])}–{_fmt_cap(g['market_cap_max'])}",
         "market cap unknown")
    need("moonshot", _known(price, price is not None and price <= g["price_max"]),
         f"price above ${g['price_max']:.0f}", "price unknown")
    need("moonshot", _known(adv, adv is not None and adv >= g["min_avg_dollar_volume"]),
         f"avg dollar volume below {_fmt_cap(g['min_avg_dollar_volume'])}/day",
         "avg dollar volume unknown")
    potential = (
        (upside is not None and n_an is not None
         and upside >= g["upside_min"] and n_an >= g["min_analysts"])
        or (v("revenue_growth") is not None
            and v("revenue_growth") >= g["revenue_growth_min"])
    )
    if not potential:
        report["moonshot"].append(
            f"no potential signal (needs analyst upside ≥ "
            f"+{g['upside_min'] * 100:.0f}% with ≥ {g['min_analysts']:.0f} analysts, "
            f"or revenue growth ≥ +{g['revenue_growth_min'] * 100:.0f}% YoY)")

    return report


# --------------------------------------------------------------------------- #
# Penalties (value-trap guard etc.)
# --------------------------------------------------------------------------- #
def _value_trap_multiplier(df: pd.DataFrame) -> pd.Series:
    """Graduated value-trap penalty.

    Revenue AND earnings both shrinking already hurts; shrinking fundamentals
    *plus* a bearish analyst consensus hurts badly. Graduated so a name doesn't
    jump half its score on a single analyst-revision tick.
    """
    rg = pd.to_numeric(df["revenue_growth"], errors="coerce")
    eg = pd.to_numeric(df["earnings_growth"], errors="coerce")
    rec = pd.to_numeric(df["recommendation_mean"], errors="coerce")
    shrinking = ((rg < 0) & (eg < 0)).fillna(False)
    bearish = (rec > config.VALUE_TRAP_REC_MEAN).fillna(False)
    mult = np.where(
        shrinking & bearish, config.VALUE_TRAP_FULL_MULT,
        np.where(shrinking, config.VALUE_TRAP_SHRINKING_MULT, 1.0),
    )
    return pd.Series(mult, index=df.index)


# --------------------------------------------------------------------------- #
# Reason chips
# --------------------------------------------------------------------------- #
def _fmt_cap(v: float) -> str:
    """Compact market-cap string for reason chips ($4.2B / $780M)."""
    if v >= 1e12:
        return f"${v / 1e12:.1f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v / 1e6:.0f}M"


def _reasons(category: str, row: pd.Series) -> list[str]:
    chips: list[Optional[str]] = []
    g = lambda k: row.get(k)

    if category == "growth":
        if g("revenue_growth") is not None:
            chips.append(f"Rev {g('revenue_growth') * 100:+.0f}% YoY")
        if g("revenue_cagr") is not None:
            chips.append(f"3y CAGR {g('revenue_cagr') * 100:+.0f}%")
        if g("gross_margin") is not None and g("gross_margin") > 0.5:
            chips.append(f"Gross margin {g('gross_margin') * 100:.0f}%")
        if g("ps_ratio") is not None:
            chips.append(f"P/S {g('ps_ratio'):.1f}")
        if g("analyst_upside") is not None and g("analyst_upside") > 0.05:
            chips.append(f"Analyst upside {g('analyst_upside') * 100:+.0f}%")
        if g("ret_6m") is not None:
            chips.append(f"6m {g('ret_6m') * 100:+.0f}%")
        if g("rsi") is not None and g("rsi") > config.RSI_OVERBOUGHT:
            chips.append(f"⚠ RSI {g('rsi'):.0f} (hot)")

    elif category == "value":
        if g("off_high") is not None:
            chips.append(f"{g('off_high') * 100:.0f}% below 52-wk high")
        if g("pe_trailing") is not None:
            chips.append(f"P/E {g('pe_trailing'):.0f}")
        if g("roe") is not None and g("roe") > 0.10:
            chips.append(f"ROE {g('roe') * 100:.0f}%")
        if g("debt_to_equity") is not None:
            chips.append(f"D/E {g('debt_to_equity'):.1f}")
        if g("analyst_upside") is not None and g("analyst_upside") > 0.05:
            chips.append(f"Analyst upside {g('analyst_upside') * 100:+.0f}%")
        if g("free_cash_flow") is not None and g("free_cash_flow") > 0:
            chips.append("Positive FCF")

    elif category == "compounder":
        if g("roe") is not None and g("roe") > 0.10:
            chips.append(f"ROE {g('roe') * 100:.0f}%")
        if g("profit_margin") is not None and g("profit_margin") > 0:
            chips.append(f"Net margin {g('profit_margin') * 100:.0f}%")
        if g("dividend_yield") is not None and g("dividend_yield") > 0.001:
            chips.append(f"Yield {g('dividend_yield') * 100:.1f}%")
        if g("pe_forward") is not None:
            chips.append(f"Fwd P/E {g('pe_forward'):.0f}")
        if g("beta") is not None:
            chips.append(f"Beta {g('beta'):.2f}")
        if g("debt_to_equity") is not None and g("debt_to_equity") < 1.0:
            chips.append(f"Low debt (D/E {g('debt_to_equity'):.1f})")

    elif category == "dip":
        if pd.notna(g("ret_1d")):
            chips.append(f"{g('ret_1d') * 100:+.0f}% last day")
        if pd.notna(g("drop_sigma")):
            chips.append(f"{g('drop_sigma'):+.1f}σ move")
        if pd.notna(g("ret_1w")):
            chips.append(f"{g('ret_1w') * 100:+.0f}% 1wk")
        if pd.notna(g("rsi")) and g("rsi") <= 40:
            chips.append(f"RSI {g('rsi'):.0f} (oversold)")
        if pd.notna(g("pct_vs_ma200")) and g("pct_vs_ma200") < 0:
            chips.append(f"{abs(g('pct_vs_ma200')) * 100:.0f}% below 200d MA")
        if pd.notna(g("analyst_upside")) and g("analyst_upside") > 0.05:
            chips.append(f"Analyst upside {g('analyst_upside') * 100:+.0f}%")
        # Risk flags — the whole point of this bucket is to show, not hide, danger.
        prof = g("profitable")
        if prof is not None and not bool(prof):
            chips.append("⚠ Unprofitable")
        if pd.notna(g("debt_to_equity")) and g("debt_to_equity") > 2.0:
            chips.append(f"⚠ High debt (D/E {g('debt_to_equity'):.1f})")
        if pd.notna(g("off_high")) and g("off_high") > config.DRAWDOWN_BROKEN_THESIS:
            chips.append("⚠ 60%+ off high")

    else:  # moonshot
        if pd.notna(g("analyst_upside")) and g("analyst_upside") > 0.05:
            n = g("num_analysts")
            tag = f" ({n:.0f} analysts)" if pd.notna(n) else ""
            chips.append(f"Analyst upside {g('analyst_upside') * 100:+.0f}%{tag}")
        if pd.notna(g("revenue_growth")):
            chips.append(f"Rev {g('revenue_growth') * 100:+.0f}% YoY")
        if pd.notna(g("revenue_cagr")):
            chips.append(f"3y CAGR {g('revenue_cagr') * 100:+.0f}%")
        if pd.notna(g("price")):
            chips.append(f"${g('price'):,.2f} share")
        if pd.notna(g("market_cap")):
            chips.append(f"Mkt cap {_fmt_cap(g('market_cap'))}")
        # Risk flags — speculative names; show the danger rather than hide it.
        prof = g("profitable")
        if prof is not None and not bool(prof):
            chips.append("⚠ Unprofitable")
        if pd.notna(g("debt_to_equity")) and g("debt_to_equity") > 2.0:
            chips.append(f"⚠ High debt (D/E {g('debt_to_equity'):.1f})")

    cap = 8 if category == "dip" else 6 if category == "moonshot" else 5
    return [c for c in chips if c][:cap]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
# Metric columns the scoring engine references that may be absent from an older
# on-disk snapshot (the "dip" signals were added later). Backfill them as NaN so
# the app keeps working — the dip bucket just stays empty until the next rebuild.
_EXPECTED_METRIC_COLS = (
    "ret_1d", "ret_1w", "ret_2w", "daily_vol", "drop_sigma",
    "pct_vs_ma50", "pct_vs_ma200",
)


def metrics_to_frame(metrics_list: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of per-ticker metric dicts."""
    df = pd.DataFrame([m for m in metrics_list if m])
    if not df.empty:
        df = df.drop_duplicates("ticker").set_index("ticker", drop=False)
    for col in _EXPECTED_METRIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def score_category(
    df: pd.DataFrame,
    category: str,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """Score and rank one category. Returns a sorted DataFrame (best first)."""
    if df.empty:
        return df.copy()

    weights = weights or config.DEFAULT_WEIGHTS[category]
    eligible_mask = _GATES[category](df)
    eligible = df[eligible_mask].copy()
    if eligible.empty:
        return eligible

    specs = _factor_specs()[category]
    total_w = sum(weights.get(f, 0) for f in specs) or 1.0

    score = pd.Series(0.0, index=eligible.index)
    factor_scores = {}
    for factor, (fn, direction) in specs.items():
        w = weights.get(factor, 0.0)
        raw = fn(eligible)
        fscore = _pct(raw, direction)
        factor_scores[f"f_{factor}"] = fscore
        score = score + fscore * (w / total_w)

    if category == "value":
        score = score * _value_trap_multiplier(eligible)

    out = eligible.copy()
    for k, v in factor_scores.items():
        out[k] = v.round(1)
    out["score"] = score.clip(0, 100).round(1)
    out["reasons"] = [
        _reasons(category, out.loc[idx]) for idx in out.index
    ]
    out["category"] = category
    return out.sort_values("score", ascending=False)


def screen(
    metrics_list: list[dict],
    weights: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """Run every category screen.

    ``weights`` is an optional {category: {factor: weight}} override (e.g. from
    the UI sliders); falls back to :data:`config.DEFAULT_WEIGHTS`.
    Returns {category: ranked DataFrame}.
    """
    df = metrics_to_frame(metrics_list)
    weights = weights or config.DEFAULT_WEIGHTS
    return {
        cat: score_category(df, cat, weights.get(cat))
        for cat in config.CATEGORIES
    }


if __name__ == "__main__":
    # Smoke test on a small fixed sample (network required).
    from screener import metrics as metrics_mod

    sample = ["NVDA", "MSFT", "AAPL", "KO", "JNJ", "PG", "ASTS", "PLTR",
              "SOFI", "CELH", "PFE", "INTC", "DIS", "TGT", "NKE"]
    mets = []
    for s in sample:
        m = metrics_mod.metrics_for(s)
        if m:
            mets.append(m)

    results = screen(mets)
    for cat in config.CATEGORIES:
        print(f"\n=== {config.CATEGORY_LABELS[cat]} ===")
        r = results[cat]
        if r.empty:
            print("  (no eligible names in sample)")
            continue
        for _, row in r.head(5).iterrows():
            chips = " · ".join(row["reasons"])
            print(f"  {row['ticker']:<6} {row['score']:>5.1f}  {chips}")

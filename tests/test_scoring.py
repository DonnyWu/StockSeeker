"""Unit tests for screener.scoring — synthetic frames, no network."""

import pandas as pd
import pytest

import config
from screener import scoring


def make_row(ticker, **over):
    """A full metric dict (all columns scoring touches) with sane defaults."""
    row = {
        "ticker": ticker, "name": ticker, "sector": "Tech", "source": "sp500",
        "industry": "Software",
        "price": 50.0, "market_cap": 5e9, "beta": 1.0,
        "high_52w": 80.0, "low_52w": 40.0,
        "off_high": 0.30, "range_position": 0.4,
        "avg_volume": 1e6, "avg_dollar_volume": 5e7,
        "revenue_ttm": 1e9, "revenue_growth": 0.10, "earnings_growth": 0.10,
        "revenue_cagr": 0.10, "revenue_acceleration": 0.0,
        "gross_margin": 0.45, "operating_margin": 0.15, "profit_margin": 0.10,
        "free_cash_flow": 1e8, "trailing_eps": 2.0, "profitable": True,
        "roe": 0.15, "roa": 0.07, "debt_to_equity": 0.8, "current_ratio": 1.5,
        "pe_trailing": 18.0, "pe_forward": 15.0, "ps_ratio": 3.0,
        "pb_ratio": 3.0, "ev_ebitda": 12.0, "peg_ratio": 1.5,
        "dividend_yield": 0.02, "payout_ratio": 0.4,
        "ret_1d": 0.0, "ret_1m": 0.02, "ret_3m": 0.05,
        "ret_6m": 0.10, "ret_12m": 0.20,
        "ret_1w": 0.0, "ret_2w": 0.0, "rsi": 55.0,
        "ma50": 49.0, "ma200": 45.0, "golden_cross": True,
        "daily_vol": 0.02, "drop_sigma": 0.0,
        "pct_vs_ma50": 0.02, "pct_vs_ma200": 0.10,
        "target_mean": 60.0, "num_analysts": 10.0,
        "recommendation_mean": 2.0, "recommendation_key": "buy",
        "analyst_upside": 0.20,
    }
    row.update(over)
    return row


def frame(*rows):
    return scoring.metrics_to_frame(list(rows))


def fillers(n=6, **over):
    return [make_row(f"FILL{i}", **over) for i in range(n)]


# --------------------------------------------------------------------------- #
# Eligibility gates
# --------------------------------------------------------------------------- #
def test_gates_membership():
    rows = frame(
        make_row("GRW", market_cap=2e9, revenue_growth=0.30),
        make_row("BIG", market_cap=50e9, beta=0.9),          # too big for growth
        make_row("NEAR", off_high=0.05),                     # too close to high for value
        make_row("DIPPY", rsi=28.0, ret_1w=-0.12),           # oversold + dropping
        make_row("MOON", price=10.0, market_cap=1e9,
                 analyst_upside=0.50, num_analysts=6.0),
        make_row("LOSS", profitable=False, market_cap=20e9),
    )
    assert scoring._eligible_growth(rows)["GRW"]
    assert not scoring._eligible_growth(rows)["BIG"]
    assert not scoring._eligible_value(rows)["NEAR"]
    assert not scoring._eligible_value(rows)["LOSS"]
    assert scoring._eligible_compounder(rows)["BIG"]
    assert not scoring._eligible_compounder(rows)["LOSS"]
    assert scoring._eligible_dip(rows)["DIPPY"]
    assert not scoring._eligible_dip(rows)["GRW"]        # no dip signal
    assert scoring._eligible_moonshot(rows)["MOON"]
    assert not scoring._eligible_moonshot(rows)["GRW"]   # price > $30


# --------------------------------------------------------------------------- #
# gate_report
# --------------------------------------------------------------------------- #
def test_gate_report_pass_is_empty():
    good = make_row("GOOD", market_cap=12e9, beta=0.9)
    rep = scoring.gate_report(good)
    assert rep["compounder"] == []


def test_gate_report_explains_failures():
    small = make_row("SMALL", market_cap=1e8)
    rep = scoring.gate_report(small)
    assert any("market cap" in f for f in rep["growth"])
    assert any("no dip signal" in f for f in rep["dip"])
    assert any("price above" in f for f in rep["moonshot"])

    unknown = make_row("UNK", market_cap=None)
    rep = scoring.gate_report(unknown)
    assert any("unknown" in f for f in rep["growth"])


def test_gate_report_lenient_when_optional_metrics_missing():
    # Value debt / compounder beta only exclude when *known* to be too high.
    row = make_row("NODATA", market_cap=12e9, beta=None, debt_to_equity=None)
    rep = scoring.gate_report(row)
    assert rep["compounder"] == []
    assert rep["value"] == []


# --------------------------------------------------------------------------- #
# Quality subscore
# --------------------------------------------------------------------------- #
def test_quality_subscore_sparse_data_is_neutral():
    sparse = make_row("SPARSE", roe=0.90, profit_margin=None,
                      debt_to_equity=None, free_cash_flow=None)
    df = frame(sparse, *fillers())
    q = scoring._quality_subscore(df)
    assert q["SPARSE"] == pytest.approx(config.NEUTRAL_SCORE)


def test_quality_subscore_orders_good_above_bad():
    good = make_row("GOOD", roe=0.30, profit_margin=0.25,
                    debt_to_equity=0.2, free_cash_flow=5e8)
    bad = make_row("BAD", roe=0.02, profit_margin=0.01,
                   debt_to_equity=3.5, free_cash_flow=-1e8)
    df = frame(good, bad, *fillers())
    q = scoring._quality_subscore(df)
    assert q["GOOD"] > q["BAD"]


# --------------------------------------------------------------------------- #
# Valuation multiples (the new value factor)
# --------------------------------------------------------------------------- #
def test_valuation_multiples_cheap_beats_rich():
    cheap = make_row("CHEAP", pe_trailing=8.0, ev_ebitda=6.0, ps_ratio=1.0)
    rich = make_row("RICH", pe_trailing=80.0, ev_ebitda=40.0, ps_ratio=15.0)
    df = frame(cheap, rich, *fillers())
    v = scoring._valuation_multiples(df)
    assert v["CHEAP"] > v["RICH"]


def test_valuation_multiples_negative_earnings_not_cheap():
    neg = make_row("NEG", pe_trailing=-5.0, pe_forward=-3.0,
                   ev_ebitda=-2.0, ps_ratio=None)
    df = frame(neg, *fillers())
    v = scoring._valuation_multiples(df)
    assert v["NEG"] == pytest.approx(config.NEUTRAL_SCORE)


# --------------------------------------------------------------------------- #
# Sector-relative ranking
# --------------------------------------------------------------------------- #
def test_pct_sector_credits_best_in_a_disadvantaged_sector():
    # Banks structurally carry more leverage; the best-levered bank should rank
    # better sector-relative than it does against the whole universe.
    banks = [make_row(f"B{i}", sector="Financials",
                      debt_to_equity=2.0 + 0.1 * i) for i in range(6)]
    techs = [make_row(f"T{i}", sector="Tech",
                      debt_to_equity=0.2 + 0.1 * i) for i in range(6)]
    df = frame(*(banks + techs))
    universe = scoring._pct(df["debt_to_equity"], "low")
    blended = scoring._pct_sector(df, df["debt_to_equity"], "low")
    assert blended["B0"] > universe["B0"]


def test_pct_sector_falls_back_for_tiny_sectors():
    banks = [make_row(f"B{i}", sector="Financials",
                      debt_to_equity=2.0 + 0.1 * i) for i in range(3)]
    techs = [make_row(f"T{i}", sector="Tech",
                      debt_to_equity=0.2 + 0.1 * i) for i in range(6)]
    df = frame(*(banks + techs))
    universe = scoring._pct(df["debt_to_equity"], "low")
    blended = scoring._pct_sector(df, df["debt_to_equity"], "low")
    # Only 3 banks (< MIN_SECTOR_PEERS) -> banks keep their universe rank.
    for t in ("B0", "B1", "B2"):
        assert blended[t] == pytest.approx(universe[t])


def test_pct_sector_disabled_matches_universe(monkeypatch):
    monkeypatch.setattr(config, "SECTOR_RELATIVE", False)
    rows = [make_row(f"R{i}", gross_margin=0.2 + 0.1 * i) for i in range(8)]
    df = frame(*rows)
    pd.testing.assert_series_equal(
        scoring._pct_sector(df, df["gross_margin"], "high"),
        scoring._pct(df["gross_margin"], "high"),
    )


# --------------------------------------------------------------------------- #
# Value-trap multiplier (graduated)
# --------------------------------------------------------------------------- #
def test_value_trap_multiplier_tiers():
    healthy = make_row("OK")
    shrinking = make_row("SHRINK", revenue_growth=-0.10, earnings_growth=-0.20,
                         recommendation_mean=2.0)
    trap = make_row("TRAP", revenue_growth=-0.10, earnings_growth=-0.20,
                    recommendation_mean=3.5)
    df = frame(healthy, shrinking, trap)
    mult = scoring._value_trap_multiplier(df)
    assert mult["OK"] == pytest.approx(1.0)
    assert mult["SHRINK"] == pytest.approx(config.VALUE_TRAP_SHRINKING_MULT)
    assert mult["TRAP"] == pytest.approx(config.VALUE_TRAP_FULL_MULT)


# --------------------------------------------------------------------------- #
# score_category end-to-end
# --------------------------------------------------------------------------- #
def test_value_ranking_prefers_quality_over_trap():
    quality = make_row(
        "QLT", off_high=0.30, pe_trailing=9.0, ev_ebitda=6.0, ps_ratio=1.2,
        roe=0.25, profit_margin=0.18, debt_to_equity=0.4, free_cash_flow=1e9,
        revenue_growth=0.08, earnings_growth=0.10,
        recommendation_mean=1.8, analyst_upside=0.30)
    trap = make_row(
        "TRP", off_high=0.55, pe_trailing=35.0, ev_ebitda=25.0, ps_ratio=8.0,
        roe=0.03, profit_margin=0.02, debt_to_equity=2.5, free_cash_flow=-2e8,
        revenue_growth=-0.10, earnings_growth=-0.20,
        recommendation_mean=3.6, analyst_upside=0.05)
    ranked = scoring.score_category(
        frame(quality, trap, *fillers(off_high=0.25)), "value")
    assert "QLT" in ranked.index and "TRP" in ranked.index
    assert ranked.index.get_loc("QLT") < ranked.index.get_loc("TRP")


def test_dip_scores_fundamental_health():
    dippers = [make_row(f"D{i}", rsi=25.0 + i, ret_1w=-0.12) for i in range(6)]
    ranked = scoring.score_category(frame(*dippers), "dip")
    assert not ranked.empty
    assert "f_fundamental_health" in ranked.columns


def test_score_category_weights_need_not_sum_to_one():
    rows = [make_row(f"R{i}", market_cap=2e9,
                     revenue_growth=0.05 + 0.05 * i) for i in range(6)]
    tripled = {k: v * 3 for k, v in config.WEIGHTS_GROWTH.items()}
    ranked = scoring.score_category(frame(*rows), "growth", weights=tripled)
    assert len(ranked) == 6
    assert ranked["score"].between(0, 100).all()


def test_screen_returns_every_category():
    results = scoring.screen([make_row("ONLY")])
    assert set(results.keys()) == set(config.CATEGORIES)

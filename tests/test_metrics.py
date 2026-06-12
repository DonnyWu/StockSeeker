"""Unit tests for screener.metrics — pure computation, no network."""

import numpy as np
import pytest

from screener.metrics import (
    _norm_ratio,
    _norm_yield,
    _pct_change,
    _rsi,
    _sma,
    compute_metrics,
)


def make_payload(closes, info=None, volumes=None, annual_revenue=None):
    return {
        "ticker": "TST",
        "info": info or {},
        "history": {
            "dates": [f"d{i}" for i in range(len(closes))],
            "close": list(closes),
            "volume": volumes if volumes is not None else [1_000_000] * len(closes),
        },
        "annual_revenue": annual_revenue or [],
    }


# --------------------------------------------------------------------------- #
# RSI (Wilder smoothing)
# --------------------------------------------------------------------------- #
def test_rsi_all_gains_is_100():
    assert _rsi(list(range(1, 20))) == 100.0


def test_rsi_all_losses_is_0():
    assert _rsi(list(range(40, 10, -1))) == pytest.approx(0.0)


def test_rsi_balanced_changes_is_50():
    closes = [100.0]
    for i in range(14):
        closes.append(closes[-1] + (1.0 if i % 2 == 0 else -1.0))
    assert _rsi(closes) == pytest.approx(50.0)


def test_rsi_insufficient_history_is_none():
    assert _rsi([1.0] * 14) is None
    assert _rsi([]) is None


def test_rsi_smooths_over_full_history():
    # A crash 10+ bars before a flat tail must still depress RSI. The old
    # last-14-diffs-only version saw an all-flat window (no losses) and
    # returned 100 — Wilder smoothing keeps the loss in the average.
    closes = [100.0] * 10 + [50.0] + [50.0] * 14
    rsi = _rsi(closes)
    assert rsi is not None
    assert rsi < 30


def test_rsi_ignores_none_values():
    closes = [None] + list(range(1, 20)) + [None]
    assert _rsi(closes) == 100.0


# --------------------------------------------------------------------------- #
# Returns / SMA
# --------------------------------------------------------------------------- #
def test_pct_change_normal_window():
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert _pct_change(closes, 5) == pytest.approx(16.0 / 11.0 - 1.0)


def test_pct_change_falls_back_to_oldest():
    assert _pct_change([10.0, 12.0, 15.0], 252) == pytest.approx(0.5)


def test_pct_change_no_usable_history():
    assert _pct_change([10.0], 5) is None
    assert _pct_change([0.0, 5.0, 6.0], 5) is None


def test_sma():
    assert _sma([1.0, 2.0, 3.0, 4.0], 2) == pytest.approx(3.5)
    assert _sma([1.0, 2.0], 3) is None


# --------------------------------------------------------------------------- #
# yfinance unit normalizers
# --------------------------------------------------------------------------- #
def test_norm_ratio_percent_vs_fraction():
    assert _norm_ratio(154.0) == pytest.approx(1.54)
    assert _norm_ratio(1.2) == pytest.approx(1.2)
    assert _norm_ratio(None) is None


def test_norm_yield_percent_vs_fraction():
    assert _norm_yield(3.0) == pytest.approx(0.03)
    assert _norm_yield(0.0035) == pytest.approx(0.0035)
    assert _norm_yield(None) is None


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_price_range_and_analyst():
    info = {
        "longName": "Test Corp",
        "sector": "Information Technology",
        "currentPrice": 80.0,
        "marketCap": 5e9,
        "fiftyTwoWeekHigh": 100.0,
        "fiftyTwoWeekLow": 50.0,
        "profitMargins": 0.12,
        "trailingEps": 2.0,
        "debtToEquity": 154.0,
        "dividendYield": 3.0,
        "targetMeanPrice": 96.0,
        "numberOfAnalystOpinions": 12,
    }
    m = compute_metrics(make_payload([80.0] * 30, info=info))
    assert m["off_high"] == pytest.approx(0.20)
    assert m["range_position"] == pytest.approx(0.60)
    assert m["profitable"] is True
    assert m["debt_to_equity"] == pytest.approx(1.54)
    assert m["dividend_yield"] == pytest.approx(0.03)
    assert m["analyst_upside"] == pytest.approx(0.20)


def test_compute_metrics_unprofitable():
    info = {"profitMargins": -0.10, "trailingEps": -1.0, "currentPrice": 10.0}
    m = compute_metrics(make_payload([10.0] * 30, info=info))
    assert m["profitable"] is False


def test_compute_metrics_revenue_cagr_and_acceleration():
    m = compute_metrics(
        make_payload([10.0] * 30, annual_revenue=[100.0, 110.0, 132.0]))
    assert m["revenue_cagr"] == pytest.approx((132.0 / 100.0) ** 0.5 - 1.0)
    assert m["revenue_acceleration"] == pytest.approx(0.20 - 0.10)

    m = compute_metrics(make_payload([10.0] * 30, annual_revenue=[100.0]))
    assert m["revenue_cagr"] is None


def test_drop_sigma_excludes_the_shock_day():
    closes = [100.0]
    for i in range(60):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    closes.append(closes[-1] * 0.90)  # -10% shock on the latest bar

    m = compute_metrics(make_payload(closes))
    arr = np.asarray(closes)
    rets = arr[1:] / arr[:-1] - 1.0
    expected_vol = float(np.std(rets[:-1][-252:]))

    assert m["daily_vol"] == pytest.approx(expected_vol)
    assert m["drop_sigma"] == pytest.approx(m["ret_1d"] / expected_vol)
    # Against its own prior ~1% noise, a -10% day is an extreme shock.
    assert m["drop_sigma"] < -5

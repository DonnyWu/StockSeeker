"""Unit tests for screener.history — temp paths, no network."""

import pandas as pd
import pytest

from screener import history


def _ranked(tickers_prices):
    return pd.DataFrame([
        {"ticker": t, "name": f"{t} Corp", "score": 90.0 - i, "price": p}
        for i, (t, p) in enumerate(tickers_prices)
    ]).set_index("ticker", drop=False)


@pytest.fixture(autouse=True)
def hist_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history, "PICKS_CSV", tmp_path / "picks_history.csv")
    return tmp_path


def test_record_picks_writes_and_loads():
    out = history.record_picks({"growth": _ranked([("AAA", 10.0), ("BBB", 20.0)])})
    assert len(out) == 2
    assert list(out["rank"]) == [1, 2]
    reloaded = history.load_history()
    assert len(reloaded) == 2


def test_record_picks_upserts_same_day():
    results = {"growth": _ranked([("AAA", 10.0), ("BBB", 20.0)])}
    history.record_picks(results)
    out = history.record_picks(results)  # refresh again the same day
    assert len(out) == 2  # overwritten, not duplicated


def test_record_picks_preserves_other_categories():
    history.record_picks({"growth": _ranked([("AAA", 10.0)])})
    out = history.record_picks({"value": _ranked([("CCC", 30.0)])})
    assert set(out["category"]) == {"growth", "value"}


def test_record_picks_caps_top_n():
    many = _ranked([(f"T{i:02d}", 10.0 + i) for i in range(history.TOP_N + 5)])
    out = history.record_picks({"growth": many})
    assert len(out) == history.TOP_N


def test_load_history_empty_when_missing():
    out = history.load_history()
    assert out.empty
    assert list(out.columns) == history._COLUMNS


def test_performance_frame_joins_and_handles_missing():
    hist = history.record_picks(
        {"growth": _ranked([("AAA", 10.0), ("GONE", 5.0)])})
    snapshot = pd.DataFrame([{"ticker": "AAA", "price": 12.0}])
    perf = history.performance_frame(hist, snapshot)

    aaa = perf[perf["ticker"] == "AAA"].iloc[0]
    gone = perf[perf["ticker"] == "GONE"].iloc[0]
    assert aaa["return_pct"] == pytest.approx(0.20)
    assert aaa["days_held"] == 0  # picked today
    assert pd.isna(gone["current_price"])
    assert pd.isna(gone["return_pct"])


def test_performance_frame_empty_history():
    perf = history.performance_frame(history.load_history(), None)
    assert perf.empty

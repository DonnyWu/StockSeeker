"""StockSeeker - a transparent multi-factor stock screener.

Three explainable buckets (High Growth / Value-on-Sale / Steady Compounders),
each a ranked table with the *reasons* behind every pick and a per-stock detail
view. Data is the on-disk snapshot built by ``refresh.py``; the user can rebuild
it from the sidebar. Scoring is recomputed live as the weight sliders move, so
nothing is a black box.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import config
import refresh
from screener import fetch, scoring, scrape

st.set_page_config(page_title="StockSeeker", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Friendly factor labels for the detail breakdown
# --------------------------------------------------------------------------- #
FACTOR_LABELS = {
    "f_revenue_growth": "Revenue growth",
    "f_revenue_cagr": "3-yr CAGR",
    "f_gross_margin": "Gross margin",
    "f_value_vs_growth": "Value vs growth",
    "f_analyst_upside": "Analyst upside",
    "f_relative_strength": "Relative strength",
    "f_revisions_insider": "Earnings momentum",
    "f_drawdown": "Discount from high",
    "f_valuation_vs_history": "Cheap vs own range",
    "f_quality": "Quality",
    "f_fundamental_stability": "Fundamental stability",
    "f_quality_consistency": "Quality & consistency",
    "f_reasonable_valuation": "Reasonable valuation",
    "f_durable_growth": "Durable growth",
    "f_dividend_quality": "Dividend quality",
    "f_low_volatility": "Low volatility",
    "f_analyst_consensus": "Analyst consensus",
    "f_shock": "Shock (σ down-day)",
    "f_short_drawdown": "1–2 week decline",
    "f_oversold_rsi": "Oversold (RSI)",
    "f_below_trend": "Below 200d trend",
    "f_room_to_grow": "Room to grow (small cap)",
    "f_analyst_conviction": "Analyst conviction",
}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_snapshot_cached(token: float):
    """Load the snapshot. ``token`` busts the cache after a refresh.

    NB: the parameter must *not* start with an underscore — Streamlit excludes
    underscore-prefixed args from the cache key, which would make the bust a
    no-op and freeze the snapshot for the whole session.
    """
    return refresh.load_snapshot()


@st.cache_data(show_spinner=False)
def _enrich_cached(ticker: str, token: float):
    return scrape.enrich(ticker)


def _snapshot_token() -> float:
    return st.session_state.get("snapshot_token", 0.0)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(meta) -> dict:
    st.sidebar.title("📈 StockSeeker")

    if meta and meta.get("built_at_iso"):
        age_h = (time.time() - meta.get("built_at", 0)) / 3600.0
        st.sidebar.caption(
            f"Snapshot: {meta['built_at_iso']}  ·  {meta.get('n_tickers', 0)} "
            f"tickers  ·  {age_h:.0f}h old"
        )
    else:
        st.sidebar.caption("No snapshot yet — build one below.")

    universe_choice = st.sidebar.radio(
        "Universe to build",
        ["S&P 500 + Growth", "S&P 500 only", "Growth seed only"],
        index=0,
        help="Affects what 'Refresh data' fetches. Growth-only is fastest.",
    )

    if st.sidebar.button("🔄 Refresh data", width="stretch",
                         help="Re-fetch the chosen universe from yfinance."):
        _run_refresh(universe_choice)

    st.sidebar.divider()

    # Display filters (applied without re-fetching).
    st.sidebar.subheader("Filters")
    mc_min, mc_max = st.sidebar.slider(
        "Market cap ($B)", 0.0, 4000.0, (0.0, 4000.0), step=10.0,
    )
    source_filter = st.sidebar.multiselect(
        "Source list", ["sp500", "growth", "both"],
        default=["sp500", "growth", "both"],
        help="Where a ticker came from in the universe.",
    )

    # Advanced weight sliders.
    weights = {}
    with st.sidebar.expander("⚙️ Advanced: factor weights"):
        st.caption("Weights are normalized per category, so they need not sum "
                   "to 1. Reset by reloading the page.")
        for cat in config.CATEGORIES:
            st.markdown(f"**{config.CATEGORY_LABELS[cat]}**")
            weights[cat] = {}
            for factor, default in config.DEFAULT_WEIGHTS[cat].items():
                label = FACTOR_LABELS.get(f"f_{factor}", factor)
                weights[cat][factor] = st.slider(
                    label, 0.0, 1.0, float(default), step=0.05,
                    key=f"w_{cat}_{factor}",
                )

    return {
        "mc_min": mc_min * 1e9,
        "mc_max": mc_max * 1e9,
        "sources": source_filter,
        "weights": weights,
    }


def _run_refresh(universe_choice: str):
    include_sp500 = universe_choice != "Growth seed only"
    include_growth = universe_choice != "S&P 500 only"

    bar = st.sidebar.progress(0.0, text="Fetching…")

    def _progress(done, total):
        bar.progress(min(done / max(total, 1), 1.0), text=f"Fetching {done}/{total}")

    with st.spinner("Building snapshot — this can take a few minutes…"):
        df = refresh.compute_snapshot(
            include_sp500=include_sp500,
            include_growth=include_growth,
            force=True,
            progress=_progress,
        )
        refresh.save_snapshot(df)
    bar.empty()
    # Bust the snapshot cache so the rerun re-reads the file we just wrote.
    _load_snapshot_cached.clear()
    st.session_state["snapshot_token"] = time.time()

    stats = df.attrs.get("fetch_stats", {})
    fresh, stale = stats.get("fresh", 0), stats.get("stale", 0)
    if fresh == 0 and stale:
        st.sidebar.warning(
            f"Refreshed {len(df)} tickers, but all came from cache — the data "
            "source may be rate-limiting. Numbers are unchanged; try again shortly."
        )
    elif stale:
        st.sidebar.success(
            f"Refreshed {len(df)} tickers ({fresh} fresh, {stale} from cache)."
        )
    else:
        st.sidebar.success(f"Refreshed {len(df)} tickers (all fresh).")
    st.rerun()


# --------------------------------------------------------------------------- #
# Filtering + scoring
# --------------------------------------------------------------------------- #
def _apply_filters(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    out = df.copy()
    mc = pd.to_numeric(out.get("market_cap"), errors="coerce")
    out = out[(mc.fillna(0) >= controls["mc_min"]) & (mc.fillna(0) <= controls["mc_max"])]
    if "source" in out.columns and controls["sources"]:
        out = out[out["source"].isin(controls["sources"])]
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_b(v):
    if v is None or pd.isna(v):
        return "—"
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"


def _fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.0f}%"


def render_movers(df: pd.DataFrame):
    """Top-of-page strip surfacing the biggest unusual recent drops, universe-wide.

    Independent of the category buckets — its job is simply to make a fresh,
    outsized drop impossible to miss the moment you open the app.
    """
    sigma = pd.to_numeric(df.get("drop_sigma"), errors="coerce")
    movers = df.assign(_sigma=sigma)
    movers = movers[movers["_sigma"] <= config.MOVERS_SIGMA]
    if movers.empty:
        return
    movers = movers.nsmallest(config.MOVERS_TOP_N, "_sigma")

    with st.container(border=True):
        st.markdown("#### 🩸 Big movers — unusual recent drops")
        st.caption(
            f"Names that fell at least {abs(config.MOVERS_SIGMA):.0f}σ in their "
            "latest session, across the whole universe. See the **🩸 Sharp Drops** "
            "tab to rank and inspect them."
        )
        cols = st.columns(min(len(movers), 4))
        for i, (_, r) in enumerate(movers.iterrows()):
            col = cols[i % len(cols)]
            price, r1d, sig = r.get("price"), r.get("ret_1d"), r.get("_sigma")
            delta = None
            if pd.notna(r1d):
                delta = f"{r1d * 100:+.1f}%"
                if pd.notna(sig):
                    delta += f"  ({sig:+.1f}σ)"
            col.metric(
                label=str(r.get("ticker", "—")),
                value=f"${price:,.2f}" if pd.notna(price) else "—",
                delta=delta,
                help=str(r.get("name", "")),
            )


def render_table(ranked: pd.DataFrame, category: str):
    if ranked.empty:
        st.info("No stocks cleared this category's eligibility gate with the "
                "current filters. Try widening the market-cap range or refresh.")
        return

    view = pd.DataFrame({
        "Rank": range(1, len(ranked) + 1),
        "Ticker": ranked["ticker"].values,
        "Name": ranked["name"].values,
        "Score": ranked["score"].values,
        "Price": [f"${p:,.2f}" if pd.notna(p) else "—" for p in ranked["price"]],
        "Mkt Cap": [_fmt_b(v) for v in ranked["market_cap"]],
        "Analyst ▲": [_fmt_pct(v) for v in ranked["analyst_upside"]],
        "Why": [" · ".join(r) for r in ranked["reasons"]],
    })

    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=min(560, 56 + 35 * len(view)),
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f"),
            "Why": st.column_config.TextColumn("Why it surfaced", width="large"),
        },
    )

    # Detail view selector.
    options = ranked["ticker"].tolist()
    chosen = st.selectbox(
        "🔍 Inspect a pick", options, key=f"detail_{category}",
        format_func=lambda t: f"{t} — {ranked.loc[t, 'name']}"
        if t in ranked.index else t,
    )
    if chosen:
        render_detail(ranked.loc[chosen], category)


def render_detail(row: pd.Series, category: str):
    ticker = row["ticker"]
    st.markdown(f"### {ticker} — {row['name']}")
    sector = row.get("sector") or "—"
    st.caption(f"{sector}  ·  Score **{row['score']:.0f}/100** in "
               f"{config.CATEGORY_LABELS[category]}")

    if row.get("reasons"):
        st.markdown(" ".join(f"`{c}`" for c in row["reasons"]))

    left, right = st.columns([3, 2])

    # --- price chart (from cached history, no network) ---
    with left:
        st.markdown("**Price — last 12 months**")
        payload = fetch.peek_cache(ticker)
        hist = (payload or {}).get("history", {})
        dates, closes = hist.get("dates", []), hist.get("close", [])
        if dates and closes:
            chart_df = pd.DataFrame({"Close": closes},
                                    index=pd.to_datetime(dates))
            st.line_chart(chart_df, height=260)
        else:
            st.caption("No cached price history. Refresh to populate.")

    # --- factor breakdown ---
    with right:
        st.markdown("**Factor breakdown (0–100)**")
        factor_cols = [c for c in row.index if c.startswith("f_")]
        if factor_cols:
            fb = pd.DataFrame({
                "Factor": [FACTOR_LABELS.get(c, c) for c in factor_cols],
                "Score": [row[c] for c in factor_cols],
            }).set_index("Factor")
            st.bar_chart(fb, height=260, horizontal=True)

    # --- key metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market cap", _fmt_b(row.get("market_cap")))
    c2.metric("Rev growth YoY", _fmt_pct(row.get("revenue_growth")))
    c3.metric("Analyst upside", _fmt_pct(row.get("analyst_upside")))
    rec = row.get("recommendation_mean")
    c4.metric("Analyst rating", f"{rec:.2f}/5" if pd.notna(rec) else "—",
              help="1 = strong buy … 5 = sell")

    # --- live analyst + news sentiment (optional, cached) ---
    with st.expander("📰 Live sentiment — news tone + analysts "
                     "(Yahoo / Finviz / Finnhub / FMP)"):
        enr = _enrich_cached(ticker, _snapshot_token())
        news = enr.get("news_sentiment")
        if news:
            _emoji = {"Bullish": "🟢", "Bearish": "🔴",
                      "Neutral": "⚪", "Unscored": "⚪"}.get(news["label"], "⚪")
            score = news.get("score")
            tone = (f"{_emoji} **Recent news tone: {news['label']}** "
                    f"({score:+.2f}) " if score is not None
                    else f"{_emoji} **Recent news tone: {news['label']}** ")
            st.markdown(tone + f"· {news['n']} headlines")
            for h in news["headlines"]:
                title, link = h.get("title"), h.get("link")
                pub = h.get("publisher") or ""
                bullet = f"[{title}]({link})" if link else title
                if pub:
                    bullet += f"  — *{pub}*"
                st.markdown(f"- {bullet}")
        if enr.get("finviz"):
            st.write("**Finviz:**", enr["finviz"])
        if enr.get("finnhub"):
            st.write("**Finnhub recommendation trend:**", enr["finnhub"])
        if enr.get("fmp"):
            st.write("**FMP scores:**", enr["fmp"])
        if not any(enr.values()):
            st.caption("No live sentiment available (sources unreachable or no "
                       "API keys configured).")

    # --- source links ---
    st.markdown(
        f"**Sources:** "
        f"[Yahoo](https://finance.yahoo.com/quote/{ticker}) · "
        f"[Finviz](https://finviz.com/quote.ashx?t={ticker}) · "
        f"[StockAnalysis](https://stockanalysis.com/stocks/{ticker}/)"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df, meta = _load_snapshot_cached(_snapshot_token())
    controls = sidebar(meta)

    st.title("StockSeeker")
    st.warning(config.DISCLAIMER)

    if df is None or df.empty:
        st.info("No data yet. Use **🔄 Refresh data** in the sidebar to build "
                "the first snapshot. *(Growth seed only* is the quickest start — "
                "about a minute.)")
        return

    filtered = _apply_filters(df, controls)
    render_movers(filtered)
    results = scoring.screen(refresh.metrics_records(filtered), controls["weights"])

    tabs = st.tabs([config.CATEGORY_LABELS[c] for c in config.CATEGORIES])
    blurbs = {
        "growth": "Small/mid-caps growing fast at a reasonable price — caught "
                  "*before* they get expensive. Momentum reward is capped so we "
                  "favour names on the way up, not at the peak.",
        "value": "Quality companies that fell hard and now trade at a discount "
                 "to their own recent range, with a value-trap guard.",
        "compounder": "Large, profitable, lower-beta names for buy-and-hold / "
                      "dollar-cost-averaging.",
        "dip": "Names that just dropped hard or look oversold (shock + oversold "
               "blend), shown *with their risk flags* so you can judge an "
               "overreaction from a broken thesis. Not gated on profitability.",
        "moonshot": "Low-priced (≤ $30/share), small/mid-cap speculative names "
                    "with big analyst upside and/or fast growth — the "
                    "lottery-ticket profile (think ACHR/JOBY), ranked by upside + "
                    "growth and shown *with risk flags*. Not gated on profitability.",
    }
    for tab, cat in zip(tabs, config.CATEGORIES):
        with tab:
            st.caption(blurbs[cat])
            render_table(results[cat], cat)


if __name__ == "__main__":
    main()

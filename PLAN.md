# StockSeeker — Stock Recommendation App — Implementation Plan

## Context

A simple, no-database web app that surfaces stock ideas into three buckets:

1. **High growth potential** — small/mid-cap long-term plays to catch *before they get expensive* (ASTS, PL, STEM a year ago).
2. **Value on sale** — quality companies that dropped hard and are now discounted (MSFT-type pullbacks).
3. **Everyday compounders** — steady, buy-and-hold names for long-term dollar-cost-averaging.

Constraints: only **free APIs + light web scraping** for analyst opinions, and **no database**.

**Honest framing that shapes the design:** reliably *predicting* winners with free data + ML is not realistic. What *is* achievable and genuinely useful is a **transparent multi-factor screener** (the factor-investing approach — growth, value, quality, momentum, size — used by real quant screens) that ranks candidates into the three buckets and **shows the reasons** for each pick. The value is the explainable shortlist, not a black-box oracle. A prominent "not financial advice / educational" disclaimer is part of the build.

**Decisions locked in:**
- Framework: **Streamlit** (pure-Python, fastest path, free to deploy on Streamlit Community Cloud).
- Universe: **S&P 500 + a curated small/mid-cap growth seed list** (~600–900 tickers).
- Engine: **Rules-based factor scoring** (0–100 per category, fully explainable).
- Data: **APIs first + light scraping** — yfinance/FMP/Finnhub for the numbers, light Finviz/stockanalysis scraping only for extra analyst-sentiment color.

---

## Tech stack

- **Python 3.11+**, **Streamlit** (UI), **pandas/numpy** (compute).
- **yfinance** — primary free source (prices, fundamentals, analyst targets/recommendations; no API key). Note: 2026 yfinance uses `curl_cffi` sessions to dodge rate-limiting — pin a recent version.
- **requests + beautifulsoup4 (lxml)** — light scraping; **requests-cache** to throttle/cache HTTP.
- **python-dotenv** — load optional free API keys (FMP, Finnhub) from `.env`.
- **pyarrow** — persist the screen snapshot as a parquet/JSON file (this is file caching, **not** a database).
- Charts via Streamlit's built-in `st.line_chart` (add `plotly` only if richer detail charts are wanted).

### Free data sources & their role
| Source | Key? | Role | Caveat |
|---|---|---|---|
| **yfinance** | no | Primary: prices, fundamentals, analyst targets, recommendations | Unofficial, can break → cache + fallback |
| **SEC EDGAR companyfacts** | no | Authoritative fundamentals fallback | Needs a real `User-Agent`; fair-use rate |
| **FMP free** (250 req/day) | yes (free) | Enrichment: ratios, financial scores | Daily cap → use for top candidates only |
| **Finnhub free** (60/min) | yes (free) | Analyst recommendation trends | 20-min delayed |
| **Finviz / stockanalysis.com** | no | *Light* scraping: analyst consensus color | ToS gray area + fragile → optional, wrapped, cached, never blocks the app |

> Alpha Vantage (25 req/day now) and Tiingo (fundamentals paywalled) are intentionally **not** core sources — too thin on free tiers.

---

## Project structure

```
StockSeeker/
├── app.py                  # Streamlit entry: 3 tabs, tables, detail view, filters, disclaimer
├── requirements.txt
├── config.py               # thresholds, factor weights, market-cap bands, paths, env keys
├── .env.example            # FMP_API_KEY=, FINNHUB_API_KEY= (both optional)
├── data/
│   ├── universe_sp500.csv     # S&P 500 constituents (fetched once from Wikipedia, cached)
│   ├── universe_growth.csv    # curated small/mid-cap growth seed list (editable by you)
│   └── cache/                 # per-ticker JSON + snapshot.parquet (gitignored)
├── screener/
│   ├── universe.py         # build/load + merge the ticker universe
│   ├── fetch.py            # data layer: yfinance + file cache + EDGAR/FMP fallback + throttling
│   ├── scrape.py           # light Finviz/stockanalysis analyst-sentiment scrape (graceful)
│   ├── metrics.py          # compute raw financial metrics per ticker
│   └── scoring.py          # 3 category algorithms + normalization + reason generation
├── refresh.py              # optional: precompute snapshot (run via Windows Task Scheduler)
└── README.md
```

---

## Data & metrics layer (`fetch.py`, `metrics.py`)

For each ticker, gather and cache (TTL ~24h fundamentals, shorter for price):
- Price, 52-wk high/low, **% off 52-wk high/low**, avg volume, beta.
- Market cap.
- Revenue (TTM), **revenue growth YoY**, 3-yr revenue CAGR, gross margin + trend.
- Net income/EPS (**profitable?**), free cash flow.
- Valuation: trailing & forward **P/E, P/S, P/B, EV/EBITDA, PEG**.
- Quality: **ROE/ROIC**, debt/equity, current ratio.
- Dividend: yield, payout ratio (growth-streak approximated where possible).
- Momentum: 3/6/12-mo returns, 50/200-day MA cross, RSI.
- Analyst: mean target, # analysts, recommendation mean (1=strong buy…5=sell), **implied upside %**.

**Caching / no-DB strategy:** `@st.cache_data(ttl=…)` in memory + per-ticker JSON on disk + a single `snapshot.parquet` of the last full screen so the app loads instantly and works offline. "Refresh data" re-fetches. Everything wrapped in try/except → a failed fetch/scrape degrades gracefully and never crashes the app.

**Valuation-vs-history note:** free APIs don't directly expose a stock's 5-yr average P/E. We approximate it from monthly price history × historical TTM EPS to get a "cheap vs its own norm" signal — documented as an approximation, with `% off 52-wk high` as a robust complementary discount proxy.

---

## Scoring engine (`scoring.py`)

Each category produces a **0–100 score**. Two-step method:
1. **Eligibility gate** — a stock only enters a category if it meets minimums (keeps buckets clean).
2. **Weighted factor score** — each factor normalized to 0–100 via **cross-sectional percentile rank within the universe** (robust to outliers) or threshold buckets, then weighted-summed. Weights live in `config.py` and are exposed as sliders.

Every pick ships with its **top reason chips** (e.g., `Revenue +48% YoY`, `32% below 52-wk high`, `ROE 28%`, `Analyst upside +35%`, `Fwd P/E 14 vs ~22 hist`). Transparency is the product.

### Category 1 — High growth potential (catch before expensive)
- **Gate:** market cap ~$300M–$15B, positive revenue growth, price > $2, adequate liquidity.
- **Factors:** Revenue growth YoY (30%) · 3-yr CAGR/acceleration (15%) · gross margin & trend (10%) · valuation-vs-growth, i.e. low P/S-relative-to-growth (15%) · analyst upside (15%) · 6–12-mo relative strength (10%) · estimate revisions/insider buys if available (5%).
- **"Before too expensive" nuance:** explicitly reward *reasonable valuation relative to growth* and **cap the momentum reward** (RSI > 80 reduces it) so we catch names on the way up, not at the peak.

### Category 2 — Value on sale (quality fallen)
- **Gate:** profitable, market cap > ~$2B, **≥15–20% below 52-wk high**, debt not extreme.
- **Factors:** drawdown from high (20%, but capped — a >60% drop may signal a broken thesis) · valuation below own history (25%) · quality composite: ROE/ROIC, margins, low debt, positive FCF (25%) · revenue/earnings still stable-or-growing (15%) · analyst upside + rating not deteriorating (15%).
- **Value-trap guard:** declining revenue/earnings **and** analysts cutting → heavy penalty.

### Category 3 — Everyday compounder (buy & hold / DCA)
- **Gate:** large/established (market cap > ~$10B), profitable, lower beta.
- **Factors:** quality & consistency — stable ROE/ROIC, strong margins, low debt, consistent FCF, earnings stability (35%) · reasonable valuation, penalize euphoria (20%) · durable steady growth (15%) · dividend quality, weighted down if none (15%) · low volatility / drawdown resilience (10%) · analyst consensus (5%).

---

## UI (`app.py`)

- **Sidebar:** Refresh button + last-updated timestamp, universe selector, market-cap min/max filters, advanced factor-weight sliders.
- **Three tabs:** 🚀 High Growth · 💎 Value / On Sale · 🛡️ Steady Compounders.
- **Each tab:** ranked table (ticker, name, price, score, key metrics, analyst upside) with reason chips; selecting a row opens a **detail view** — price chart, factor-breakdown bar, analyst summary, and links to source pages.
- **Disclaimer banner:** "Educational only. Not financial advice. Data may be delayed or inaccurate."

---

## Build phases (incremental — value early)

0. **Scaffold:** repo, `requirements.txt`, `config.py`, `.env.example`, disclaimer.
1. **Universe:** fetch S&P 500 (Wikipedia, cached to CSV) + author `universe_growth.csv` seed; merge/dedupe.
2. **Data layer:** `fetch.py` (yfinance + file cache + throttling) and `metrics.py`; validate on ~20 known tickers.
3. **Scoring:** implement the three algorithms, normalization, and reason generation.
4. **UI:** Streamlit 3-tab app, tables, detail view, filters, disclaimer.
5. **Enrichment:** light Finviz/stockanalysis analyst-sentiment scrape (graceful) + optional FMP/Finnhub fallback.
6. **Snapshot + polish:** `refresh.py` for optional nightly precompute (Windows Task Scheduler), README, `.gitignore`.

---

## Verification

- **Metrics sanity:** `python -m screener.metrics AAPL` prints metrics; eyeball against known values (e.g., AAPL market-cap ballpark).
- **Scoring sanity on a fixed sample:** confirm a known high-grower ranks high in Cat 1, an MSFT-type ranks in Cat 3, and a beaten-down quality name surfaces in Cat 2 — with sensible reason chips.
- **End-to-end:** `streamlit run app.py` → click all three tabs, verify tables populate, Refresh works, a detail view renders its chart.
- **Graceful degradation:** kill network / break the scraper → app still loads from `snapshot.parquet`, no crash.

---

## Key risks & mitigations
- **yfinance fragility / endpoint changes** → file cache + EDGAR/FMP fallback + try/except everywhere.
- **Free-tier rate limits** → curated universe, throttling, batching, snapshot caching.
- **Scraping ToS/fragility** → optional, wrapped, cached, never blocks the app; APIs are the source of truth.
- **Historical-avg-P/E approximation** → documented; `% off 52-wk high` as a complementary signal.
- **Over-trust** → prominent "not financial advice" disclaimer; scores are a *starting point for research*, not a recommendation.

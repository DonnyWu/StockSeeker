# 📈 StockSeeker

A **transparent, multi-factor stock screener** that sorts ideas into three
explainable buckets — and *shows its reasoning* for every pick.

| Bucket | What it surfaces |
|---|---|
| 🚀 **High Growth** | Small/mid-caps growing fast *at a reasonable price* — caught before they get expensive. |
| 💎 **Value / On Sale** | Quality companies that fell hard and now trade at a discount, with a value-trap guard. |
| 🛡️ **Steady Compounders** | Large, profitable, lower-beta names for buy-and-hold / dollar-cost-averaging. |

> ### ⚠️ Educational only. Not financial advice.
> Data may be delayed or inaccurate. Scores are a transparent **starting point
> for your own research**, not a recommendation to buy or sell.

---

## The honest premise

Reliably *predicting* winners with free data and ML is not realistic. What **is**
achievable and genuinely useful is a transparent **factor-investing screen**
(growth, value, quality, momentum, size — the factors real quant screens use)
that ranks candidates and **explains every pick with reason chips**. The value is
the explainable shortlist, not a black-box oracle.

No database. No paid APIs required. Everything degrades gracefully — a flaky data
source can never crash the app.

---

## Quick start

```bash
# 1. Install (Python 3.11+ recommended; tested on 3.14)
python -m pip install -r requirements.txt

# 2. Build the first data snapshot (start with the fast growth-only list)
python refresh.py --growth-only        # ~1 minute, ~60 tickers
#   ...or the full universe (S&P 500 + growth, ~550 tickers, a few minutes)
python refresh.py

# 3. Launch the app
streamlit run app.py
```

Then open the three tabs, sort by score, click **🔍 Inspect a pick** for a
price chart + factor breakdown + live analyst sentiment, and use the sidebar to
filter or to rebuild the snapshot. Move the **factor-weight sliders** to
re-score instantly (no re-fetch).

### Optional free API keys

The app works fully on yfinance alone. For extra analyst-sentiment colour, copy
`.env.example` → `.env` and add any of:

- `FMP_API_KEY` — [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs) (free ~250 req/day): financial scores.
- `FINNHUB_API_KEY` — [Finnhub](https://finnhub.io/) (free 60/min): analyst recommendation trends.
- `SEC_USER_AGENT_EMAIL` — your email, used in the SEC EDGAR User-Agent header.

---

## How scoring works

Each category produces a **0–100 score** in two steps:

1. **Eligibility gate** — a stock only enters a category if it meets that
   category's minimums (market-cap band, profitability, drawdown, liquidity…).
   This keeps the buckets clean and distinct. Gates live in `config.py`.
2. **Weighted factor score** — each factor is normalized to 0–100 via a
   **cross-sectional percentile rank within the eligible set** (robust to
   outliers), then weighted-summed. Weights live in `config.py` and are exposed
   as sliders in the UI.

A few deliberate design choices:

- **Growth, not froth.** The growth bucket rewards *valuation relative to growth*
  (revenue growth ÷ P/S) and **caps the momentum reward when RSI is hot** (>70
  tapers, >80 is heavily discounted) so it catches names on the way up, not at a
  blow-off peak.
- **Discount, not disaster.** The value bucket rewards drawdown from the 52-week
  high *up to a point* — a >60% collapse is folded back, since it more often
  signals a broken thesis than a bargain. A **value-trap guard** halves the score
  when revenue *and* earnings are shrinking *and* analysts are cutting.
- **Compounding, not euphoria.** The compounder bucket weights quality &
  consistency most, penalizes nosebleed valuations, and favours lower beta.

Every pick ships with **reason chips** (`Rev +48% YoY`, `32% below 52-wk high`,
`ROE 28%`, `Analyst upside +35%`, `Fwd P/E 14`) generated from its real metrics.

> **Known approximation:** free APIs don't expose a stock's multi-year average
> P/E. "Cheap vs its own history" is approximated by the price's position within
> its 52-week range, complemented by `% off 52-wk high`. This is a documented
> proxy, not a true historical-multiple comparison.

---

## Project structure

```
StockSeeker/
├── app.py                  # Streamlit UI: 3 tabs, tables, detail view, filters
├── refresh.py              # build/save/load the snapshot (CLI + used by the app)
├── config.py               # gates, factor weights, paths, TTLs, env keys
├── requirements.txt
├── .env.example
├── data/
│   ├── universe_sp500.csv  # S&P 500 constituents (auto-fetched, weekly TTL)
│   ├── universe_growth.csv # curated small/mid-cap growth seed (edit me!)
│   └── cache/              # per-ticker JSON + snapshot.parquet (gitignored)
└── screener/
    ├── universe.py         # build/merge the ticker universe
    ├── fetch.py            # yfinance + file cache + throttling (graceful)
    ├── metrics.py          # raw payload -> flat analytical metrics
    ├── scoring.py          # gates + percentile normalization + reasons
    └── scrape.py           # optional Finviz/Finnhub/FMP enrichment (graceful)
```

### Data sources & their role

| Source | Key? | Role |
|---|---|---|
| **yfinance** | no | Primary: prices, fundamentals, analyst targets/recommendations |
| **Finviz** | no | *Light* scrape: analyst consensus + price target colour |
| **Finnhub** | free | Analyst recommendation trend (buy/hold/sell counts) |
| **FMP** | free | Piotroski / Altman-Z financial scores |

yfinance is the source of truth; the rest are optional colour, wrapped, cached,
and never block the app.

---

## Customizing

- **Add growth names:** edit `data/universe_growth.csv` (columns: `ticker, name,
  sector, note`). They're merged and de-duped with the S&P 500 automatically.
- **Tune the screen:** edit gates and `DEFAULT_WEIGHTS` in `config.py`, or just
  move the sliders in the app.
- **Schedule a nightly refresh (Windows):** point Task Scheduler at
  `python refresh.py` so the app always opens to a fresh snapshot.

---

## Verifying it works

```bash
python -m screener.metrics AAPL    # prints metrics; eyeball vs known values
python -m screener.scoring         # scores a fixed sample (network required)
python refresh.py --growth-only    # builds a snapshot
streamlit run app.py               # click all three tabs + a detail view
```

**Graceful degradation:** kill the network and the app still loads from
`snapshot.parquet` and draws charts from the per-ticker JSON cache — no crash.

---

## Deploying free

Push to GitHub and deploy on
[Streamlit Community Cloud](https://streamlit.io/cloud) (free). Add any API keys
as Streamlit *Secrets*. Note that yfinance can be rate-limited from shared cloud
IPs; the snapshot cache mitigates this, and you can commit a pre-built snapshot
if needed.

---

## Risks & mitigations

- **yfinance fragility** → file cache + try/except everywhere; stale cache beats a crash.
- **Free-tier rate limits** → curated universe, throttling, snapshot caching.
- **Scraping is fragile / ToS gray-area** → optional, wrapped, cached, never blocks the app.
- **Over-trust** → prominent disclaimer; scores are a research starting point, not advice.

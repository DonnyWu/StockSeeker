"""Central configuration for StockSeeker.

Everything tunable lives here: filesystem paths, cache TTLs, optional API keys,
the per-category eligibility gates, and the factor weights used by the scoring
engine. The Streamlit UI exposes the weights as sliders, but these are the
defaults and the single source of truth for the batch/refresh path.

Nothing here touches the network or the filesystem at import time (beyond
ensuring the cache directories exist), so importing `config` is always cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
TICKER_CACHE_DIR = CACHE_DIR / "tickers"

UNIVERSE_SP500_CSV = DATA_DIR / "universe_sp500.csv"
UNIVERSE_GROWTH_CSV = DATA_DIR / "universe_growth.csv"
SNAPSHOT_PARQUET = CACHE_DIR / "snapshot.parquet"
SNAPSHOT_META_JSON = CACHE_DIR / "snapshot_meta.json"

# requests-cache backing store for any HTTP scraping/enrichment calls.
HTTP_CACHE_PATH = CACHE_DIR / "http_cache"

for _d in (DATA_DIR, CACHE_DIR, TICKER_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Optional free API keys (loaded from .env). The app runs fine without them.
# --------------------------------------------------------------------------- #
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
SEC_USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "").strip()

# A polite, identifiable User-Agent is required by SEC EDGAR fair-use rules.
SEC_USER_AGENT = (
    f"StockSeeker/1.0 ({SEC_USER_AGENT_EMAIL or 'anonymous@example.com'})"
)


# --------------------------------------------------------------------------- #
# Caching / throttling
# --------------------------------------------------------------------------- #
# Fundamentals change slowly; prices change fast. TTLs are in seconds.
FUNDAMENTALS_TTL = 24 * 3600       # 24h
PRICE_TTL = 60 * 60               # 1h
SNAPSHOT_TTL = 24 * 3600          # consider the on-disk snapshot fresh for a day

# Be polite to data sources. Seconds to sleep between successive yfinance pulls.
FETCH_THROTTLE_SECONDS = 0.4
# How many network workers to use when refreshing the whole universe.
FETCH_MAX_WORKERS = 6
# Cap how many of the top candidates get expensive enrichment (FMP/Finnhub/scrape).
ENRICH_TOP_N = 25
# HTTP timeout for the enrichment scraping/API calls.
SCRAPE_TIMEOUT_SECONDS = 12


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
# Wikipedia is the free, no-key source for current S&P 500 constituents.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_CSV_TTL = 7 * 24 * 3600  # refresh the S&P 500 list at most weekly


# --------------------------------------------------------------------------- #
# Category eligibility gates
# --------------------------------------------------------------------------- #
# A stock only enters a category if it clears that category's gate. This keeps
# the three buckets clean and conceptually distinct.

GATE_GROWTH = {
    "market_cap_min": 300e6,
    "market_cap_max": 15e9,
    "price_min": 2.0,
    "min_revenue_growth": 0.0,        # must be growing at all
    "min_avg_dollar_volume": 1e6,     # adequate liquidity
}

GATE_VALUE = {
    "market_cap_min": 2e9,
    "price_min": 3.0,
    "min_off_high": 0.15,             # >=15% below the 52-week high
    "max_off_high": 0.75,             # >75% drop usually signals a broken thesis
    "max_debt_to_equity": 3.0,
    "require_profitable": True,
}

GATE_COMPOUNDER = {
    "market_cap_min": 10e9,
    "price_min": 5.0,
    "max_beta": 1.4,
    "require_profitable": True,
}

# "Sharp Drops" is deliberately permissive: we *want* to surface beaten-down
# names (even unprofitable ones) and flag their risks rather than gate them out.
# A name only needs basic liquidity/size AND to actually be dipping/oversold.
GATE_DIP = {
    "market_cap_min": 300e6,          # skip true microcaps
    "price_min": 3.0,
    "min_avg_dollar_volume": 1e6,     # must be tradable
    "shock_sigma": -2.0,              # a >=2-sigma down day, OR
    "short_drop": -0.10,             # >=10% over ~1-2 weeks, OR
    "oversold_rsi": 35.0,            # RSI <= 35 (oversold)
}

# "Moonshots" hunts the Archer (ACHR) profile: low-priced, small/mid-cap,
# speculative names with big upside. Like the dip bucket it is deliberately
# permissive on quality (unprofitable allowed — risks are surfaced, not gated),
# but a name must show *real* potential on at least one signal, and the share
# price + company size must actually be "small/cheap". Note the price gate is
# the literal "low share price" filter the user asked for; absolute price is
# financially arbitrary, so it is a *gate* only — never a scored factor.
GATE_MOONSHOT = {
    "market_cap_min": 300e6,          # skip true microcaps
    "market_cap_max": 20e9,           # small/mid only — room to multiply
    "price_max": 30.0,                # the "low share price" filter (tunable)
    "min_avg_dollar_volume": 1e6,     # must be tradable
    # "high potential" — must clear at least ONE of these:
    "upside_min": 0.30,               # analyst upside >= +30% ...
    "min_analysts": 4,                # ... but only if >= 4 analysts cover it
    "revenue_growth_min": 0.20,       # OR revenue growing >= 20% YoY
}


# --------------------------------------------------------------------------- #
# Factor weights (must sum to ~1.0 within each category)
# --------------------------------------------------------------------------- #
# Each factor is normalized to 0-100 (cross-sectional percentile within the
# eligible set, or a threshold bucket) and then weighted-summed.

WEIGHTS_GROWTH = {
    "revenue_growth": 0.30,
    "revenue_cagr": 0.15,
    "gross_margin": 0.10,
    "value_vs_growth": 0.15,   # low P/S relative to growth (PSG-style)
    "analyst_upside": 0.15,
    "relative_strength": 0.10,  # 6-12mo momentum, capped when overbought
    "earnings_momentum": 0.05,
}

WEIGHTS_VALUE = {
    "drawdown": 0.20,          # discount from 52-wk high, capped
    "valuation_vs_history": 0.25,
    "quality": 0.25,           # ROE/ROIC, margins, low debt, positive FCF
    "fundamental_stability": 0.15,
    "analyst_upside": 0.15,
}

WEIGHTS_COMPOUNDER = {
    "quality_consistency": 0.35,
    "reasonable_valuation": 0.20,
    "durable_growth": 0.15,
    "dividend_quality": 0.15,
    "low_volatility": 0.10,
    "analyst_consensus": 0.05,
}

WEIGHTS_DIP = {
    "shock": 0.25,             # size of the recent sigma down-move
    "short_drawdown": 0.20,    # 1-2 week decline
    "oversold_rsi": 0.20,      # low RSI
    "below_trend": 0.15,       # distance below the 200d MA (capped)
    "analyst_upside": 0.10,    # tilt toward overreactions vs. broken theses
    "fundamental_health": 0.10,  # quality dips rank above junk dips at equal drop
}

WEIGHTS_MOONSHOT = {
    "analyst_upside": 0.30,     # near-term "underpriced" signal
    "revenue_growth": 0.20,     # the growth engine
    "revenue_cagr": 0.10,       # durable multi-year growth
    "room_to_grow": 0.15,       # smaller cap = more room to multiply
    "relative_strength": 0.15,  # some positive momentum (alive, not a falling knife)
    "analyst_conviction": 0.10, # upside corroborated by coverage breadth
}

DEFAULT_WEIGHTS = {
    "growth": WEIGHTS_GROWTH,
    "value": WEIGHTS_VALUE,
    "compounder": WEIGHTS_COMPOUNDER,
    "dip": WEIGHTS_DIP,
    "moonshot": WEIGHTS_MOONSHOT,
}


# --------------------------------------------------------------------------- #
# Scoring tuning knobs
# --------------------------------------------------------------------------- #
RSI_OVERBOUGHT = 80          # above this, the growth momentum reward is cut
RSI_HALVE_REWARD_AT = 70     # momentum reward starts tapering here
DRAWDOWN_BROKEN_THESIS = 0.60  # value: drops beyond this are penalized, not rewarded
MIN_ELIGIBLE_FOR_PERCENTILE = 5  # below this, fall back to absolute thresholds

DIP_DEEP_DROP = 0.60         # dip: below-trend distance is capped here so a total collapse doesn't dominate
MOVERS_SIGMA = -2.0          # "Big movers" strip: surface drops at/below this many sigmas
MOVERS_TOP_N = 8             # how many movers to show in the top-of-page strip

NEUTRAL_SCORE = 50.0         # "factor unavailable" score on the 0-100 scale

# Sector-relative ranking: rank sector-sensitive factors (margins, ROE, debt,
# valuation multiples) against sector peers instead of the whole universe, so
# software isn't auto-rewarded on gross margin nor banks auto-punished on D/E.
SECTOR_RELATIVE = True
MIN_SECTOR_PEERS = 5         # sectors with fewer eligible names fall back to universe rank

# Quality subscore (ROE / margin / D-E / FCF) tuning.
QUALITY_MIN_KNOWN = 2        # need at least this many known components, else neutral
FCF_POSITIVE_SCORE = 100.0   # positive free cash flow bucket score
FCF_NEGATIVE_SCORE = 20.0    # negative free cash flow bucket score

# Dividend-quality factor tuning.
DIVIDEND_YIELD_FLOOR = 0.001   # below this the name is treated as a non-payer
NO_DIVIDEND_SCORE = 20.0       # non-payers get a low (not zero) dividend score
PAYOUT_UNSUSTAINABLE = 0.80    # payout ratios above this are penalized
PAYOUT_PENALTY_MULT = 0.6      # ... by this multiplier

# Value-trap guard (graduated, applied to the final value score).
VALUE_TRAP_REC_MEAN = 3.2          # analyst rec mean worse than this = bearish
VALUE_TRAP_SHRINKING_MULT = 0.75   # revenue AND earnings both shrinking
VALUE_TRAP_FULL_MULT = 0.5         # ... and analysts bearish too

CATEGORIES = ("growth", "value", "compounder", "dip", "moonshot")
CATEGORY_LABELS = {
    "growth": "🚀 High Growth",
    "value": "💎 Value / On Sale",
    "compounder": "🛡️ Steady Compounders",
    "dip": "🩸 Sharp Drops",
    "moonshot": "🌙 Moonshots",
}

DISCLAIMER = (
    "**Educational only. Not financial advice.** Data may be delayed or "
    "inaccurate. Scores are a transparent starting point for your own "
    "research — not a recommendation to buy or sell."
)

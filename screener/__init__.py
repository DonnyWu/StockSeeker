"""StockSeeker screening engine.

Submodules:
    universe  - build/load the ticker universe (S&P 500 + curated growth seed)
    fetch     - data layer: yfinance + file cache + EDGAR/FMP fallback + throttling
    scrape    - light, graceful analyst-sentiment scraping
    metrics   - compute raw financial metrics per ticker
    scoring   - the three category algorithms + normalization + reason chips
"""

__all__ = ["universe", "fetch", "scrape", "metrics", "scoring"]

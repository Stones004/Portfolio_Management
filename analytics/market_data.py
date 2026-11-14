"""
Shared, cached access to Yahoo Finance (via yfinance).

Everything that talks to the network lives here so the rest of the analytics
package is plain maths and easy to test. Results are cached in-process for an
hour, so a "Re-run" or a second page view does not hit Yahoo again.
"""
import math
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_TTL_SECONDS = 60 * 60  # 1 hour

_cache = {}
_lock = threading.Lock()


class DataError(Exception):
    """Raised when required market / statement data is unavailable."""


def _cached(key, loader):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
    value = loader()
    with _lock:
        _cache[key] = (now, value)
    return value


def clear_cache():
    with _lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Currency helpers
# ---------------------------------------------------------------------------
# Some exchanges quote in minor units (pence, cents, agorot).
_MINOR_UNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0),
                "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}


def split_currency(code):
    """Return (major_currency, units_per_major) e.g. 'GBp' -> ('GBP', 100)."""
    if code in _MINOR_UNITS:
        return _MINOR_UNITS[code]
    return (code or "").upper(), 1.0


def fx_rate(from_ccy, to_ccy):
    """Spot FX: how many `to_ccy` for one `from_ccy`."""
    from_ccy, to_ccy = (from_ccy or "").upper(), (to_ccy or "").upper()
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return 1.0

    def load():
        hist = yf.Ticker(f"{from_ccy}{to_ccy}=X").history(period="5d")
        if hist.empty:
            raise DataError(f"Could not fetch FX rate {from_ccy}/{to_ccy}")
        return float(hist["Close"].dropna().iloc[-1])

    return _cached(("fx", from_ccy, to_ccy), load)


def weekly_closes(symbol, period="5y"):
    """Weekly closing prices as a tz-naive Series (flat columns, unlike yf.download)."""
    def load():
        hist = yf.Ticker(symbol).history(period=period, interval="1wk", auto_adjust=True)
        if hist.empty:
            raise DataError(f"No price history for {symbol}")
        closes = hist["Close"].dropna()
        closes.index = closes.index.tz_localize(None).normalize()
        return closes

    return _cached(("weekly", symbol, period), load)


def ensure_df_close(data):
    """Normalise yf.download output to a DataFrame of close prices (columns = tickers)."""
    if data is None or len(data) == 0:
        return pd.DataFrame()
    try:
        close = data["Close"]
    except (KeyError, TypeError):
        close = data
    if isinstance(close, pd.Series):
        close = close.to_frame(name=str(close.name) if close.name else "0")
    close = close.select_dtypes(include=[np.number])
    return close.dropna(how="all")


def download_closes(tickers, period="5y"):
    """Daily close prices for several tickers (cached). Raises DataError if nothing comes back."""
    tickers = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    if not tickers:
        raise DataError("No tickers given.")

    def load():
        try:
            raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        except Exception as exc:
            raise DataError(f"Error fetching prices from Yahoo Finance: {exc}") from exc
        closes = ensure_df_close(raw)
        if closes.empty:
            raise DataError("No closing price data returned for the given tickers.")
        if len(tickers) == 1 and list(closes.columns) != tickers:
            closes.columns = tickers
        return closes

    return _cached(("closes", tuple(tickers), period), load).copy()


def latest_quote(symbol):
    def load():
        hist = yf.Ticker(symbol).history(period="5d")
        return float(hist["Close"].dropna().iloc[-1]) if not hist.empty else None
    return _cached(("quote", symbol), load)


# ---------------------------------------------------------------------------
# Company data
# ---------------------------------------------------------------------------
class CompanyData:
    """Financial statements + metadata for one ticker.

    Statement values are converted into the currency the stock *trades* in,
    so the per-share value is comparable with the market price (e.g. INFY.NS
    reports in USD but trades in INR).
    """

    def __init__(self, ticker):
        self.ticker = ticker.strip().upper()
        self.warnings = []
        t = yf.Ticker(self.ticker)

        # Yahoo's quote-summary endpoint ("info") is often rate-limited or
        # blocked from cloud servers (HTTP 429 / 401 "Invalid Crumb"). Statements
        # and price history come from other endpoints, so carry on without it.
        try:
            self.info = dict(t.info or {})
        except Exception:
            self.info = {}
        if not self.info.get("sector") and not self.info.get("marketCap"):
            self.info.update(self._chart_metadata(t))
            self.warnings.append(
                "Yahoo company profile was unavailable (rate-limited), so sector, country and "
                "currency were inferred from the ticker. Pick the industry model manually if needed.")

        # Throttled requests come back as empty tables rather than errors: retry once.
        for attempt in range(2):
            try:
                self.income = self._clean(t.income_stmt)
                self.balance = self._clean(t.balance_sheet)
                self.cashflow = self._clean(t.cashflow)
            except Exception as exc:
                if attempt:
                    raise DataError(f"Could not load financial statements for '{self.ticker}': {exc}") from exc
                self.income = self.balance = pd.DataFrame()
            if not (self.income.empty or self.balance.empty):
                break
            if attempt == 0:
                time.sleep(2)
                t = yf.Ticker(self.ticker)

        if self.income.empty or self.balance.empty:
            raise DataError(
                f"No financial statements found for '{self.ticker}'. "
                "Check the symbol (Indian stocks need .NS or .BO), or try again in a minute "
                "if Yahoo Finance is rate-limiting requests."
            )

        suffix_ccy, suffix_country = self._suffix_defaults(self.ticker)
        quote_ccy = self.info.get("currency") or suffix_ccy
        self.price_currency, self.minor_units = split_currency(quote_ccy)
        if self.info.get("financialCurrency"):
            self.reporting_currency = self.info["financialCurrency"].upper()
        else:
            self.reporting_currency = self.price_currency
            self.warnings.append(
                f"Reporting currency unknown; assumed statements are in {self.price_currency}.")
        # Multiply statement values by this to express them in price currency.
        self.fx = fx_rate(self.reporting_currency, self.price_currency)

        self.sector = self.info.get("sector") or ""
        self.industry = self.info.get("industry") or ""
        self.country = self.info.get("country") or suffix_country
        self.name = self.info.get("longName") or self.info.get("shortName") or self.ticker

    _SUFFIXES = {
        ".NS": ("INR", "India"), ".BO": ("INR", "India"), ".L": ("GBp", "United Kingdom"),
        ".T": ("JPY", "Japan"), ".DE": ("EUR", "Germany"), ".F": ("EUR", "Germany"),
    }

    @classmethod
    def _suffix_defaults(cls, ticker):
        for suffix, value in cls._SUFFIXES.items():
            if ticker.endswith(suffix):
                return value
        return ("USD", "United States") if "." not in ticker else ("USD", "")

    @staticmethod
    def _chart_metadata(t):
        """Name/currency/price from the chart endpoint, which is rarely blocked."""
        try:
            meta = t.get_history_metadata() or {}
        except Exception:
            return {}
        out = {k: meta[k] for k in ("currency", "longName", "shortName") if meta.get(k)}
        if meta.get("regularMarketPrice"):
            out["currentPrice"] = meta["regularMarketPrice"]
        return out

    @staticmethod
    def _clean(df):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(axis=1, how="all")          # yfinance pads with empty years
        df = df.reindex(sorted(df.columns, reverse=True), axis=1)  # newest first
        return df

    # ---- accessors --------------------------------------------------------
    @staticmethod
    def _row(df, names):
        for name in names:
            if name in df.index:
                return df.loc[name]
        return None

    def series(self, statement, names, convert=True):
        """Return a row (newest first) as floats, or None if not reported."""
        df = getattr(self, statement)
        row = self._row(df, names)
        if row is None:
            return None
        row = pd.to_numeric(row, errors="coerce")
        return row * self.fx if convert else row

    def value(self, statement, names, col=0, default=None, convert=True):
        """Single value from column position `col` (0 = latest year)."""
        row = self.series(statement, names, convert=convert)
        if row is None or col >= len(row):
            return default
        v = row.iloc[col]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)

    def at(self, statement, names, date, default=None, convert=True):
        """Value for a specific fiscal-year-end date (statements can have different columns)."""
        row = self.series(statement, names, convert=convert)
        if row is None or date not in row.index:
            return default
        v = row[date]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)

    def previous_date(self, statement, date):
        """The fiscal year-end immediately before `date` in a statement, if present."""
        cols = [c for c in getattr(self, statement).columns if c < date]
        return max(cols) if cols else None

    def analyst_growth(self, kind="revenue"):
        """Consensus growth estimate (decimal) or None.

        Uses the analysts' long-term growth (LTG) when published, otherwise the
        average of this-year and next-year revenue (or EPS) growth estimates.
        """
        cache_attr = f"_analyst_{kind}"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        value = None
        try:
            t = yf.Ticker(self.ticker)
            ltg = None
            try:
                ge = t.growth_estimates
                if ge is not None and "LTG" in ge.index and "stockTrend" in ge.columns:
                    ltg = ge.loc["LTG", "stockTrend"]
            except Exception:
                pass
            if ltg is not None and not pd.isna(ltg):
                value = float(ltg)
            else:
                est = t.revenue_estimate if kind == "revenue" else t.earnings_estimate
                if est is not None and "growth" in est.columns:
                    g = pd.to_numeric(est["growth"].reindex(["0y", "+1y"]), errors="coerce").dropna()
                    if not g.empty:
                        value = float(g.mean())
        except Exception:
            value = None  # analyst data is optional; never fail a valuation over it

        setattr(self, cache_attr, value)
        return value

    @property
    def years(self):
        return list(self.income.columns)

    @property
    def shares_outstanding(self):
        shares = self.info.get("sharesOutstanding") or self.info.get("impliedSharesOutstanding")
        if not shares:
            shares = self.value("balance", ["Ordinary Shares Number", "Share Issued"], convert=False)
        if not shares:
            raise DataError(f"Share count unavailable for {self.ticker}")
        return float(shares)

    @property
    def market_cap(self):
        cap = self.info.get("marketCap")
        if cap:
            return float(cap) / self.minor_units
        price = self.current_price
        return price * self.shares_outstanding if price else None

    @property
    def current_price(self):
        """Price in major currency units (e.g. GBP, not pence)."""
        price = self.info.get("currentPrice") or self.info.get("regularMarketPrice")
        if not price:
            price = latest_quote(self.ticker)
        return float(price) / self.minor_units if price else None


def get_company_data(ticker):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise DataError("Please enter a ticker symbol.")
    return _cached(("company", ticker), lambda: CompanyData(ticker))

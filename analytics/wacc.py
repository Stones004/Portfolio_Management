"""
Cost of capital, Damodaran style.

  Cost of equity  = Rf + beta x (mature-market ERP + country risk premium)
  beta            = bottom-up industry unlevered beta, relevered at the firm's
                    market D/E (regression beta is the fallback / cross-check)
  Pre-tax Kd      = Rf + country default spread + synthetic-rating spread
                    (rating from the interest coverage ratio)
  WACC            = E/V x Ke + D/V x Kd x (1 - marginal tax)

Country parameters below are editable defaults. Refresh them from Damodaran's
"Country Default Spreads and Risk Premiums" page (updated each January/July).
"""
import numpy as np
import pandas as pd

from .industry import industry_beta
from .market_data import DataError, get_company_data, latest_quote, weekly_closes

MATURE_MARKET_ERP = 0.0433  # implied ERP for the S&P 500 (Damodaran, Jan 2025)

# govt_bond: local-currency 10y government bond yield
# default_spread: sovereign default spread (subtracted to get a risk-free rate,
#                 added back to the cost of debt)
# crp: additional equity country risk premium
# tax: marginal corporate tax rate
COUNTRY_PARAMS = {
    "India":          {"currency": "INR", "govt_bond": 0.064, "default_spread": 0.0200, "crp": 0.0300, "tax": 0.2517, "index": "^NSEI"},
    "United States":  {"currency": "USD", "govt_bond": None,  "default_spread": 0.0,    "crp": 0.0,    "tax": 0.25,   "index": "^GSPC"},
    "United Kingdom": {"currency": "GBP", "govt_bond": 0.045, "default_spread": 0.0060, "crp": 0.0090, "tax": 0.25,   "index": "^FTSE"},
    "Germany":        {"currency": "EUR", "govt_bond": 0.026, "default_spread": 0.0,    "crp": 0.0,    "tax": 0.30,   "index": "^GDAXI"},
    "Japan":          {"currency": "JPY", "govt_bond": 0.015, "default_spread": 0.0060, "crp": 0.0090, "tax": 0.31,   "index": "^N225"},
}
DEFAULT_COUNTRY = {"currency": None, "govt_bond": None, "default_spread": 0.0, "crp": 0.0, "tax": 0.25, "index": "^GSPC"}

# Risk-free rates by *cash-flow currency* when the company's country differs
# from its trading currency (e.g. ADRs). USD is fetched live from ^TNX.
RISK_FREE_BY_CURRENCY = {"INR": 0.044, "USD": None, "GBP": 0.039, "EUR": 0.026, "JPY": 0.009}
FALLBACK_US_10Y = 0.043

# Damodaran's synthetic rating table (large non-financial firms):
# (minimum interest coverage, rating, default spread)
SYNTHETIC_RATINGS = [
    (8.50, "AAA", 0.0059), (6.50, "AA", 0.0078), (5.50, "A+", 0.0098),
    (4.25, "A", 0.0108), (3.00, "A-", 0.0122), (2.50, "BBB", 0.0156),
    (2.25, "BB+", 0.0200), (2.00, "BB", 0.0240), (1.75, "B+", 0.0293),
    (1.50, "B", 0.0360), (1.25, "B-", 0.0447), (0.80, "CCC", 0.0812),
    (0.65, "CC", 0.1018), (0.20, "C", 0.1418), (-1e9, "D", 0.1900),
]

BETA_FLOOR, BETA_CAP = 0.4, 3.0


def us_10y():
    try:
        q = latest_quote("^TNX")
        return q / 100 if q else FALLBACK_US_10Y
    except Exception:
        return FALLBACK_US_10Y


def synthetic_rating(coverage):
    for floor, rating, spread in SYNTHETIC_RATINGS:
        if coverage >= floor:
            return rating, spread
    return SYNTHETIC_RATINGS[-1][1:]


class WACCModel:

    def __init__(self, ticker, risk_free=None, erp=None, tax_rate=None,
                 market_index=None, beta_method="bottom_up"):
        self.data = get_company_data(ticker)
        self.ticker = self.data.ticker
        self.country = COUNTRY_PARAMS.get(self.data.country, DEFAULT_COUNTRY)
        self.beta_method = beta_method
        self.warnings = []

        self.risk_free = risk_free if risk_free is not None else self._risk_free()
        self.erp = erp if erp is not None else MATURE_MARKET_ERP + self.country["crp"]
        self.tax_rate = tax_rate if tax_rate is not None else self.country["tax"]
        self.market_index = market_index or self._market_index()

    # -------------------------------------------------
    # Inputs
    # -------------------------------------------------
    def _risk_free(self):
        ccy = self.data.price_currency
        if ccy == "USD":
            return us_10y()
        if self.country.get("currency") == ccy and self.country.get("govt_bond") is not None:
            # Damodaran: local govt bond minus sovereign default spread
            return self.country["govt_bond"] - self.country["default_spread"]
        rf = RISK_FREE_BY_CURRENCY.get(ccy)
        if rf is None:
            self.warnings.append(f"No risk-free rate configured for {ccy}; using US 10y.")
            return us_10y()
        return rf

    def _market_index(self):
        t = self.ticker
        if t.endswith(".NS") or t.endswith(".BO"):
            return "^NSEI"
        return self.country.get("index", "^GSPC")

    def capital_structure(self):
        """(market value of equity, gross debt incl. leases), both in price currency."""
        E = self.data.market_cap
        if not E:
            raise DataError(f"Market capitalisation unavailable for {self.ticker}")
        D = self.data.value("balance", ["Total Debt"], default=0.0) or 0.0
        return float(E), max(float(D), 0.0)

    # -------------------------------------------------
    # Beta
    # -------------------------------------------------
    def regression_beta(self):
        """5y weekly regression beta, Blume-adjusted (0.67*raw + 0.33)."""
        stock = weekly_closes(self.ticker).pct_change()
        market = weekly_closes(self.market_index).pct_change()
        df = pd.concat([stock, market], axis=1, join="inner").dropna()
        df.columns = ["stock", "market"]
        if len(df) < 52:
            raise DataError("Not enough price history for a regression beta")
        raw = df["stock"].cov(df["market"]) / df["market"].var()
        return 0.67 * raw + 0.33

    def unlevered_beta(self):
        E, D = self.capital_structure()
        beta_u, source = industry_beta(self.data.sector, self.data.industry)
        if beta_u is not None and self.beta_method == "bottom_up":
            self.beta_source = f"bottom-up ({source})"
            return beta_u
        beta_l = self.regression_beta()
        self.beta_source = "regression (5y weekly, Blume-adjusted)"
        return beta_l / (1 + (1 - self.tax_rate) * D / E)

    def levered_beta(self):
        E, D = self.capital_structure()
        beta = self.unlevered_beta() * (1 + (1 - self.tax_rate) * D / E)
        return float(np.clip(beta, BETA_FLOOR, BETA_CAP))

    def equity_beta_financial(self):
        """Banks: industry betas are already levered; D/E (deposits) is not meaningful."""
        beta, source = industry_beta(self.data.sector, self.data.industry)
        if beta is not None and self.beta_method == "bottom_up":
            self.beta_source = f"bottom-up levered ({source})"
        else:
            beta = self.regression_beta()
            self.beta_source = "regression (5y weekly, Blume-adjusted)"
        return float(np.clip(beta, BETA_FLOOR, BETA_CAP))

    # -------------------------------------------------
    # Cost of equity (CAPM)
    # -------------------------------------------------
    def cost_of_equity(self, financial=False):
        beta = self.equity_beta_financial() if financial else self.levered_beta()
        self.beta = beta
        return float(self.risk_free + beta * self.erp)

    # -------------------------------------------------
    # Cost of debt (synthetic rating)
    # -------------------------------------------------
    def cost_of_debt(self):
        """Returns (pre-tax Kd, rating, interest coverage)."""
        ebit = self.data.value("income", ["Operating Income", "EBIT"])
        interest = self.data.value("income", ["Interest Expense", "Interest Expense Non Operating"])
        interest = abs(interest) if interest else 0.0

        if not interest:
            coverage = float("inf")
        elif ebit is None:
            coverage = 0.0
        else:
            coverage = ebit / interest
        rating, spread = synthetic_rating(coverage)
        kd = self.risk_free + self.country["default_spread"] + spread
        return kd, rating, coverage

    # -------------------------------------------------
    # WACC
    # -------------------------------------------------
    def wacc(self, verbose=False):
        return self.summary(verbose)["wacc"]

    def summary(self, verbose=False):
        E, D = self.capital_structure()
        Re = self.cost_of_equity()
        Rd, rating, coverage = self.cost_of_debt()
        V = D + E
        wacc = (E / V) * Re + (D / V) * Rd * (1 - self.tax_rate)

        out = {
            "wacc": wacc,
            "cost_of_equity": Re,
            "pre_tax_cost_of_debt": Rd,
            "after_tax_cost_of_debt": Rd * (1 - self.tax_rate),
            "rating": rating,
            "interest_coverage": coverage,
            "beta": self.beta,
            "beta_source": self.beta_source,
            "risk_free": self.risk_free,
            "erp": self.erp,
            "tax_rate": self.tax_rate,
            "equity_weight": E / V,
            "debt_weight": D / V,
        }
        if verbose:
            print(f"Risk-free      : {self.risk_free:.2%}")
            print(f"ERP            : {self.erp:.2%}")
            print(f"Beta           : {self.beta:.2f} ({self.beta_source})")
            print(f"Cost of Equity : {Re:.2%}")
            print(f"Cost of Debt   : {Rd:.2%} pre-tax ({rating}, coverage {coverage:.1f}x)")
            print(f"Equity Weight  : {E / V:.1%}")
            print(f"Debt Weight    : {D / V:.1%}")
            print(f"WACC           : {wacc:.2%}")
        return out


if __name__ == "__main__":
    WACCModel("INFY.NS").summary(verbose=True)

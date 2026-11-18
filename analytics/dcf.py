"""
Damodaran-style intrinsic valuation with industry-specific models.

Non-financial firms (FCFF, discounted at WACC)
    NOPAT          = Operating income x (1 - tax rate)
    Reinvestment   = (Capex + acquisitions - D&A) + increase in working capital
    ROIC           = NOPAT / beginning invested capital
                     (invested capital = book equity + debt + minority - cash)
    Equity         = EV + cash & short-term investments + cross holdings
                     - debt (incl. leases) - minority interest

Financial firms (FCFE, discounted at cost of equity)
    FCFE           = Net income x (1 - retention)

Growth (both models)
    1. Triangulate: blend up to three estimates
         fundamental  reinvestment rate x ROIC   (retention x ROE for banks)  40%
         historical   revenue CAGR               (net income CAGR for banks)  30%
         analysts     consensus growth from Yahoo                             30%
       (weights are re-scaled over whichever estimates are available)
    2. Longer, not higher: if the blend exceeds the industry's sustainable
       rate, growth is held at that rate for MORE years so the company reaches
       the same size, instead of assuming an unrealistically high rate.
    3. Reinvestment = growth / ROIC, so growth is always paid for.

Forecast: high-growth years, then a linear fade to stable growth, then a
terminal value with reinvestment = stable growth / terminal ROIC.
"""
import math

import numpy as np
import pandas as pd

from . import wacc as wacc_module
from .industry import classify, get_profile
from .market_data import DataError, get_company_data

MAX_TAX_RATE = 0.40
MAX_YEARS = 30

BASE_HIGH_GROWTH_YEARS = 5
MAX_HIGH_GROWTH_YEARS = 10
FADE_YEARS = 5

GROWTH_WEIGHTS = {"fundamental": 0.4, "historical": 0.3, "analyst": 0.3}
GROWTH_ESTIMATE_BOUNDS = (-0.20, 0.60)   # each estimate is clipped before blending
BLENDED_GROWTH_BOUNDS = (-0.05, 0.40)
MAX_PLAUSIBLE_PRICE_TO_SALES = 60   # above this, unverified statement currency is treated as wrong

ASSUMPTION_KEYS = ("discount_rate", "growth_rate", "reinvestment_rate", "terminal_growth",
                   "terminal_return", "high_growth_years", "fade_years")


class DCFError(Exception):
    """A valuation-level problem worth showing to the user."""


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _safe_div(a, b):
    if _nan(a) or _nan(b) or b == 0:
        return float("nan")
    return a / b


def cagr(series):
    """Compound annual growth of a newest-first series of positive values, or NaN."""
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return float("nan")
    return (s.iloc[0] / s.iloc[-1]) ** (1 / (len(s) - 1)) - 1


def blend_growth(estimates, weights=GROWTH_WEIGHTS):
    """Weighted average of the available estimates.

    Returns (blended, rows) where rows = [(name, clipped value, weight used)].
    """
    lo, hi = GROWTH_ESTIMATE_BOUNDS
    usable = {k: float(np.clip(v, lo, hi)) for k, v in estimates.items() if not _nan(v)}
    if not usable:
        return float("nan"), []
    total = sum(weights[k] for k in usable)
    rows = [(k, v, weights[k] / total) for k, v in usable.items()]
    blended = sum(v * w for _, v, w in rows)
    return float(np.clip(blended, *BLENDED_GROWTH_BOUNDS)), rows


def growth_schedule(blended, sustainable, base_years=BASE_HIGH_GROWTH_YEARS,
                    max_years=MAX_HIGH_GROWTH_YEARS):
    """"Longer, not higher".

    If `blended` is above the industry's sustainable rate, keep growth at the
    sustainable rate for enough years to compound to the same size that
    `blended` would reach in `base_years` (capped at `max_years`).
    Returns (rate, high_growth_years).
    """
    if blended <= sustainable:
        return blended, base_years
    target = base_years * math.log1p(blended)
    years = math.ceil(target / math.log1p(sustainable) - 1e-9)
    return sustainable, int(min(max(years, base_years), max_years))


class DCF_Valuation:

    def __init__(self, ticker, industry=None):
        self.data = get_company_data(ticker)
        self.ticker = self.data.ticker
        self.detected_industry = classify(self.data.sector, self.data.industry)
        self.profile = get_profile(industry or self.detected_industry)
        self.model = self.profile["model"]
        self.warnings = list(getattr(self.data, "warnings", []))
        self.growth_sources = []
        self._wacc_model = None
        self._capital = None
        self._defaults = None

    # ------------------------------------------------------------------
    # Cost of capital (computed once per instance)
    # ------------------------------------------------------------------
    @property
    def wacc_model(self):
        if self._wacc_model is None:
            self._wacc_model = wacc_module.WACCModel(self.ticker)
        return self._wacc_model

    @property
    def marginal_tax(self):
        return self.wacc_model.tax_rate

    def cost_of_capital(self):
        if self._capital is None:
            wm = self.wacc_model
            if self.model == "fcfe":
                ke = wm.cost_of_equity(financial=True)
                self._capital = {
                    "discount_rate": ke, "cost_of_equity": ke, "beta": wm.beta,
                    "beta_source": wm.beta_source, "risk_free": wm.risk_free,
                    "erp": wm.erp, "tax_rate": wm.tax_rate,
                }
            else:
                summary = wm.summary()
                self._capital = dict(summary, discount_rate=summary["wacc"])
            self.warnings.extend(wm.warnings)
        return self._capital

    # ------------------------------------------------------------------
    # Historical fundamentals
    # ------------------------------------------------------------------
    def _cash(self, date):
        d = self.data
        total = d.at("balance", ["Cash Cash Equivalents And Short Term Investments"], date)
        if not _nan(total):
            return total
        cash = d.at("balance", ["Cash And Cash Equivalents"], date, default=0.0)
        sti = d.at("balance", ["Other Short Term Investments"], date, default=0.0)
        return cash + sti

    def _invested_capital(self, date):
        d = self.data
        if date is None or date not in d.balance.columns:
            return float("nan")
        equity = d.at("balance", ["Stockholders Equity", "Common Stock Equity"], date)
        if _nan(equity):
            return float("nan")
        debt = d.at("balance", ["Total Debt"], date, default=0.0)
        minority = d.at("balance", ["Minority Interest"], date, default=0.0)
        return equity + debt + minority - self._cash(date)

    def fcff(self):
        """Historical FCFF fundamentals, newest year first."""
        d = self.data
        rows = {}
        for date in d.years:
            revenue = d.at("income", ["Total Revenue", "Operating Revenue"], date)
            ebit = d.at("income", ["Operating Income", "EBIT"], date)
            if _nan(ebit):
                continue  # yfinance often returns a partially-empty oldest year

            tax = d.at("income", ["Tax Provision"], date)
            pretax = d.at("income", ["Pretax Income"], date)
            if _nan(tax) or _nan(pretax) or pretax <= 0:
                tax_rate = self.marginal_tax
            else:
                tax_rate = float(np.clip(tax / pretax, 0.0, MAX_TAX_RATE))
            nopat = ebit * (1 - tax_rate)

            # Cash-flow statement signs: capex/acquisitions are negative outflows,
            # Change In Working Capital is negative when working capital INCREASES.
            capex = -d.at("cashflow", ["Capital Expenditure"], date, default=float("nan"))
            acquisitions = -d.at("cashflow", ["Net Business Purchase And Sale"], date, default=0.0)
            dep = d.at("cashflow", ["Depreciation And Amortization",
                                    "Depreciation Amortization Depletion"], date)
            if _nan(dep):
                dep = d.at("income", ["Reconciled Depreciation"], date, default=0.0)
            delta_wc = -d.at("cashflow", ["Change In Working Capital"], date, default=0.0)

            net_capex = capex + acquisitions - dep
            reinvestment = net_capex + delta_wc

            ic_begin = self._invested_capital(d.previous_date("balance", date))
            if _nan(ic_begin):
                ic_begin = self._invested_capital(date)
            roic = _safe_div(nopat, ic_begin) if ic_begin and ic_begin > 0 else float("nan")

            rows[date] = {
                "Revenue": revenue,
                "Operating Income": ebit,
                "Operating Margin": _safe_div(ebit, revenue),
                "Tax Rate": tax_rate,
                "NOPAT": nopat,
                "Net Capex": net_capex,
                "Change in WC": delta_wc,
                "Reinvestment": reinvestment,
                "Reinvestment_Rate": _safe_div(reinvestment, nopat),
                "Invested Capital": ic_begin,
                "ROIC": roic,
            }
        if not rows:
            raise DCFError(f"Not enough operating data to value {self.ticker}.")
        return pd.DataFrame(rows).T

    def fcfe_history(self):
        """Historical equity fundamentals for financial firms, newest year first."""
        d = self.data
        rows = {}
        for date in d.years:
            ni = d.at("income", ["Net Income Common Stockholders", "Net Income"], date)
            if _nan(ni):
                continue
            prev = d.previous_date("balance", date)
            bv_begin = d.at("balance", ["Common Stock Equity", "Stockholders Equity"], prev) if prev else None
            if _nan(bv_begin):
                bv_begin = d.at("balance", ["Common Stock Equity", "Stockholders Equity"], date)
            dividends = -d.at("cashflow", ["Cash Dividends Paid", "Common Stock Dividend Paid"], date, default=0.0)
            buybacks = -d.at("cashflow", ["Repurchase Of Capital Stock", "Common Stock Payments"], date, default=0.0)
            payout = _safe_div(max(dividends, 0) + max(buybacks, 0), ni)
            rows[date] = {
                "Net Income": ni,
                "Book Equity": bv_begin,
                "ROE": _safe_div(ni, bv_begin) if bv_begin and bv_begin > 0 else float("nan"),
                "Dividends": dividends,
                "Buybacks": buybacks,
                "Payout": payout,
                "Retention": 1 - payout if not _nan(payout) else float("nan"),
            }
        if not rows:
            raise DCFError(f"No net income data to value {self.ticker}.")
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # Default (model-derived) assumptions
    # ------------------------------------------------------------------
    def _historical_inputs(self):
        """Returns (history, historical reinvestment/retention rate, current ROIC/ROE, historical growth)."""
        p = self.profile
        if self.model == "fcfe":
            hist = self.fcfe_history()
            paid = hist["Dividends"].clip(lower=0) + hist["Buybacks"].clip(lower=0)
            rr = 1 - _safe_div(paid.sum(), hist["Net Income"].sum())
            rr = float(np.clip(rr, 0.0, 1.0)) if not _nan(rr) else float("nan")
            ret = hist["ROE"].iloc[0]
            if _nan(ret):
                ret = hist["ROE"].mean()
            hist_growth = cagr(hist["Net Income"])
        else:
            hist = self.fcff()
            valid = hist.dropna(subset=["NOPAT", "Reinvestment"])
            rr = _safe_div(valid["Reinvestment"].sum(), valid["NOPAT"].sum())
            if p["normalize"]:
                v = hist.dropna(subset=["NOPAT", "Invested Capital"])
                v = v[v["Invested Capital"] > 0]
                ret = _safe_div(v["NOPAT"].sum(), v["Invested Capital"].sum())
            else:
                ret = hist["ROIC"].iloc[0]
                if _nan(ret):
                    ret = hist["ROIC"].mean()
            hist_growth = cagr(hist["Revenue"])
        return hist, rr, ret, hist_growth

    def default_assumptions(self):
        if self._defaults is not None:
            return dict(self._defaults)

        cap = self.cost_of_capital()
        r = cap["discount_rate"]
        rf = cap["risk_free"]
        g_term = float(min(rf, r - 0.01))   # Damodaran: stable growth <= risk-free rate
        p = self.profile
        what = "ROE" if self.model == "fcfe" else "ROIC"

        hist, rr_hist, ret, hist_growth = self._historical_inputs()
        self.history = hist
        terminal_return = r if _nan(ret) else max(r, min(ret, r + p["terminal_spread"]))

        # 1. Triangulate growth
        fundamental = rr_hist * ret if (not _nan(rr_hist) and not _nan(ret) and rr_hist > 0 and ret > 0) else float("nan")
        analyst = self.data.analyst_growth("earnings" if self.model == "fcfe" else "revenue")
        blended, rows = blend_growth({"fundamental": fundamental, "historical": hist_growth, "analyst": analyst})
        labels = {
            "fundamental": "Fundamental (retention x ROE)" if self.model == "fcfe" else "Fundamental (reinvestment x ROIC)",
            "historical": "Historical net income CAGR" if self.model == "fcfe" else "Historical revenue CAGR",
            "analyst": "Analyst consensus",
        }
        self.growth_sources = [(labels[k], v, w) for k, v, w in rows]

        if _nan(blended):
            self.warnings.append("No usable growth estimates; defaulting to stable growth from year 1.")
            blended = g_term

        # 2. Longer, not higher
        rate, high_years = growth_schedule(blended, p["max_growth"])
        if rate < blended:
            self.warnings.append(
                f"Blended growth {blended:.1%} is above the {p['max_growth']:.0%} a "
                f"{p['label'].lower()} business typically sustains. Modelled as {rate:.0%} for "
                f"{high_years} years (reaching the same size as {blended:.1%} for "
                f"{BASE_HIGH_GROWTH_YEARS} years{', capped at %d years' % MAX_HIGH_GROWTH_YEARS if high_years == MAX_HIGH_GROWTH_YEARS else ''}), "
                f"then a {FADE_YEARS}-year fade."
            )

        # 3. Reinvestment that pays for the growth
        if not _nan(ret) and ret > 0:
            rr = rate / ret
        else:
            rr = rate / terminal_return
            self.warnings.append(
                f"Historical {what} is negative/unavailable, so reinvestment uses the terminal "
                f"{what} of {terminal_return:.1%}. Review the assumptions.")
        if self.model == "fcfe" and rr > 1:
            # A bank cannot retain more than all its profit without issuing shares.
            self.warnings.append(
                f"Growth of {rate:.1%} at the current ROE of {ret:.1%} would need retention above 100%. "
                "Retention capped at 100%: the extra growth implicitly assumes ROE improves.")
            rr = 1.0
        rr = float(np.clip(rr, -0.5, 1.5))

        self._defaults = {
            "discount_rate": r,
            "growth_rate": float(rate),
            "reinvestment_rate": rr,
            "terminal_growth": g_term,
            "terminal_return": float(terminal_return),
            "high_growth_years": int(high_years),
            "fade_years": FADE_YEARS,
        }
        self.blended_growth = blended
        return dict(self._defaults)

    # ------------------------------------------------------------------
    # Valuation engine (shared by the first run and every re-run)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(a):
        high, fade = int(a["high_growth_years"]), int(a["fade_years"])
        if high < 0 or fade < 0 or not 1 <= high + fade <= MAX_YEARS:
            raise DCFError(f"High-growth + fade years must total between 1 and {MAX_YEARS}.")
        r, g, gt, rt = a["discount_rate"], a["growth_rate"], a["terminal_growth"], a["terminal_return"]
        if not 0 < r < 1:
            raise DCFError("Discount rate must be between 0% and 100%.")
        if gt >= r:
            raise DCFError("Terminal growth must be below the discount rate (otherwise value is infinite).")
        if rt <= 0 or gt >= rt:
            raise DCFError("Terminal return (ROIC/ROE) must be positive and above terminal growth.")
        if not -0.5 < g < 1:
            raise DCFError("High growth rate must be between -50% and 100%.")
        if not -1 <= a["reinvestment_rate"] <= 2:
            raise DCFError("Reinvestment/retention rate must be between -100% and 200%.")
        return high, fade

    def _projection(self, base, a, tax_start=None, tax_end=None):
        """3-stage projection. `base` is year-0 operating income (FCFF) or net income (FCFE).
        Returns (table, terminal_value, pv_terminal, pv_sum)."""
        high, fade = self._validate(a)
        n = high + fade
        r, g, rr = a["discount_rate"], a["growth_rate"], a["reinvestment_rate"]
        g_term, ret_term = a["terminal_growth"], a["terminal_return"]
        rr_term = g_term / ret_term
        is_fcff = self.model == "fcff"

        rows, level = [], base
        for t in range(1, n + 1):
            frac = 0.0 if t <= high else (t - high) / fade
            g_t = g + (g_term - g) * frac
            rr_t = rr + (rr_term - rr) * frac
            level *= (1 + g_t)
            df_t = 1 / (1 + r) ** t
            phase = "High growth" if t <= high else "Fade"
            if is_fcff:
                tax_t = tax_start + (tax_end - tax_start) * frac
                nopat = level * (1 - tax_t)
                reinvest = nopat * rr_t
                cf = nopat - reinvest
                rows.append({"Year": f"Year {t}", "Phase": phase, "Growth": g_t,
                             "Operating Income": level, "Tax Rate": tax_t, "NOPAT": nopat,
                             "Reinvestment Rate": rr_t, "Reinvestment": reinvest, "FCFF": cf,
                             "Discount Factor": df_t, "PV": cf * df_t})
            else:
                reinvest = level * rr_t
                cf = level - reinvest
                rows.append({"Year": f"Year {t}", "Phase": phase, "Growth": g_t,
                             "Net Income": level, "Retention": rr_t, "Retained": reinvest,
                             "FCFE": cf, "Discount Factor": df_t, "PV": cf * df_t})

        next_level = level * (1 + g_term)
        if is_fcff:
            next_cf = next_level * (1 - tax_end) * (1 - rr_term)
        else:
            next_cf = next_level * (1 - rr_term)
        tv = next_cf / (r - g_term)
        pv_tv = tv / (1 + r) ** n

        table = pd.DataFrame(rows)
        return table, tv, pv_tv, table["PV"].sum()

    def _check_units(self, per_share, price, shares):
        """Catch currency/unit mix-ups instead of returning a value that is off by ~100x."""
        d = self.data
        if not price or price <= 0:
            return
        if getattr(d, "reporting_currency_assumed", False):
            # Statements in an unknown currency: if the market values the company at an absurd
            # multiple of its reported sales, the statements are almost surely in another currency.
            revenue = d.value("income", ["Total Revenue", "Operating Revenue"])
            if revenue and revenue > 0 and price * shares / revenue > MAX_PLAUSIBLE_PRICE_TO_SALES:
                raise DCFError(
                    f"Could not confirm which currency {self.ticker}'s financial statements are "
                    f"reported in (Yahoo Finance did not say), and they look inconsistent with a "
                    f"{d.price_currency} share price. Try again in a minute; Yahoo may be rate-limiting.")
        ratio = per_share / price
        if ratio < 0.05 or ratio > 20:
            self.warnings.append(
                f"Intrinsic value is {ratio:.2f}x the market price. A gap this large usually means "
                "unusual inputs (one-off earnings, very low or negative cash flow) - review the "
                "assumptions and historical table before relying on it.")

    def run(self, assumptions=None):
        """Value the company. `assumptions` overrides any default (decimals)."""
        defaults = self.default_assumptions()
        overrides = {k: v for k, v in (assumptions or {}).items() if v is not None and k in ASSUMPTION_KEYS}
        a = dict(defaults, **overrides)
        d = self.data
        shares = d.shares_outstanding

        if self.model == "fcff":
            hist = self.history
            latest_rev = hist["Revenue"].iloc[0]
            if self.profile["normalize"] and not _nan(latest_rev):
                margin = hist["Operating Margin"].dropna().mean()
                base = latest_rev * margin
                tax_start = float(hist["Tax Rate"].mean())
                self.warnings.append(
                    f"Normalised operating income: {margin:.1%} average margin x latest revenue.")
            else:
                base = hist["Operating Income"].iloc[0]
                tax_start = float(hist["Tax Rate"].iloc[0])
            if base <= 0:
                raise DCFError("Base operating income is negative; an FCFF growth model "
                               "cannot value this company without margin-recovery assumptions.")

            table, tv, pv_tv, pv_sum = self._projection(base, a, tax_start, self.marginal_tax)
            ev = pv_sum + pv_tv

            latest = d.years[0]
            cash = self._cash(latest)
            cross = d.at("balance", ["Long Term Equity Investment"], latest, default=0.0)
            debt = d.at("balance", ["Total Debt"], latest, default=0.0)
            minority = d.at("balance", ["Minority Interest"], latest, default=0.0)
            equity = ev + cash + cross - debt - minority
            bridge = [("PV of forecast FCFF", pv_sum), ("PV of terminal value", pv_tv),
                      ("Enterprise value", ev), ("+ Cash & short-term investments", cash),
                      ("+ Cross holdings", cross), ("- Debt (incl. leases)", -debt),
                      ("- Minority interest", -minority), ("Equity value", equity)]
        else:
            base = self.history["Net Income"].iloc[0]
            if base <= 0:
                raise DCFError("Latest net income is negative; FCFE growth model does not apply.")
            table, tv, pv_tv, pv_sum = self._projection(base, a)
            ev = None
            equity = pv_sum + pv_tv
            bridge = [("PV of forecast FCFE", pv_sum), ("PV of terminal value", pv_tv),
                      ("Equity value", equity)]

        per_share = equity / shares
        price = d.current_price
        self._check_units(per_share, price, shares)
        if d.reporting_currency != d.price_currency:
            self.warnings.append(
                f"Statements reported in {d.reporting_currency}; converted to "
                f"{d.price_currency} at spot {d.fx:.4f}.")

        return {
            "ticker": self.ticker,
            "name": d.name,
            "model": self.model,
            "profile": self.profile,
            "detected_industry": self.detected_industry,
            "sector": d.sector,
            "industry": d.industry,
            "currency": d.price_currency,
            "assumptions": a,
            "defaults": defaults,
            "growth_sources": self.growth_sources,
            "blended_growth": self.blended_growth,
            "capital": self.cost_of_capital(),
            "enterprise_value": ev,
            "equity_value": equity,
            "per_share_value": per_share,
            "current_price": price,
            "upside": (per_share / price - 1) if price else None,
            "terminal_value": tv,
            "pv_terminal": pv_tv,
            "terminal_share": pv_tv / (pv_sum + pv_tv) if (pv_sum + pv_tv) else None,
            "shares": shares,
            "bridge": bridge,
            "dcf_table": table,
            "history": self.history,
            "warnings": list(dict.fromkeys(self.warnings)),
        }

    # ------------------------------------------------------------------
    # Backwards-compatible helper
    # ------------------------------------------------------------------
    def run_raw_dcf(self, wacc, growth_rate, reinvestment_rate, terminal_growth,
                    years=10, terminal_roic=None):
        high = math.ceil(years / 2)
        return self.run({
            "discount_rate": wacc, "growth_rate": growth_rate,
            "reinvestment_rate": reinvestment_rate, "terminal_growth": terminal_growth,
            "terminal_return": terminal_roic, "high_growth_years": high, "fade_years": years - high,
        })


def print_valuation(res):
    ccy = res["currency"]
    a = res["assumptions"]
    print(f"\n{res['name']} ({res['ticker']}) - {res['profile']['label']} [{res['model'].upper()}]")
    for label, value, weight in res["growth_sources"]:
        print(f"  growth source {label:<38} {value:7.2%}  weight {weight:.0%}")
    print(f"Discount rate {a['discount_rate']:.2%} | growth {a['growth_rate']:.2%} for "
          f"{a['high_growth_years']}y + {a['fade_years']}y fade | reinvestment {a['reinvestment_rate']:.1%} | "
          f"terminal g {a['terminal_growth']:.2%} | terminal return {a['terminal_return']:.2%}")
    print(res["dcf_table"].round(3))
    for label, value in res["bridge"]:
        print(f"{label:<35} {ccy} {value / 1e9:,.2f}B")
    upside = f"{res['upside']:.1%}" if res["upside"] is not None else "n/a"
    print(f"Value per share: {ccy} {res['per_share_value']:,.2f}  (price {res['current_price']}, upside {upside})")
    for w in res["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    import sys
    for tk in sys.argv[1:] or ["PARADEEP.NS"]:
        try:
            print_valuation(DCF_Valuation(tk).run())
        except (DCFError, DataError) as exc:
            print(f"{tk}: {exc}")

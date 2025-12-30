import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 50)


# ------------------ HELPERS ------------------

def safe_get(df, field, year, default=0):
    try:
        val = df.loc[field, year]
        return 0 if pd.isna(val) else val
    except KeyError:
        return default


# ------------------ DCF CLASS ------------------

class UniversalDCF:

    def __init__(self, ticker, wacc):
        self.ticker = ticker
        self.wacc = wacc
        self.stock = yf.Ticker(ticker)

    # ---------- FCFF + ROIC ----------
    def historical_fcff(self):
        income = self.stock.income_stmt
        bs = self.stock.balance_sheet
        cf = self.stock.cashflow

        years = income.columns[:4]
        rows = []

        for year in years:
            ebit = safe_get(income, "EBIT", year)
            tax = safe_get(income, "Tax Provision", year)
            pretax = safe_get(income, "Pretax Income", year)

            tax_rate = tax / pretax if pretax != 0 else 0.25
            nopat = ebit * (1 - tax_rate)

            dep = safe_get(cf, "Depreciation And Amortization", year)
            capex = abs(safe_get(cf, "Capital Expenditure", year))
            delta_wc = safe_get(cf, "Change In Working Capital", year)

            fcff = nopat + dep - capex - delta_wc

            # ----- Invested Capital -----
            total_assets = safe_get(bs, "Total Assets", year)
            cash = safe_get(bs, "Cash And Cash Equivalents", year)
            curr_liab = safe_get(bs, "Current Liabilities", year)
            curr_debt = (
                safe_get(bs, "Current Debt And Capital Lease Obligation", year)
                or safe_get(bs, "Short Term Debt", year)
            )

            invested_capital = (total_assets - cash) - (curr_liab - curr_debt)
            roic = nopat / invested_capital if invested_capital > 0 else 0

            rows.append({
                "Year": year,
                "NOPAT": nopat,
                "FCFF": fcff,
                "ROIC": roic
            })

        return pd.DataFrame(rows).set_index("Year")

    # ---------- SCENARIOS ----------
    def scenario_fcff(self, years=5):
        hist = self.historical_fcff()

        base_nopat = hist.iloc[0]["NOPAT"]
        base_roic = hist["ROIC"][hist["ROIC"] > 0].median()

        print("Base ROIC:", base_roic)
        print("WACC:", self.wacc)

        scenarios = {
            "Base": {"growth_cap": 0.05},
            "Bull": {"growth_cap": 0.06},
            "Bear": {"growth_cap": 0.03}
        }

        outputs = {}

        for name, s in scenarios.items():

            ### CHANGED — growth decided FIRST (Damodaran-style)
            growth = min(
                s["growth_cap"],
                base_roic * 0.8
            )

            ### CHANGED — reinvestment DERIVED, not assumed
            reinvestment_rate = growth / base_roic if base_roic > 0 else 0

            fcffs = []
            pvs = []

            for t in range(1, years + 1):
                nopat = base_nopat * (1 + growth) ** t
                fcff = nopat * (1 - reinvestment_rate)
                pv = fcff / (1 + self.wacc) ** (t - 0.5)

                fcffs.append(fcff)
                pvs.append(pv)

            terminal_g = min(0.03, self.wacc - 0.03)
            tv = fcffs[-1] * (1 + terminal_g) / (self.wacc - terminal_g)
            pv_tv = tv / (1 + self.wacc) ** years

            ev = sum(pvs) + pv_tv

            outputs[name] = {
                "EV": ev,
                "FCFFs": fcffs
            }

        return outputs

    # ---------- DEBT REDUCTION ----------
    def equity_with_debt_reduction(self, scenario_outputs, paydown_ratio=0.4):
        bs = self.stock.balance_sheet
        year = bs.columns[0]

        debt = safe_get(bs, "Total Debt", year)
        cash = safe_get(bs, "Cash And Cash Equivalents", year)
        shares = self.stock.info.get("sharesOutstanding", 1)

        results = {}

        for name, data in scenario_outputs.items():
            ev = data["EV"]
            fcffs = data["FCFFs"]

            remaining_debt = debt
            if name == "Bull":
                for fcff in fcffs:
                    paydown = max(0, paydown_ratio * fcff)
                    remaining_debt = max(0, remaining_debt - paydown)

            equity = ev + cash - remaining_debt
            per_share = equity / shares

            results[name] = {
                "Equity_Value": equity,
                "Per_Share": per_share
            }

        return results


# ------------------ USAGE ------------------

from WACC import WACCModel

ticker = "eternal.NS"

wacc_model = WACCModel(ticker=ticker)
print(f"\nStock selected: {ticker}")
wacc = wacc_model.wacc()

dcf = UniversalDCF(ticker, wacc)

scenarios = dcf.scenario_fcff()
equity_results = dcf.equity_with_debt_reduction(scenarios)

print("\nFCFF Scenarios:")
for k, v in scenarios.items():
    print(f"{k}: EV = ₹{v['EV']/1e9:.1f}B")

print("\nEquity Value Scenarios (with debt reduction in Bull):")
for k, v in equity_results.items():
    print(f"{k}: Equity = ₹{v['Equity_Value']/1e9:.1f}B | Per Share = ₹{v['Per_Share']:.0f}")

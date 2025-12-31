import yfinance as yf
import pandas as pd
import WACC as wacc
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)

class DCF_Valuation:  # Fixed spelling
    def __init__(self, ticker):
        self.ticker = ticker

    def fcff(self):
        stock = yf.Ticker(self.ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        years = income_stmt.columns[:4]
        results = {}

        for year in years:
            try:
                ebit = income_stmt.loc["EBIT", year]
                tax = income_stmt.loc["Tax Provision", year]
                pretax = income_stmt.loc["Pretax Income", year]
                tax_rate = tax / pretax if pretax != 0 else 0.25
                nopat = ebit * (1 - tax_rate)

                operating_assets = (
                    balance_sheet.loc["Total Assets", year]
                    - balance_sheet.loc["Cash And Cash Equivalents", year]
                )

                current_debt = (
                    balance_sheet.loc["Current Debt And Capital Lease Obligation", year]
                    if "Current Debt And Capital Lease Obligation" in balance_sheet.index
                    else 0
                )

                operating_liabilities = (
                    balance_sheet.loc["Current Liabilities", year]
                    - current_debt
                )

                invested_capital = operating_assets - operating_liabilities

                roic = nopat / invested_capital if invested_capital != 0 else 0

                # Economic assumptions (not accounting noise)
                assumed_growth = 0.07   # 7% explicit growth
                reinvestment_rate = assumed_growth / roic if roic > 0 else 0.2
                reinvestment_rate = min(max(reinvestment_rate, 0.05), 0.35)

                reinvestment = nopat * reinvestment_rate
                fcff = nopat - reinvestment

                results[year] = {
                    "NOPAT": nopat,
                    "ROIC": roic,
                    "Reinvestment": reinvestment,
                    "Reinvestment_Rate": reinvestment_rate,
                    "FCFF": fcff
                }

            except KeyError as e:
                print(f"Missing field {e} for {year}")
                continue

        return pd.DataFrame(results).T
    
    def project_fcff(self, discount_rate, years=5):
        results = self.fcff()
        base_nopat = results.iloc[0]["NOPAT"]

        assumed_growth = 0.07
        reinvestment_rate = min(results["Reinvestment_Rate"].median(), 0.35)
        roic = results["ROIC"].median()

        print(f"Base NOPAT: ₹{base_nopat/1e9:.0f}B")
        print(f"Growth Rate: {assumed_growth:.1%}")
        print(f"Reinvestment Rate: {reinvestment_rate:.0%}")
        print(f"ROIC: {roic:.1%}")

        projections = []

        for t in range(1, years + 1):
            nopat = base_nopat * (1 + assumed_growth) ** t
            reinvestment = nopat * reinvestment_rate
            fcff = nopat - reinvestment
            pv_fcff = fcff / (1 + discount_rate) ** (t - 0.5)

            projections.append({
                "Year": f"Year {t}",
                "NOPAT": nopat,
                "Reinvestment": reinvestment,
                "FCFF": fcff,
                "PV_FCF": pv_fcff
            })

        proj_df = pd.DataFrame(projections)

        terminal_g = min(0.04, discount_rate - 0.02)
        terminal_fcff = proj_df.iloc[-1]["FCFF"] * (1 + terminal_g)
        terminal_value = terminal_fcff / (discount_rate - terminal_g)
        pv_terminal = terminal_value / (1 + discount_rate) ** (years - 0.5)

        enterprise_value = proj_df["PV_FCF"].sum() + pv_terminal

        return enterprise_value, proj_df, {
            "wacc": discount_rate,
            "assumed_growth": assumed_growth,
            "reinvestment_rate": reinvestment_rate,
            "terminal_growth": terminal_g,
            "years": years
        }

    def equity_value(self, enterprise_value):
        stock = yf.Ticker(self.ticker)
        bs = stock.balance_sheet
        latest_year = bs.columns[0]
    
        cash = bs.loc['Cash And Cash Equivalents', latest_year]
        total_debt = bs.loc['Total Debt', latest_year] if 'Total Debt' in bs.index else 0
        shares = stock.info["sharesOutstanding"]

        if not shares or shares < 100_000_000:
            raise ValueError("Invalid sharesOutstanding fetched from yfinance")

        equity_value = enterprise_value + cash - total_debt
        per_share = equity_value / shares
    
#        print(f"Cash: ₹{cash/1e9:.0f}B")
#        print(f"Debt: ₹{total_debt}") 
#        print(f"Equity Value: ₹{equity_value}")
#        print(f"Per Share: ₹{per_share}") 

        print(f"Cash: ₹{cash}")
        print(f"Debt: ₹{total_debt}") 
        print(f"Equity Value: ₹{equity_value}")
        print(f"Per Share: ₹{per_share}")
    
        return equity_value, per_share

    def run_raw_dcf(self, wacc, growth_rate, reinvestment_rate, terminal_growth, years=5):
        """
        Raw DCF using fcff() as the base source.
        Only assumptions change.
        """

        base_df = self.fcff()

        base_nopat = base_df.iloc[0]["NOPAT"]

        stock = yf.Ticker(self.ticker)
        bs = stock.balance_sheet
        year = bs.columns[0]

        cash = bs.loc["Cash And Cash Equivalents", year]
        debt = bs.loc["Total Debt", year]
        shares = stock.info["sharesOutstanding"]

        projections = []

        for t in range(1, years + 1):
            nopat = base_nopat * (1 + growth_rate) ** t
            reinvestment = nopat * reinvestment_rate
            fcff = nopat - reinvestment
            pv_fcff = fcff / (1 + wacc) ** (t - 0.5)

            projections.append({
                "Year": t,
                "NOPAT": nopat,
                "Reinvestment": reinvestment,
                "FCFF": fcff,
                "PV_FCFF": pv_fcff
            })

        df = pd.DataFrame(projections)

        terminal_fcff = df.iloc[-1]["FCFF"] * (1 + terminal_growth)
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        pv_terminal = terminal_value / (1 + wacc) ** years

        enterprise_value = df["PV_FCFF"].sum() + pv_terminal
        equity_value = enterprise_value + cash - debt
        per_share = equity_value / shares

        return {
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "per_share_value": per_share,
            "dcf_table": df
        }


#print(yf.Ticker("infy.NS").balance_sheet.index.tolist())
print()
#print(yf.Ticker("infy.NS").cash_flow.index.tolist())
# FIXED Usage:
ticker = "sail.NS" # [INFY.NS, TCS.NS, WIPRO.NS, GESHIP.NS, RELIANCE.NS, TATAMOTORS.NS, JINDALSTEL.NS, bhel.NS, ]

model_wacc = wacc.WACCModel(ticker=ticker)
print(f'Stock selected: {ticker}')
wacc_value = model_wacc.wacc()
model = DCF_Valuation(ticker)
ev, proj, metrics = model.project_fcff(wacc_value)
eq_val, per_share = model.equity_value(ev)
print(f"\nFinal EV: ₹{ev/1e9:.0f}B")
print(f"Final Equity Value: ₹{eq_val/1e9:.0f}B")
print(f"Final Per Share: ₹{per_share:.0f}")
#print(model.infy_diagnostics())

metrics = {}
metrics['Final EV'] = ev
metrics['Final Equity Value'] = eq_val
metrics['Final Per Share'] = per_share 

#return metrics



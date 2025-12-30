import yfinance as yf
import pandas as pd
import WACC as wacc
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)

class DCF_Valuation:  # Fixed spelling
    def __init__(self, ticker):
        self.ticker = ticker
    
    def fcff(self):
        stock = yf.Ticker(self.ticker)  # FIXED: use self.ticker
        income_stmt = stock.income_stmt # FIXED: get_incomestmt()
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        years = income_stmt.columns[:4]
        results = {}

        for year in years:
            try:
                ebit = income_stmt.loc["EBIT", year]
                tax = income_stmt.loc["Tax Provision", year]
                pretax = income_stmt.loc["Pretax Income", year]  # FIXED field name
                tax_rate = tax / pretax if pretax != 0 else 0.25
                nopat = ebit * (1 - tax_rate)

                depreciation = cash_flow.loc["Depreciation And Amortization", year]  # FIXED
                capex_cf = cash_flow.loc["Capital Expenditure", year]
                capex = -capex_cf
                delta_wc = cash_flow.loc["Change In Working Capital", year]  # FIXED

                reinvestment = capex + delta_wc
                reinvestment_rate = max(min(reinvestment / nopat, 0.8), 0.05)
                
                operating_assets = (
                    balance_sheet.loc["Total Assets", year]
                    - balance_sheet.loc["Cash And Cash Equivalents", year]
                )

                operating_liabilities = (
                    balance_sheet.loc["Current Liabilities", year]
                    - balance_sheet.loc["Current Debt And Capital Lease Obligation", year]
                )

                invested_capital = operating_assets - operating_liabilities
                roic = nopat / invested_capital if invested_capital != 0 else 0

                results[year] = {
                    'Reinvestment': reinvestment,
                    'Reinvestment_Rate': reinvestment_rate, 'ROIC': roic, 'NOPAT': nopat
                }
            except KeyError as e:
                print(f"Missing field {e} for {year}")
                continue

        return pd.DataFrame(results).T

    def project_fcff(self, discount_rate, years=5):  # FIXED: self param
        results = self.fcff()
        latest_year = results.index[0]
        base_nopat = results.loc[latest_year, 'NOPAT']
        reinvestment_rate = min(results['Reinvestment_Rate'].median(), 0.65)  # Cap at 80%
        roic = results['ROIC'].median()    
        growth_rate = reinvestment_rate * roic
        
        print(f"Base NOPAT: ₹{base_nopat/1e9:.0f}B")
        print(f"Growth Rate: {growth_rate:.1%}")
        print(f"Reinvestment Rate: {reinvestment_rate:.0%}")

        projections = []
        for t in range(1, int(years)+1):
            nopat = base_nopat * (1 + growth_rate)**t
            reinvestment = nopat * reinvestment_rate
            fcff = nopat - reinvestment  # FIXED: minus!
            pv = fcff / (1 + discount_rate)**(t-0.5)

            projections.append({
                'Year': f"Year {t}",
                'NOPAT': nopat, 'Reinvestment': reinvestment,
                'FCFF': fcff, 'PV_FCF': pv
            })
        
        proj_df = pd.DataFrame(projections)
        
        # FIXED: Terminal value OUTSIDE loop
        final_fcff = projections[-1]['FCFF']
        terminal_g = min(0.032,discount_rate-0.032)
        tv = final_fcff * (1 + terminal_g) / (discount_rate - terminal_g)
        pv_tv = tv / (1 + discount_rate)**years
        
        enterprise_value = proj_df['PV_FCF'].sum() + pv_tv
        
        print("\nExplicit Forecast:")
        print(proj_df.round(0))
        print(f"\nTerminal Value (PV): ₹{pv_tv/1e9:.0f}B")
        print(f"Enterprise Value: ₹{enterprise_value/1e9:.0f}B")
        
        return enterprise_value, proj_df, {
            "wacc": discount_rate,
            "growth_rate": growth_rate,
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
        shares = bs.loc['Ordinary Shares Number', latest_year]
    
        equity_value = enterprise_value + cash - total_debt
        per_share = equity_value / shares
    
        print(f"Cash: ₹{cash/1e9:.0f}B")
        print(f"Debt: ₹{total_debt/1e9:.0f}B") 
        print(f"Equity Value: ₹{equity_value/1e9:.0f}B")
        print(f"Per Share: ₹{per_share:.0f}")
    
        return equity_value, per_share

    #--------------- FCFF Scenarios -------------------
    def project_fcff_scenarios(self, wacc, years=5):
        base_df = self.fcff()

        base_nopat = base_df.iloc[0]['NOPAT']
        base_roic = base_df['ROIC'].median()
        base_reinv = base_df['Reinvestment_Rate'].median()

        results = {}

        for name, s in SCENARIOS.items():
            roic = base_roic * s["roic_mult"]
            reinv_rate = base_reinv * s["reinv_mult"]
            growth = roic * reinv_rate

            projections = []
            for t in range(1, years + 1):
                nopat = base_nopat * (1 + growth) ** t
                reinvestment = nopat * reinv_rate
                fcff = nopat - reinvestment
                pv = fcff / (1 + wacc) ** (t - 0.5)

                projections.append(pv)

            terminal_g = min(0.03, wacc - 0.03)
            terminal_fcff = fcff * (1 + terminal_g)
            terminal_value = terminal_fcff / (wacc - terminal_g)
            pv_tv = terminal_value / (1 + wacc) ** years

            ev = sum(projections) + pv_tv
            results[name] = ev

        return results

    #--------------- Equity Value Scenarios -------------------
    def equity_value_scenarios(self, ev_dict):
        stock = yf.Ticker(self.ticker)
        bs = stock.balance_sheet
        year = bs.columns[0]

        cash = bs.loc["Cash And Cash Equivalents", year]
        debt = bs.loc["Total Debt", year]
        shares = stock.info["sharesOutstanding"]

        out = {}
        for k, ev in ev_dict.items():
            eq = ev + cash - debt
            out[k] = eq / shares

        return out
    

        """
        Single-run raw DCF.
        All inputs are provided once via a dictionary.
        """

        required = [
            "base_nopat",
            "growth_rate",
            "reinvestment_rate",
            "wacc",
            "terminal_growth",
            "years",
            "cash",
            "debt",
            "shares_outstanding"
        ]

        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")

        base_nopat = inputs["base_nopat"]
        growth = inputs["growth_rate"]
        reinv = inputs["reinvestment_rate"]
        wacc = inputs["wacc"]
        tg = inputs["terminal_growth"]
        years = inputs["years"]

        projections = []

        for t in range(1, years + 1):
            nopat = base_nopat * (1 + growth) ** t
            reinvestment = nopat * reinv
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

        terminal_fcff = df.iloc[-1]["FCFF"] * (1 + tg)
        terminal_value = terminal_fcff / (wacc - tg)
        pv_terminal = terminal_value / (1 + wacc) ** years

        enterprise_value = df["PV_FCFF"].sum() + pv_terminal
        equity_value = enterprise_value + inputs["cash"] - inputs["debt"]
        per_share_value = equity_value / inputs["shares_outstanding"]

        return {
            "Enterprise Value": enterprise_value,
            "Equity Value": equity_value,
            "Per Share Value": per_share_value,
            "PV Terminal Value": pv_terminal,
            "DCF Table": df
        }

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




# FIXED Usage:
ticker = "hindalco.NS" # [INFY.NS, TCS.NS, WIPRO.NS, GESHIP.NS, RELIANCE.NS, TATAMOTORS.NS, JINDALSTEL.NS, bhel.NS, ]
SCENARIOS = {
    "Base": {
        "roic_mult": 1.00,
        "reinv_mult": 1.00
    },
    "Bull": {
        "roic_mult": 1.20,   # efficiency improves
        "reinv_mult": 0.75   # capital discipline improves
    },
    "Bear": {
        "roic_mult": 0.80,
        "reinv_mult": 1.10
    }
}

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

#--------------- FCFF Scenarios -------------------
ev_dict = model.project_fcff_scenarios(wacc_value)
eq_dict = model.equity_value_scenarios(ev_dict)
print("\nFCFF Scenarios:")
print(ev_dict)
print("\nEquity Value Scenarios:")
print(eq_dict)

metrics = {}
metrics['Final EV'] = ev
metrics['Final Equity Value'] = eq_val
metrics['Final Per Share'] = per_share 

#return metrics



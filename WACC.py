import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class WACCModel:

    def __init__(
        self,
        ticker,
        year=datetime.now().year,
        risk_free=0.06,
        tax_rate=0.30,
        market_index="^NSEI"
    ):
        self.ticker = ticker
        self.year = year
        self.risk_free = risk_free
        self.tax_rate = tax_rate
        self.market_index = market_index

        # Cache data
        self.stock = yf.Ticker(ticker)
        self.market_prices = yf.download(market_index, period="5y")
        self.stock_prices = yf.download(ticker, period="5y")

    # -------------------------------------------------
    # BETA CALCULATIONS
    # -------------------------------------------------
    def levered_beta(self):
        market_ret = self.market_prices['Close'].pct_change()
        stock_ret = self.stock_prices['Close'].pct_change()

        df = pd.concat([market_ret, stock_ret], axis=1).dropna()
        df.columns = ['market', 'stock']

        beta_l = df['stock'].cov(df['market']) / df['market'].var()
        return beta_l

    def unlever_beta(self):
        beta_l = self.levered_beta()
        D, E = self.capital_structure()

        beta_u = beta_l / (1 + (1 - self.tax_rate) * (D / E))
        return beta_u

    def relever_beta(self):
        beta_u = self.unlever_beta()
        D, E = self.capital_structure()

        beta_l_target = beta_u * (1 + (1 - self.tax_rate) * (D / E))
        return beta_l_target

    # -------------------------------------------------
    # COST OF EQUITY (CAPM)
    # -------------------------------------------------
    def cost_of_equity(self):
        beta = self.relever_beta()

        market_returns = self.market_prices['Close']
        if isinstance(market_returns, pd.DataFrame):
            market_returns = market_returns.iloc[:, 0]

        market_returns = market_returns.pct_change().dropna()

        years = (market_returns.index[-1] - market_returns.index[0]).days / 365.25
        market_cagr = ((1 + market_returns).prod()) ** (1 / years) - 1

        Re = self.risk_free + beta * (market_cagr - self.risk_free)
        return float(Re)


    # -------------------------------------------------
    # COST OF DEBT (AFTER TAX)
    # -------------------------------------------------
    def cost_of_debt(self):
        income = self.stock.get_incomestmt()
        balance = self.stock.balance_sheet

        interest_expense = abs(
            income.at['InterestExpense', f'{self.year}-03-31']
        )

        debt_cy = balance.at['Total Debt', f'{self.year}-03-31']
        debt_py = balance.at['Total Debt', f'{self.year-1}-03-31']

        avg_debt = np.mean([debt_cy, debt_py])
        rd_pre_tax = interest_expense / avg_debt

        return rd_pre_tax * (1 - self.tax_rate)

    # -------------------------------------------------
    # CAPITAL STRUCTURE (GROSS DEBT)
    # -------------------------------------------------
    def capital_structure(self):
        balance = self.stock.balance_sheet
        market_cap = self.stock.info['marketCap']

        gross_debt = balance.at['Total Debt', f'{self.year}-03-31']
        return gross_debt, market_cap

    # -------------------------------------------------
    # WACC
    # -------------------------------------------------
    def wacc(self):
        Re = float(self.cost_of_equity())
        Rd = float(self.cost_of_debt())
        D, E = self.capital_structure()

        V = D + E
        wacc = (E / V) * Re + (D / V) * Rd

        print(f"Cost of Equity : {Re * 100:.2f}%")
        print(f"Cost of Debt   : {Rd * 100:.2f}%")
        print(f"Equity Weight  : {E / V:.1%}")
        print(f"Debt Weight    : {D / V:.1%}")
        print(f"WACC           : {wacc * 100:.2f}%")

        return wacc

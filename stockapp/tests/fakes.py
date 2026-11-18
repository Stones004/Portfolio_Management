"""Offline stand-ins for Yahoo Finance data so tests never touch the network."""
import numpy as np
import pandas as pd

from analytics.market_data import CompanyData

YEARS = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31"])


def fake_prices(symbols, days=600, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    data = {s: 100 * np.cumprod(1 + rng.normal(0.0005 + 0.0002 * i, 0.01 + 0.003 * i, days))
            for i, s in enumerate(symbols)}
    return pd.DataFrame(data, index=idx)


def _frame(rows):
    return pd.DataFrame(rows, index=YEARS).T


def make_company(ticker="TEST.NS", sector="Technology", industry="Information Technology Services",
                 analyst=0.08, financial=False, reporting="INR", fx=1.0, statement_divisor=1.0,
                 reporting_assumed=False):
    """A CompanyData built from hand-made statements (values in INR).

    To mimic a company that reports in another currency (INFY.NS reports in USD, trades in INR),
    divide the statements by `statement_divisor` and set `reporting` / `fx`. `reporting_assumed`
    mimics Yahoo not saying which currency the statements are in.
    """
    c = CompanyData.__new__(CompanyData)
    c.ticker = ticker
    c.warnings = []
    c.info = {"sharesOutstanding": 1_000_000_000, "marketCap": 2_000_000_000_000,
              "currentPrice": 2000.0, "currency": "INR", "financialCurrency": reporting}
    c.price_currency, c.minor_units, c.reporting_currency, c.fx = "INR", 1.0, reporting, fx
    c.reporting_currency_assumed = reporting_assumed
    c.sector, c.industry, c.country, c.name = sector, industry, "India", "Test Co"

    revenue = [1_300e8, 1_150e8, 1_000e8, 900e8]
    if financial:
        c.income = _frame({"Net Income Common Stockholders": [150e8, 130e8, 115e8, 100e8],
                           "Total Revenue": revenue, "Pretax Income": [200e8] * 4,
                           "Tax Provision": [50e8] * 4})
        c.balance = _frame({"Common Stock Equity": [1_000e8, 900e8, 800e8, 720e8],
                            "Stockholders Equity": [1_000e8, 900e8, 800e8, 720e8],
                            "Total Debt": [5_000e8] * 4})
        c.cashflow = _frame({"Cash Dividends Paid": [-40e8, -35e8, -30e8, -25e8]})
    else:
        c.income = _frame({"Total Revenue": revenue,
                           "Operating Income": [260e8, 230e8, 200e8, 180e8],
                           "Pretax Income": [255e8, 225e8, 195e8, 175e8],
                           "Tax Provision": [64e8, 56e8, 49e8, 44e8],
                           "Interest Expense": [5e8] * 4})
        c.balance = _frame({"Stockholders Equity": [900e8, 800e8, 700e8, 620e8],
                            "Total Debt": [100e8] * 4,
                            "Cash Cash Equivalents And Short Term Investments": [150e8, 140e8, 130e8, 120e8],
                            "Minority Interest": [5e8] * 4})
        c.cashflow = _frame({"Capital Expenditure": [-60e8, -55e8, -50e8, -45e8],
                             "Depreciation And Amortization": [35e8, 32e8, 30e8, 28e8],
                             "Change In Working Capital": [-8e8, -7e8, -6e8, -5e8]})
    if statement_divisor != 1.0:
        for name in ("income", "balance", "cashflow"):
            setattr(c, name, getattr(c, name) / statement_divisor)
    for kind in ("revenue", "earnings"):
        setattr(c, f"_analyst_{kind}", analyst)
    return c

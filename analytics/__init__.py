"""
Finance engine for the Portfolio Management app.

Plain Python (no Django) so it can be used from the web app, scripts and tests:

    market_data      cached Yahoo Finance access (prices, statements, FX, analyst estimates)
    industry         industry classification + valuation profiles + bottom-up betas
    wacc             cost of capital (CAPM, synthetic rating, WACC)
    dcf              Damodaran-style FCFF / FCFE valuation
    portfolio        portfolio return, volatility, Sharpe, VaR/CVaR
    frontier         Monte-Carlo efficient frontier
    charts           matplotlib charts as base64 PNGs
    hrp, black_litterman   research models (not yet in the UI)
"""

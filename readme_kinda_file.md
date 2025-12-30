📊 Stock Portfolio Analytics & Valuation Platform

A full-stack Django-based financial analytics application that enables users to analyze stock portfolios, optimize asset allocation, and perform intrinsic valuation using Modern Portfolio Theory (MPT) and Discounted Cash Flow (DCF) models.

🔧 Tech Stack

Backend: Django (Python)
Frontend: HTML, CSS, Bootstrap
Data Analysis: NumPy, Pandas
Visualization: Matplotlib (server-side rendering)
Financial Data: yFinance API
Auth: Django Authentication System

🧠 Core Concepts & Topics Used

1️⃣ Portfolio Management
✔ Portfolio & Holdings
Users can create multiple portfolios
Each portfolio consists of:
Stock symbol
Weight allocation
Data persisted using Django Models:
Portfolio
Holding

✔ File Upload Support
Import portfolios via:
CSV
Excel
Auto-validation of columns (symbol, weight)

2️⃣ Financial Time Series Analysis

✔ Price Data Handling
Fetches 5-year historical prices using yfinance
Handles:
Single ticker vs multiple tickers
Missing or invalid tickers
Normalization of data formats

✔ Return Computation
Daily returns using:
return_t = (P_t / P_{t-1}) - 1
Annualization using 252 trading days

3️⃣ Modern Portfolio Theory (MPT)
✔ Expected Return
Weighted average of individual stock returns

✔ Portfolio Volatility
Uses covariance matrix:
σ_p = √(wᵀ Σ w)

✔ Sharpe Ratio
Risk-adjusted return metric:
Sharpe = (Rp − Rf) / σp

4️⃣ Risk Metrics
✔ Value at Risk (VaR – 95%)
Measures maximum expected loss under normal conditions

✔ Conditional VaR (CVaR)
Expected loss beyond VaR
Captures tail risk

✔ Distribution Visualization
Histogram of daily portfolio returns
Rendered server-side & embedded using Base64

5️⃣ Efficient Frontier Optimization
✔ Monte Carlo Simulation
Generates 50,000 random portfolios
Computes:
Return
Risk
Sharpe Ratio

✔ Optimal Portfolio
Identifies Maximum Sharpe Ratio portfolio
Compares:
Original weights
Optimized weights
Allocation changes

✔ Efficient Frontier Plot
Risk vs Return scatter
Color-coded by Sharpe ratio

6️⃣ Discounted Cash Flow (DCF) Valuation

✔ Free Cash Flow to Firm (FCFF)
Projects future cash flows
Includes:
Growth rate
Reinvestment rate
Terminal growth

✔ Terminal Value
Gordon Growth Model:
TV = FCFF × (1 + g) / (WACC − g)

7️⃣ Cost of Capital (WACC)

✔ CAPM (Cost of Equity)
Uses:
Risk-free rate
Market return
Beta (covariance-based)

✔ Cost of Debt
Derived from:
Interest expense
Average debt
Tax shield

✔ Capital Structure
Market value of equity
Book value of debt

8️⃣ Interactive DCF Sensitivity Analysis

✔ First Run
Model-driven assumptions auto-calculated
Stored as immutable baseline

✔ Subsequent Runs
Users can manually tweak:
WACC
Growth rate
Reinvestment rate
Terminal growth
Projection years
Enables what-if valuation analysis

9️⃣ Authentication & Security
User registration & login
Portfolio isolation per user
CSRF protection
Session-based parameter storage

🔄 Application Flow Summary
User logs in / registers
Creates or loads a portfolio
Analyzes:
Return
Risk
Sharpe
VaR / CVaR
Optimizes portfolio via Efficient Frontier
Performs intrinsic valuation via DCF
Runs sensitivity analysis interactively

🎯 Key Strengths of the Project
Combines investment theory + real data
Fully end-to-end (UI → Analytics → Valuation)
Handles edge cases & bad data
Scalable for:
Sector-level modeling
Multi-segment DCF
Advanced risk models
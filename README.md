# 📊 Portfolio Analytics & Valuation Platform

A Django app for analysing stock portfolios, optimising allocations and valuing
companies with an industry-aware, Damodaran-style DCF. Market data comes from
Yahoo Finance (via `yfinance`).

**Stack:** Django 5.2 · pandas / NumPy · matplotlib · yfinance · Postgres (Neon) in production, SQLite locally · deployed on Render with Docker + gunicorn.

## Features

| Page | What it does |
|---|---|
| **Portfolio** | Enter or upload (CSV/Excel) holdings → 5-year annual return, volatility, Sharpe ratio, 1-day 95% VaR / CVaR, per-stock stats, returns histogram. Save / load / delete portfolios per user. |
| **Efficient Frontier** | Simulates 30,000 long-only portfolios (vectorised Monte Carlo) and shows the max-Sharpe and min-volatility mixes vs your current weights, with a reallocation table. |
| **DCF Valuation** | Intrinsic value per share with a model chosen by industry, editable assumptions, value bridge, cost-of-capital breakdown, forecast and history tables. |
| **Portfolio DCF** | Runs the DCF on every holding of a saved portfolio (one stock at a time from the browser, so no request timeouts), with price vs intrinsic value, upside and verdict per stock and a weight-averaged portfolio upside. Each user's latest valuation per ticker is saved. |

### How the DCF works

- **Industry models.** Banks, insurers and NBFCs use **FCFE** at the cost of equity (deposits are raw material, not financing). Everything else uses **FCFF** at WACC. Cyclicals, utilities and real estate use multi-year average margins. Tech and consumer brands keep a moat premium on terminal ROIC.
- **Growth is triangulated.** It's a weighted blend of fundamental growth (reinvestment × ROIC, or retention × ROE for banks) at 40%, historical revenue / net-income CAGR at 30%, and analyst consensus at 30%.
- **Longer, not higher.** If blended growth is above what the industry typically sustains, the model keeps the industry rate for more high-growth years (up to 10) rather than using an unrealistic rate.
- **Growth is paid for.** Reinvestment = growth ÷ ROIC. In the terminal phase, reinvestment = stable growth ÷ terminal ROIC, and stable growth is capped at the risk-free rate.
- **Cost of capital.**
  - Beta: bottom-up industry beta, relevered at the company's market D/E.
  - Equity risk premium: implied ERP plus a country risk premium.
  - Cost of debt: from a synthetic rating based on interest coverage.
- **Equity bridge.** EV + cash & short-term investments + cross holdings − debt (incl. leases) − minority interest. Statements reported in another currency are converted at spot (e.g. INFY.NS reports in USD).

Country inputs (risk-free rates, ERP, country premiums) and industry betas are editable constants in
`analytics/wacc.py` and `analytics/industry.py`. Refresh them from Damodaran's datasets once a year.

## Project layout

```
analytics/            finance engine - plain Python, no Django
  market_data.py      cached Yahoo Finance access (prices, statements, FX, analyst estimates)
  industry.py         industry classification, valuation profiles, bottom-up betas
  wacc.py             cost of capital
  dcf.py              FCFF / FCFE valuation, growth triangulation
  portfolio.py        return, volatility, Sharpe, VaR/CVaR
  frontier.py         Monte-Carlo efficient frontier
  charts.py           matplotlib charts → base64 PNG
  hrp.py, black_litterman.py   research models (not in the UI yet)
stockapp/             Django app
  views/              one module per page (portfolio, frontier, dcf, auth)
  templates/stockapp/ base.html + one template per page
  holdings.py         CSV/Excel upload parsing
  tests/              offline test suite (no network)
stocksite/            Django project (settings, urls, wsgi)
scripts/              command-line tools (run_dcf, dcf_screener)
docs/                 notes (HRP maths)
Dockerfile, entrypoint.sh, docker-compose.yml, requirements.txt
```

## Run locally

```bash
python -m venv venv && venv\Scripts\activate      # source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver                         # http://127.0.0.1:8000
python manage.py test stockapp                     # offline tests
python -m scripts.run_dcf INFY.NS HDFCBANK.NS      # DCF from the command line
```

With no environment variables set, the app uses SQLite and `DEBUG=True`. To mirror production
(Postgres + gunicorn), run `docker compose up --build`.

## Deploy: Render + Neon

1. **Neon.** Create a project and copy the **pooled** connection string (host contains `-pooler`, ends with `?sslmode=require`).
2. **Render.** Open the web service → *Environment* and set:
   - `DATABASE_URL`: the Neon string
   - `DJANGO_SECRET_KEY`: a long random value, e.g. `python -c "import secrets; print(secrets.token_urlsafe(50))"`
   - `DJANGO_DEBUG`: `False`
3. **Deploy.** `entrypoint.sh` runs `migrate` (which creates the tables in Neon), then `collectstatic`, then starts gunicorn on `$PORT` with a 120-second timeout.

Without `DATABASE_URL` the app falls back to SQLite, and SQLite on Render is wiped on every deploy and restart, so users would have to register again. The deploy log shows warning `stockapp.W001` when this happens. Alternative: attach a Render persistent disk (paid plans) and set `SQLITE_PATH=/var/data/db.sqlite3`.

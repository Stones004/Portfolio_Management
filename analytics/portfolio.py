"""Portfolio risk/return metrics (Modern Portfolio Theory)."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.05


def align_weights(prices, symbols, weights):
    """Keep only symbols that returned data; renormalise their weights.

    Returns (prices restricted to those symbols, weights array, list of skipped symbols).
    """
    col_map = {c.upper(): c for c in prices.columns}
    cols, kept, skipped = [], [], []
    for s, w in zip(symbols, weights):
        if s.upper() in col_map:
            cols.append(col_map[s.upper()])
            kept.append(float(w))
        else:
            skipped.append(s)
    if not cols:
        raise ValueError("None of the requested tickers returned price data.")
    w = np.array(kept, dtype=float)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Weights must add up to more than zero.")
    return prices[cols], w / w.sum(), skipped


def portfolio_metrics(prices, weights, risk_free=DEFAULT_RISK_FREE):
    """Annualised return/volatility, Sharpe, 95% VaR/CVaR and per-asset stats.

    prices  : DataFrame of daily closes, one column per asset (same order as weights)
    weights : array summing to 1
    """
    weights = np.asarray(weights, dtype=float)
    if prices.shape[1] != len(weights):
        raise ValueError("Weights length must match number of assets.")

    daily = prices.pct_change(fill_method=None).dropna()
    if daily.empty:
        raise ValueError("Not enough price history to compute returns.")

    mean_returns = daily.mean() * TRADING_DAYS
    cov = daily.cov() * TRADING_DAYS
    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (port_return - risk_free) / port_vol if port_vol else 0.0

    port_daily = daily.values @ weights
    cutoff = np.percentile(port_daily, 5)
    var_95 = float(-cutoff)
    tail = port_daily[port_daily <= cutoff]
    cvar_95 = float(-tail.mean()) if len(tail) else 0.0

    details = [{
        "symbol": col,
        "weight": round(float(weights[i]), 4),
        "mean": round(float(mean_returns.iloc[i] * 100), 2),
        "vol": round(float(np.sqrt(cov.iloc[i, i]) * 100), 2),
    } for i, col in enumerate(prices.columns)]

    return {
        "portfolio_return": round(port_return * 100, 2),
        "portfolio_volatility": round(port_vol * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "var_95": round(var_95 * 100, 2),
        "cvar_95": round(cvar_95 * 100, 2),
        "worst_day": round(float(-port_daily.min()) * 100, 2),
        "details": details,
        "daily_returns": pd.Series(port_daily, index=daily.index),
    }

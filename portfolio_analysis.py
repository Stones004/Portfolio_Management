import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

RISK_FREE_RATE = 0.05


def analyze_portfolio(data: pd.DataFrame, weights: np.ndarray):
    """
    Core portfolio analytics.
    data   : price dataframe (Close prices)
    weights: normalized weights
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data)}")

    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D array")

    if len(weights) != data.shape[1]:
        raise ValueError("Weights length must match number of assets")

    if not isinstance(weights, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(weights)}")
    returns = data.pct_change().dropna()

    if returns.shape[0] == 0:
        raise ValueError(
            "Not enough price history to compute returns. "
            "Need at least 2 price rows."
        )

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol else 0.0

    daily_port_returns = returns.values @ weights

    if daily_port_returns.size == 0:
        raise ValueError(
            "Portfolio daily returns are empty. "
            "Cannot compute VaR / CVaR."
        )

    var_95 = -np.percentile(daily_port_returns, 5) * 100
    cvar_95 = -daily_port_returns[daily_port_returns <= np.percentile(daily_port_returns, 5)].mean() * 100

    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(daily_port_returns, bins=50)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    histogram = base64.b64encode(buf.read()).decode()

    details = []
    for i, col in enumerate(data.columns):
        details.append({
            "symbol": col,
            "weight": round(weights[i], 4),
            "mean": round(mean_returns[i] * 100, 2),
            "vol": round(np.sqrt(cov_matrix.iloc[i, i]) * 100, 2)
        })

    return {
        "portfolio_return": round(port_return * 100, 2),
        "portfolio_volatility": round(port_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "details": details,
        "histogram": histogram
    }

"""Monte-Carlo efficient frontier (vectorised).

The old version looped 50,000 times in Python; the same simulation as one
matrix operation runs in a fraction of a second, which matters on a small
server with a request timeout.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_frontier(returns, risk_free=0.05, n_portfolios=30000, current_weights=None, seed=None):
    """Simulate random long-only portfolios and pick the best ones.

    returns         : DataFrame of daily returns, one column per asset
    current_weights : optional array (same column order) to mark the user's portfolio
    """
    if returns.shape[1] < 2:
        raise ValueError("At least two assets with price history are required.")

    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    rng = np.random.default_rng(seed)

    # Dirichlet(1,...,1) samples weights uniformly over all long-only portfolios
    w = rng.dirichlet(np.ones(len(mu)), size=n_portfolios)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sharpes = np.divide(rets - risk_free, vols, out=np.zeros_like(rets), where=vols > 0)

    best = int(np.argmax(sharpes))
    min_vol = int(np.argmin(vols))
    assets = list(returns.columns)

    out = {
        "assets": assets,
        "rets": rets, "vols": vols, "sharpes": sharpes,
        "best_weights": pd.Series(w[best], index=assets).sort_values(ascending=False),
        "best": {"return": float(rets[best]), "volatility": float(vols[best]), "sharpe": float(sharpes[best])},
        "min_vol_weights": pd.Series(w[min_vol], index=assets).sort_values(ascending=False),
        "min_vol": {"return": float(rets[min_vol]), "volatility": float(vols[min_vol]),
                    "sharpe": float(sharpes[min_vol])},
        "current": None,
    }
    if current_weights is not None:
        cw = np.asarray(current_weights, dtype=float)
        c_ret, c_vol = float(cw @ mu), float(np.sqrt(cw @ cov @ cw))
        out["current"] = {"return": c_ret, "volatility": c_vol,
                          "sharpe": (c_ret - risk_free) / c_vol if c_vol else 0.0}
    return out


def weight_changes(current, optimal):
    """Compare {symbol: weight} dicts; sorted by optimal weight."""
    rows = []
    for symbol, new_w in optimal.items():
        old_w = float(current.get(symbol, 0.0))
        rows.append({"symbol": symbol, "old_weight": round(old_w, 4),
                     "new_weight": round(float(new_w), 4), "change": round(float(new_w) - old_w, 4)})
    return sorted(rows, key=lambda r: r["new_weight"], reverse=True)

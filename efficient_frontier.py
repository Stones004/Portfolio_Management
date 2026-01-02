import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_efficient_frontier(returns, risk_free_rate=0.05, n_portfolios=50000):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)

    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        w = np.random.random(num_assets)
        w /= w.sum()

        port_return = w @ mean_returns
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol else 0

        results[:, i] = [port_return, port_vol, sharpe]
        weights_record.append(w)

    weights_record = np.array(weights_record)
    max_idx = np.argmax(results[2])

    best_weights = weights_record[max_idx]
    best_return, best_vol, best_sharpe = results[:, max_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1], results[0], c=results[2], cmap="viridis")
    plt.scatter(best_vol, best_return, c="red", marker="*", s=150)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    plot = base64.b64encode(buf.read()).decode()

    return {
        "plot": plot,
        "best_weights": best_weights,
        "best_return": best_return * 100,
        "best_vol": best_vol * 100,
        "best_sharpe": best_sharpe
    }

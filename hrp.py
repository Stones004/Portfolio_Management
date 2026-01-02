import numpy as np
import pandas as pd
from yfinance import Ticker
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def returns(tickers):
    raw = yf.download(tickers, period="5y", progress=False)
    data = _ensure_df_close(raw)
    if data.empty:
        raise ValueError("No closing price data returned for the given tickers.")
    return data

def correl_distance(corr):
    """
    Convert correlation matrix to distance matrix
    """
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    """
    Reorder assets so correlated ones are adjacent
    by flattening the hierarchical clustering tree.
    """
    link = link.astype(int)

    # Start from the root cluster (last merge)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original assets

    # Expand clusters until only leaf nodes remain
    while sort_ix.max() >= num_items:

        # Create gaps to insert children
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)

        # Identify cluster nodes
        clusters = sort_ix[sort_ix >= num_items]
        idx = clusters.index
        rows = clusters.values - num_items

        # Replace cluster with left child
        sort_ix[idx] = link[rows, 0]

        # Insert right child next to left child
        right = pd.Series(link[rows, 1], index=idx + 1)
        sort_ix = sort_ix._append(right)

        # Restore sequential order
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(len(sort_ix))

    return sort_ix.tolist()

def cluster_variance(cov, items):
    """
    Compute cluster variance wᵀΣw
    """
    cov_slice = cov.iloc[items, items]
    weights = np.ones(len(items)) / len(items)
    return weights @ cov_slice.values @ weights

def recursive_bisection(cov, sorted_items):
    """
    Allocate weights recursively using cluster variances
    """
    weights = pd.Series(1.0, index=sorted_items)
    clusters = [sorted_items]

    while len(clusters) > 0:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]

        var_left = cluster_variance(cov, left)
        var_right = cluster_variance(cov, right)

        alpha = 1 - var_left / (var_left + var_right)

        weights[left] *= alpha
        weights[right] *= 1 - alpha

        clusters.append(left)
        clusters.append(right)

    return weights

def hrp_weights(returns_df):
    """
    Compute HRP portfolio weights
    """
    cov = returns_df.cov()
    corr = returns_df.corr()

    dist = correl_distance(corr)
    dist_condensed = squareform(dist, checks=False)

    link = linkage(dist_condensed, method="single")
    sorted_idx = get_quasi_diag(link)

    hrp_w = recursive_bisection(cov, sorted_idx)
    hrp_w = hrp_w / hrp_w.sum()

    return hrp_w    


assets=["INFY.NS", "TCS.NS", "WIPRO.NS", "GESHIP.NS", "RELIANCE.NS", "TATAMOTORS.NS", "JINDALSTEL.NS", "bhel.NS", ]
returns_df = returns(assets)

print(hrp_weights(returns_df))
print(f'Sum: {hrp_weights(returns_df).sum()}')
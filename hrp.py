import numpy as np
import pandas as pd
from yfinance import Ticker
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from data_utils import ensure_df_close
import matplotlib.pyplot as plt


class HRP :

    def returns(self,tickers):
        raw = yf.download(tickers, period="5y", progress=False)
        data = ensure_df_close(raw)
        data = data.pct_change().dropna()
        if data.empty:
            raise ValueError("No closing price data returned for the given tickers.")
        return data

    def correl_distance(self,corr):
        """
        Convert correlation matrix to distance matrix
        """
        return np.sqrt(0.5 * (1 - corr))

    def get_quasi_diag(self,link):
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

    def cluster_variance(self,cov, items):
        """
        Compute cluster variance wᵀΣw
        """
        cov_slice = cov.iloc[items, items]
        weights = np.ones(len(items)) / len(items)
        return weights @ cov_slice.values @ weights

    def recursive_bisection(self,cov, sorted_items):
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

            var_left = self.cluster_variance(cov, left)
            var_right = self.cluster_variance(cov, right)

            alpha = 1 - var_left / (var_left + var_right)

            weights[left] *= alpha
            weights[right] *= 1 - alpha

            clusters.append(left)
            clusters.append(right)

        return weights

    def hrp_weights(self,returns_df):
        """
        Compute HRP portfolio weights
        """
        cov = returns_df.cov() * 0.9 + np.diag(np.diag(returns_df.cov())) * 0.1
        corr = returns_df.corr()

        dist = self.correl_distance(corr)
        dist_condensed = squareform(dist, checks=False)

        link = linkage(dist_condensed, method="ward")
        sorted_idx = self.get_quasi_diag(link)

        hrp_w = self.recursive_bisection(cov, sorted_idx)
        hrp_w = hrp_w / hrp_w.sum()

        asset_names = returns_df.columns
        hrp_w.index = asset_names[hrp_w.index]

        return hrp_w.sort_values(ascending=False), cov, corr


    def get_returns(self,tickers):
        raw = yf.download(tickers, period="5y", progress=False)
        prices = ensure_df_close(raw)

        if prices.empty or prices.shape[0] < 50:
            raise ValueError("Insufficient price data")

        returns = prices.pct_change(fill_method=None).dropna()
        return returns

    def max_drawdown(self,port_ret):
        port_ret = pd.Series(port_ret)   # 👈 critical line
        cum = (1 + port_ret).cumprod()
        peak = cum.expanding().max()
        drawdown = (cum - peak) / peak
        return drawdown.min()

    def equal_weight(self,n):
        return np.ones(n) / n

    def portfolio_performance(self,returns, weights, rf=0.05):
        port_ret = returns.values @ weights
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else 0

        print(f'\nAnnualized Return: {ann_ret:.2%}')
        print(f'Annualized Volatility: {ann_vol:.2%}')
        print(f'Sharpe Ratio: {sharpe:.2f}')
        print(f'Max Drawdown: {self.max_drawdown(port_ret):.2%}\n')
        return ann_ret, ann_vol, sharpe

    def hrp_adapter(self,returns):
        w , cov, corr = self.hrp_weights(returns)
        return w.loc[returns.columns].values

    def mpt_adapter(self,returns):
        mu = returns.mean().values
        cov = returns.cov().values
        inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ mu
        return w / w.sum()

    def rolling_backtest(self,
        returns,
        weight_func,
        train_years=2,
        test_years=1
    ):
        train = train_years * 252
        test = test_years * 252

        results = []

        total_rows = len(returns)
        print("Total return rows:", total_rows)
        print("Required rows:", train + test)

        if total_rows < train + test:
            raise ValueError(
                f"Not enough data: need {train + test}, have {total_rows}"
            )

        for start in range(0, total_rows - train - test + 1, test):
            train_slice = returns.iloc[start : start + train]
            test_slice = returns.iloc[start + train : start + train + test]

            w = weight_func(train_slice)

            # portfolio returns
            port_ret = test_slice.values @ w
            results.append(port_ret)

        return pd.Series(np.concatenate(results))


'''assets = [
    "INFY.NS", "TCS.NS", "WIPRO.NS",
    "HDFCBANK.NS", "HINDUNILVR.NS", "RELIANCE.NS","ASIANPAINT.NS",
    "DRREDDY.NS","LTIM.NS","MARUTI.NS"
]

assets=["INFY.NS","jklakshmi.NS","geship.NS","RELIANCE.NS","bel.ns","jyothylab.ns","ptc.ns","sbin.ns","paradeep.ns","sail.ns","ioc.ns","apollotyre.ns","idfcfirstb.ns","ncc.ns","exideind.ns","nhpc.ns","itc.ns","goldbees.ns","hfcl.ns","federalbnk.ns","tatasteel.ns","orienthot.ns"]

hrp = HRP()
returns = hrp.get_returns(assets)

bt_hrp = hrp.rolling_backtest(returns, hrp.hrp_adapter)
bt_eq  = hrp.rolling_backtest(returns, lambda r: hrp.equal_weight(r.shape[1]))
bt_mpt = hrp.rolling_backtest(returns, hrp.mpt_adapter)

summary = pd.DataFrame({
    "HRP": bt_hrp,
    "Equal Weight": bt_eq,
    "MPT": bt_mpt
}).agg(["mean", "std"])

summary.loc["sharpe"] = summary.loc["mean"] / summary.loc["std"]
print(summary)

print(f'HRP: ', end='')
hrp.portfolio_performance(returns, hrp.hrp_adapter(returns))
print(f'Equal Weight: ', end='')
hrp.portfolio_performance(returns, hrp.equal_weight(returns.shape[1]))
print(f'MPT: ', end='')
hrp.portfolio_performance(returns, hrp.mpt_adapter(returns))

(1 + bt_hrp).cumprod().plot(label="HRP")
(1 + bt_eq).cumprod().plot(label="Equal")
(1 + bt_mpt).cumprod().plot(label="MPT")
plt.legend()
plt.show() '''

#tickers = ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "AXISBANK.NS"]
#hrp = HRP()
#returns = hrp.returns(tickers)
#hrp_weights, cov, corr = hrp.hrp_weights(returns)
#print(hrp_weights)
#print(cov)
#print(corr) 
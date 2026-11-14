"""Server-side matplotlib charts returned as base64 PNGs (embedded with data: URLs)."""
import base64
import io

import matplotlib

matplotlib.use("Agg")  # no display on servers
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#00ffcc"
BG = "#05080a"


def _to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color("#335")


def returns_histogram(daily_returns):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.hist(daily_returns * 100, bins=50, color=ACCENT, edgecolor="black", alpha=0.85)
    _style(ax, "Portfolio daily returns", "Daily return (%)", "Days")
    return _to_base64(fig)


def frontier_scatter(vols, rets, sharpes, best, min_vol=None, current=None):
    """best/min_vol/current are (vol, ret) tuples in decimals."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    sc = ax.scatter(vols * 100, rets * 100, c=sharpes, cmap="viridis", s=4, alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe ratio", color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.scatter(best[0] * 100, best[1] * 100, c="#ff3366", marker="*", s=220, label="Max Sharpe", zorder=3)
    if min_vol is not None:
        ax.scatter(min_vol[0] * 100, min_vol[1] * 100, c="#ffcc4d", marker="D", s=70, label="Min volatility", zorder=3)
    if current is not None:
        ax.scatter(current[0] * 100, current[1] * 100, c="white", marker="o", s=70, label="Your portfolio", zorder=3)
    _style(ax, "Efficient frontier (simulated portfolios)", "Volatility (%)", "Expected return (%)")
    ax.legend(facecolor=BG, labelcolor="white", edgecolor="#335")
    return _to_base64(fig)

"""Server-side matplotlib charts returned as base64 PNGs (embedded with data: URLs)."""
import base64
import io

import matplotlib

matplotlib.use("Agg")  # no display on servers
import matplotlib.pyplot as plt  # noqa: E402

GOLD = "#d6b26d"
INK = "#8b95a7"      # mid-grey text that reads on both the dark and light themes
GRID = "#8b95a7"


def _to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax, xlabel, ylabel):
    ax.set_facecolor("none")
    ax.set_xlabel(xlabel, color=INK, labelpad=8)
    ax.set_ylabel(ylabel, color=INK, labelpad=8)
    ax.tick_params(colors=INK, length=0, pad=6)
    ax.grid(color=GRID, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(GRID)
        spine.set_alpha(0.4)


def returns_histogram(daily_returns):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)
    data = daily_returns * 100
    _, bins, patches = ax.hist(data, bins=60, color=GOLD, edgecolor="none", alpha=0.9, rwidth=0.85)
    for left, patch in zip(bins[:-1], patches):
        if left < 0:
            patch.set_facecolor("#f0575d")
            patch.set_alpha(0.75)
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.5, linestyle="--")
    _style(ax, "Daily return (%)", "Days")
    return _to_base64(fig)


def frontier_scatter(vols, rets, sharpes, best, min_vol=None, current=None):
    """best/min_vol/current are (vol, ret) tuples in decimals."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_alpha(0)
    sc = ax.scatter(vols * 100, rets * 100, c=sharpes, cmap="cividis", s=4, alpha=0.7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe ratio", color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_tick_params(color=INK, labelcolor=INK, length=0)
    ax.scatter(best[0] * 100, best[1] * 100, c="#f0575d", marker="*", s=320, label="Max Sharpe",
               zorder=3, edgecolors="white", linewidths=0.8)
    if min_vol is not None:
        ax.scatter(min_vol[0] * 100, min_vol[1] * 100, c="#6b95f2", marker="D", s=80, label="Min volatility",
                   zorder=3, edgecolors="white", linewidths=0.8)
    if current is not None:
        ax.scatter(current[0] * 100, current[1] * 100, c="#3ccf8e", marker="o", s=90, label="Your portfolio",
                   zorder=3, edgecolors="white", linewidths=0.8)
    _style(ax, "Volatility (%)", "Expected return (%)")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    return _to_base64(fig)

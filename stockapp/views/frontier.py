"""Efficient frontier optimiser."""
import logging

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from analytics.charts import frontier_scatter
from analytics.frontier import simulate_frontier, weight_changes
from analytics.market_data import DataError, download_closes
from analytics.portfolio import align_weights

from ..holdings import normalise, parse_csv_fields, read_holdings_file
from ..models import Portfolio

logger = logging.getLogger(__name__)

N_PORTFOLIOS = 30000
DEFAULT_RF = 0.05


def _risk_free(post):
    try:
        rf = float(post.get("risk_free_rate", DEFAULT_RF))
    except (TypeError, ValueError):
        return DEFAULT_RF
    return rf / 100 if rf > 1 else rf  # accept 5 as well as 0.05


def run_optimisation(symbols, weights, risk_free):
    prices = download_closes(symbols)
    prices, w, skipped = align_weights(prices, symbols, weights)
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        raise ValueError("Not enough overlapping price history.")
    sim = simulate_frontier(returns, risk_free=risk_free, n_portfolios=N_PORTFOLIOS, current_weights=w)

    best, mv, cur = sim["best"], sim["min_vol"], sim["current"]
    plot = frontier_scatter(sim["vols"], sim["rets"], sim["sharpes"],
                            best=(best["volatility"], best["return"]),
                            min_vol=(mv["volatility"], mv["return"]),
                            current=(cur["volatility"], cur["return"]))

    def pct(m):
        return {"return": round(m["return"] * 100, 2), "volatility": round(m["volatility"] * 100, 2),
                "sharpe": round(m["sharpe"], 3)}

    ctx = {
        "frontier_plot": plot,
        "portfolio_metrics": pct(best),
        "min_vol_metrics": pct(mv),
        "current_metrics": pct(cur),
        "sharpe_gain": round(best["sharpe"] - cur["sharpe"], 2),
        "weight_changes": weight_changes(dict(zip(prices.columns, w)), sim["best_weights"].to_dict()),
        "message": f"Simulated {N_PORTFOLIOS:,} portfolios across {len(prices.columns)} assets.",
    }
    if skipped:
        ctx["message"] += f" No price data for {', '.join(skipped)} - left out."
    return ctx


@login_required
def efficient_frontier(request):
    ctx = {"portfolios": Portfolio.objects.filter(user=request.user), "risk_free_rate": DEFAULT_RF}
    if request.method != "POST":
        return render(request, "stockapp/efficient_frontier.html", ctx)

    risk_free = _risk_free(request.POST)
    ctx["risk_free_rate"] = risk_free
    try:
        if request.FILES.get("file"):
            symbols, weights = read_holdings_file(request.FILES["file"])
        else:
            symbols, weights = parse_csv_fields(request.POST.get("symbols"), request.POST.get("weights"))
        if len(symbols) < 2:
            raise ValueError("At least two assets are required.")
        ctx.update(run_optimisation(symbols, normalise(weights), risk_free))
    except (DataError, ValueError) as exc:
        ctx["error"] = str(exc)
    except Exception:
        logger.exception("Efficient frontier failed")
        ctx["error"] = "Something went wrong while running the optimisation. Please try again."
    return render(request, "stockapp/efficient_frontier.html", ctx)

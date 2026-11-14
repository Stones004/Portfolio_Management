"""Portfolio analyser (home page): enter/upload holdings, see risk metrics, save portfolios."""
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import redirect, render

from analytics.charts import returns_histogram
from analytics.market_data import DataError, download_closes
from analytics.portfolio import align_weights, portfolio_metrics

from ..forms import StockFormSet
from ..holdings import normalise, parse_csv_fields, read_holdings_file
from ..models import Holding, Portfolio

logger = logging.getLogger(__name__)


def analyze_holdings(symbols, weights):
    """Fetch prices and compute metrics + histogram for a list of holdings."""
    prices = download_closes(symbols)
    prices, w, skipped = align_weights(prices, symbols, weights)
    result = portfolio_metrics(prices, w)
    result["histogram"] = returns_histogram(result.pop("daily_returns").values)
    result["symbols"] = ",".join(prices.columns)
    result["weights"] = ",".join(f"{x:.6f}" for x in w)
    if skipped:
        result["message"] = f"No price data for {', '.join(skipped)} - left out of the analysis."
    return result


@login_required
def home(request):
    portfolios = Portfolio.objects.filter(user=request.user).order_by("-created_at")
    selected = None
    portfolio_id = request.GET.get("portfolio_id")
    if portfolio_id and portfolio_id.isdigit():
        selected = portfolios.filter(id=int(portfolio_id)).first()

    if selected:
        formset = StockFormSet(initial=[{"symbol": h.symbol, "weight": h.weight}
                                        for h in selected.holdings.all()])
    else:
        formset = StockFormSet()

    ctx = {"formset": formset, "portfolios": portfolios, "selected_portfolio": selected}
    if request.method != "POST":
        return render(request, "stockapp/index.html", ctx)

    action = request.POST.get("action")

    if action == "upload_file":
        uploaded = request.FILES.get("file")
        if not uploaded:
            ctx["error"] = "No file selected."
        else:
            try:
                symbols, weights = read_holdings_file(uploaded)
                ctx["formset"] = StockFormSet(initial=[{"symbol": s, "weight": w}
                                                       for s, w in zip(symbols, normalise(weights))])
                ctx["message"] = f"Loaded {len(symbols)} stocks from {uploaded.name}."
            except Exception as exc:  # pandas raises many error types for bad files
                ctx["error"] = f"Could not read file: {exc}"
        return render(request, "stockapp/index.html", ctx)

    if action == "analyze":
        formset = StockFormSet(request.POST)
        ctx["formset"] = formset
        if not formset.is_valid():
            ctx["error"] = "Please fix the highlighted rows."
            return render(request, "stockapp/index.html", ctx)
        rows = [f.cleaned_data for f in formset
                if f.cleaned_data and not f.cleaned_data.get("DELETE")]
        if not rows:
            ctx["error"] = "Add at least one stock."
            return render(request, "stockapp/index.html", ctx)
        try:
            symbols = [r["symbol"].upper().strip() for r in rows]
            weights = normalise([r["weight"] for r in rows])
            ctx.update(analyze_holdings(symbols, weights))
        except (DataError, ValueError) as exc:
            ctx["error"] = str(exc)
        except Exception:
            logger.exception("Portfolio analysis failed")
            ctx["error"] = "Something went wrong while analysing the portfolio. Please try again."
        return render(request, "stockapp/index.html", ctx)

    if action == "save_portfolio":
        name = (request.POST.get("portfolio_name") or "").strip()[:100]
        try:
            symbols, weights = parse_csv_fields(request.POST.get("symbols"), request.POST.get("weights"))
        except ValueError as exc:
            ctx["error"] = str(exc)
            return render(request, "stockapp/index.html", ctx)
        if not name:
            ctx["error"] = "Please give the portfolio a name."
            return render(request, "stockapp/index.html", ctx)
        with transaction.atomic():
            portfolio = Portfolio.objects.create(user=request.user, name=name)
            Holding.objects.bulk_create([Holding(portfolio=portfolio, symbol=s, weight=w)
                                         for s, w in zip(symbols, weights)])
        messages.success(request, f"Saved portfolio “{name}”.")
        return redirect(f"{request.path}?portfolio_id={portfolio.id}")

    if action == "delete_portfolio":
        pid = request.POST.get("portfolio_id", "")
        if pid.isdigit():
            Portfolio.objects.filter(user=request.user, id=int(pid)).delete()
            messages.success(request, "Portfolio deleted.")
        return redirect("home")

    ctx["error"] = "Unknown action."
    return render(request, "stockapp/index.html", ctx)

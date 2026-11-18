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


def _holdings_from_formset(formset):
    """Validated (symbols, weights) from the holdings formset; duplicate tickers are combined."""
    if not formset.is_valid():
        raise ValueError("Please fix the highlighted rows.")
    combined = {}
    for f in formset:
        data = f.cleaned_data
        if data and not data.get("DELETE"):
            symbol = data["symbol"].upper().strip()
            combined[symbol] = combined.get(symbol, 0.0) + data["weight"]
    if not combined:
        raise ValueError("Add at least one stock.")
    return list(combined), normalise(list(combined.values()))


def _create_holdings(portfolio, symbols, weights):
    Holding.objects.bulk_create([Holding(portfolio=portfolio, symbol=s, weight=w)
                                 for s, w in zip(symbols, weights)])


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

    if action in ("save_holdings", "update_portfolio"):
        # Save the holdings form as a new portfolio, or overwrite the selected one - no analysis needed.
        formset = StockFormSet(request.POST)
        ctx["formset"] = formset
        try:
            symbols, weights = _holdings_from_formset(formset)
        except ValueError as exc:
            ctx["error"] = str(exc)
            return render(request, "stockapp/index.html", ctx)
        if action == "update_portfolio":
            pid = request.POST.get("portfolio_id", "")
            portfolio = portfolios.filter(id=int(pid)).first() if pid.isdigit() else None
            if portfolio is None:
                ctx["error"] = "That portfolio no longer exists."
                return render(request, "stockapp/index.html", ctx)
            with transaction.atomic():
                portfolio.holdings.all().delete()
                _create_holdings(portfolio, symbols, weights)
            messages.success(request, f"Updated portfolio “{portfolio.name}”.")
        else:
            name = (request.POST.get("new_portfolio_name") or "").strip()[:100]
            if not name:
                ctx["error"] = "Give the portfolio a name to save it."
                return render(request, "stockapp/index.html", ctx)
            with transaction.atomic():
                portfolio = Portfolio.objects.create(user=request.user, name=name)
                _create_holdings(portfolio, symbols, weights)
            messages.success(request, f"Saved portfolio “{name}”.")
        return redirect(f"{request.path}?portfolio_id={portfolio.id}")

    if action == "analyze":
        formset = StockFormSet(request.POST)
        ctx["formset"] = formset
        try:
            symbols, weights = _holdings_from_formset(formset)
        except ValueError as exc:
            ctx["error"] = str(exc)
            return render(request, "stockapp/index.html", ctx)
        try:
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
            _create_holdings(portfolio, symbols, weights)
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

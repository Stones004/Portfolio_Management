"""Portfolio valuation: run the DCF on every holding of a saved portfolio.

The page lists the holdings with any valuation already saved. The browser then
values the remaining stocks one at a time through `dcf_value_api`, so a long
portfolio never hits the server's request timeout.
"""
import logging

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from analytics.dcf import DCF_Valuation, DCFError
from analytics.market_data import DataError

from ..models import Portfolio, Valuation
from .dcf import save_valuation

logger = logging.getLogger(__name__)


@login_required
def portfolio_dcf(request):
    portfolios = Portfolio.objects.filter(user=request.user).order_by("-created_at")
    selected = None
    pid = request.GET.get("portfolio_id", "")
    if pid.isdigit():
        selected = portfolios.filter(id=int(pid)).first()
    if selected is None:
        selected = portfolios.first()

    rows = []
    if selected:
        holdings = list(selected.holdings.all())
        saved = {v.symbol: v for v in Valuation.objects.filter(
            user=request.user, symbol__in=[h.symbol for h in holdings])}
        total = sum(h.weight for h in holdings) or 1.0
        for h in holdings:
            v = saved.get(h.symbol)
            rows.append({"symbol": h.symbol, "weight": h.weight / total,
                         "valuation": v.as_dict() if v else None})
    return render(request, "stockapp/portfolio_dcf.html", {
        "portfolios": portfolios, "selected_portfolio": selected, "rows": rows,
    })


@login_required
@require_POST
def dcf_value_api(request):
    """Value one ticker with model defaults; save and return the result as JSON."""
    ticker = (request.POST.get("ticker") or "").strip().upper()[:30]
    if not ticker:
        return JsonResponse({"ok": False, "error": "No ticker given."}, status=400)
    try:
        res = DCF_Valuation(ticker).run(None)
        valuation = save_valuation(request.user, ticker, res, custom=False)
        return JsonResponse({"ok": True, "valuation": valuation.as_dict()})
    except (DCFError, DataError, ValueError) as exc:
        return JsonResponse({"ok": False, "error": str(exc)})
    except Exception:
        logger.exception("DCF failed for %s", ticker)
        return JsonResponse({"ok": False, "error": "Required data is missing or incomplete from the data provider."})

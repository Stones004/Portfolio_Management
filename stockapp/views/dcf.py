"""DCF valuation page."""
import logging
import math

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from analytics.dcf import DCF_Valuation, DCFError
from analytics.industry import profile_choices
from analytics.market_data import DataError

logger = logging.getLogger(__name__)

# (assumption key, is_percent) - same keys are used as form field names
DCF_FIELDS = [
    ("discount_rate", True), ("growth_rate", True), ("reinvestment_rate", True),
    ("terminal_growth", True), ("terminal_return", True),
    ("high_growth_years", False), ("fade_years", False),
]
INTEGER_FIELDS = {"high_growth_years", "fade_years"}

DCF_LABELS = {
    "fcff": {
        "discount_rate": "WACC (%)",
        "growth_rate": "High-growth rate in operating income (%)",
        "reinvestment_rate": "Reinvestment rate (% of NOPAT)",
        "terminal_growth": "Terminal growth (%)",
        "terminal_return": "Terminal ROIC (%)",
        "high_growth_years": "High-growth years",
        "fade_years": "Fade years (growth slides to terminal)",
    },
    "fcfe": {
        "discount_rate": "Cost of equity (%)",
        "growth_rate": "High-growth rate in net income (%)",
        "reinvestment_rate": "Retention ratio (% of net income)",
        "terminal_growth": "Terminal growth (%)",
        "terminal_return": "Terminal ROE (%)",
        "high_growth_years": "High-growth years",
        "fade_years": "Fade years (growth slides to terminal)",
    },
}

PERCENT_COLUMNS = {"Growth", "Tax Rate", "Reinvestment Rate", "Retention", "ROIC", "ROE",
                   "Operating Margin", "Reinvestment_Rate", "Payout"}


def _money_unit(ccy):
    return (1e7, "Cr") if ccy == "INR" else (1e6, "M")


def _is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt_money(x, ccy):
    if _is_missing(x):
        return "-"
    div, unit = _money_unit(ccy)
    return f"{x / div:,.0f} {unit}"


def _fmt_cell(col, val, ccy):
    if isinstance(val, str):
        return val
    if _is_missing(val):
        return "-"
    if col in PERCENT_COLUMNS:
        return f"{val:.1%}"
    if col == "Discount Factor":
        return f"{val:.4f}"
    return f"{val / _money_unit(ccy)[0]:,.0f}"


def _table(df, ccy, index_label=None):
    cols = list(df.columns)
    rows = []
    for idx, row in df.iterrows():
        cells = [_fmt_cell(c, row[c], ccy) for c in cols]
        if index_label:
            cells.insert(0, idx.strftime("%b %Y") if hasattr(idx, "strftime") else str(idx))
        rows.append(cells)
    columns = ([index_label] if index_label else []) + [c.replace("_", " ") for c in cols]
    return {"columns": columns, "rows": rows}


def _to_form(assumptions):
    # 2 decimals for display; unchanged fields are matched back to the exact model value in dcf_view
    return {k: (round(assumptions[k] * 100, 2) if pct else int(assumptions[k]))
            for k, pct in DCF_FIELDS}


def _from_form(post):
    out = {}
    for key, pct in DCF_FIELDS:
        raw = str(post.get(key, "")).strip()
        if raw == "":
            continue
        try:
            out[key] = float(raw) / 100 if pct else int(float(raw))
        except ValueError:
            raise DCFError(f"'{raw}' is not a valid number for {key.replace('_', ' ')}.")
    return out


def _present(res):
    ccy = res["currency"]
    cap = res["capital"]
    price = res["current_price"]
    if res["model"] == "fcff":
        hist_cols = ["Revenue", "Operating Income", "Operating Margin", "Tax Rate", "NOPAT",
                     "Net Capex", "Change in WC", "Reinvestment", "Reinvestment_Rate",
                     "Invested Capital", "ROIC"]
    else:
        hist_cols = ["Net Income", "Book Equity", "ROE", "Dividends", "Buybacks", "Payout", "Retention"]

    capital_rows = [("Risk-free rate", f"{cap['risk_free']:.2%}"),
                    ("Equity risk premium", f"{cap['erp']:.2%}"),
                    ("Beta", f"{cap['beta']:.2f} · {cap['beta_source']}"),
                    ("Cost of equity", f"{cap['cost_of_equity']:.2%}")]
    if res["model"] == "fcff":
        cov = cap["interest_coverage"]
        capital_rows += [
            ("Synthetic rating", f"{cap['rating']} (interest coverage {cov:.1f}x)"
             if math.isfinite(cov) else f"{cap['rating']} (no interest expense)"),
            ("Pre-tax cost of debt", f"{cap['pre_tax_cost_of_debt']:.2%}"),
            ("Marginal tax rate", f"{cap['tax_rate']:.2%}"),
            ("Debt / equity weights", f"{cap['debt_weight']:.1%} / {cap['equity_weight']:.1%}"),
            ("WACC", f"{cap['wacc']:.2%}"),
        ]

    ev = res["enterprise_value"]
    upside = res["upside"]
    if upside is None:
        verdict, verdict_class = None, ""
    elif upside > 0.10:
        verdict, verdict_class = "Undervalued", "up"
    elif upside < -0.10:
        verdict, verdict_class = "Overvalued", "down"
    else:
        verdict, verdict_class = "Fairly valued", "fair"
    term = res["terminal_share"]
    terminal_pct = round(min(max(term, 0.0), 1.0) * 100, 1) if term is not None else 0
    if upside is None:
        upside_text = ""
    elif upside >= 0:
        upside_text = f"Intrinsic value is {upside:.1%} above price"
    else:
        upside_text = f"Intrinsic value is {-upside:.1%} below price"
    return {
        "upside_text": upside_text,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "terminal_pct": terminal_pct,
        "forecast_pct": round(100 - terminal_pct, 1),
        "name": res["name"],
        "model": res["model"],
        "profile_label": res["profile"]["label"],
        "profile_note": res["profile"]["note"],
        "sector": res["sector"],
        "industry": res["industry"],
        "currency": ccy,
        "unit": _money_unit(ccy)[1],
        "enterprise_value": _fmt_money(ev, ccy) if ev is not None else None,
        "equity_value": _fmt_money(res["equity_value"], ccy),
        "per_share_value": f"{res['per_share_value']:,.2f}",
        "current_price": f"{price:,.2f}" if price else "-",
        "upside": f"{res['upside']:+.1%}" if res["upside"] is not None else None,
        "upside_positive": (res["upside"] or 0) >= 0,
        "terminal_share": f"{res['terminal_share']:.0%}" if res["terminal_share"] is not None else "-",
        "bridge": [(label, _fmt_money(v, ccy)) for label, v in res["bridge"]],
        "capital": capital_rows,
        "forecast": _table(res["dcf_table"], ccy),
        "history": _table(res["history"][[c for c in hist_cols if c in res["history"].columns]],
                          ccy, index_label="Fiscal year"),
        "warnings": res["warnings"],
        "growth_sources": [(label, f"{value:.1%}", f"{weight:.0%}")
                           for label, value, weight in res["growth_sources"]],
        "blended_growth": f"{res['blended_growth']:.1%}",
    }


@login_required
def dcf_view(request):
    ctx = {"ticker": "", "industry_choices": profile_choices(), "industry": "",
           "detected_industry": "", "result": None, "error": None, "show_inputs": False,
           "fields": []}

    if request.method != "POST":
        return render(request, "stockapp/dcf.html", ctx)

    ticker = (request.POST.get("ticker") or "").strip().upper()
    industry = request.POST.get("industry") or ""
    ctx.update(ticker=ticker, industry=industry)
    saved = request.session.get("dcf_state") or {}

    try:
        model = DCF_Valuation(ticker, industry=industry or None)
        # Only a re-run when ticker and model are unchanged; otherwise the
        # previous company's assumptions would be applied to this one.
        is_rerun = ("discount_rate" in request.POST
                    and saved.get("ticker") == ticker
                    and saved.get("profile") == model.profile["key"])

        if is_rerun:
            # Keep the user's inputs on screen if validation fails below.
            labels = DCF_LABELS[model.model]
            original = saved.get("original", {})
            ctx.update(show_inputs=True, industry=model.profile["key"],
                       fields=[(k, labels[k], original.get(k, ""), request.POST.get(k, ""), k in INTEGER_FIELDS)
                               for k, _ in DCF_FIELDS])

        overrides = None
        if is_rerun:
            # Fields left at the (rounded) model value use the exact model value,
            # so an unchanged re-run reproduces the first run to the paisa/cent.
            original_dec = _from_form(saved.get("original", {}))
            overrides = {k: v for k, v in _from_form(request.POST).items()
                         if k not in original_dec or not math.isclose(v, original_dec[k], abs_tol=1e-9)}
        res = model.run(overrides)

        labels = DCF_LABELS[res["model"]]
        original = _to_form(res["defaults"])
        inputs = _to_form(res["assumptions"])
        request.session["dcf_state"] = {"ticker": ticker, "profile": model.profile["key"],
                                        "original": original}
        ctx.update(
            industry=model.profile["key"],
            detected_industry=model.detected_industry,
            result=_present(res),
            show_inputs=True,
            fields=[(k, labels[k], original[k], inputs[k], k in INTEGER_FIELDS) for k, _ in DCF_FIELDS],
        )
    except (DCFError, DataError, ValueError) as exc:
        ctx["error"] = str(exc)
    except Exception:
        logger.exception("DCF failed for %s", ticker)
        ctx["error"] = (f"Could not value {ticker}: required data is missing or incomplete "
                        "from the data provider.")

    return render(request, "stockapp/dcf.html", ctx)

"""Live ticker strip: latest prices for a fixed watchlist per market, polled by the browser."""
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from analytics.market_data import live_quotes

# (symbol, label, is_index). Indices first, then the largest, most-traded stocks.
WATCHLISTS = {
    "IN": {
        "name": "India",
        "currency": "INR",
        "symbols": [
            ("^NSEI", "NIFTY 50", True), ("^BSESN", "SENSEX", True), ("^NSEBANK", "BANK NIFTY", True),
            ("RELIANCE.NS", "RELIANCE", False), ("HDFCBANK.NS", "HDFCBANK", False),
            ("TCS.NS", "TCS", False), ("INFY.NS", "INFY", False), ("ICICIBANK.NS", "ICICIBANK", False),
            ("BHARTIARTL.NS", "BHARTIARTL", False), ("SBIN.NS", "SBIN", False), ("ITC.NS", "ITC", False),
            ("LT.NS", "LT", False), ("HINDUNILVR.NS", "HINDUNILVR", False),
            ("BAJFINANCE.NS", "BAJFINANCE", False), ("MARUTI.NS", "MARUTI", False),
            ("SUNPHARMA.NS", "SUNPHARMA", False), ("TATASTEEL.NS", "TATASTEEL", False),
            ("ASIANPAINT.NS", "ASIANPAINT", False),
        ],
    },
    "US": {
        "name": "United States",
        "currency": "USD",
        "symbols": [
            ("^GSPC", "S&P 500", True), ("^IXIC", "NASDAQ", True), ("^DJI", "DOW", True),
            ("AAPL", "AAPL", False), ("MSFT", "MSFT", False), ("NVDA", "NVDA", False),
            ("AMZN", "AMZN", False), ("GOOGL", "GOOGL", False), ("META", "META", False),
            ("TSLA", "TSLA", False), ("JPM", "JPM", False), ("V", "V", False),
            ("BRK-B", "BRK.B", False), ("LLY", "LLY", False), ("XOM", "XOM", False),
            ("WMT", "WMT", False),
        ],
    },
}


@login_required
@require_GET
def ticker_quotes(request):
    market = (request.GET.get("market") or "IN").upper()
    watch = WATCHLISTS.get(market)
    if watch is None:
        return JsonResponse({"ok": False, "error": "Unknown market."}, status=400)
    quotes = live_quotes([s for s, _, _ in watch["symbols"]])
    items = []
    for symbol, label, is_index in watch["symbols"]:
        q = quotes.get(symbol.upper())
        if q:
            items.append(dict(q, symbol=symbol, label=label, index=is_index))
    return JsonResponse({"ok": bool(items), "market": market, "currency": watch["currency"],
                         "items": items,
                         "error": None if items else "Live prices are unavailable right now."})

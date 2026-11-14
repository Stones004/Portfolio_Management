"""
Industry classification and Damodaran-style valuation profiles.

Damodaran's point is that one DCF template does not fit every business:

* Banks / insurers / NBFCs   - debt is raw material, not financing, so FCFF and
                               WACC are meaningless. Value equity directly with
                               FCFE at the cost of equity, where reinvestment is
                               the equity retained to grow the book.
* Cyclicals / commodities   - the latest year may be a peak or trough. Use
                               normalised (multi-year average) margins and
                               returns, and assume no excess returns forever.
* Regulated utilities       - returns are capped near the cost of capital; low
                               growth, little excess return.
* Technology / brands       - can sustain excess returns (moats), so terminal
                               ROIC may stay above WACC, and growth can be higher.

Each profile below sets the default assumptions for one of these groups. All
values are starting points the user can override in the UI.

  max_growth       the highest growth rate a business in this industry typically
                   sustains. Faster growers are NOT cut down to it: they keep the
                   rate for more years instead ("longer, not higher", see dcf.py).
  terminal_spread  how far terminal ROIC (ROE for banks) may stay above the cost
                   of capital forever - a proxy for how durable the moat is.
  normalize        use multi-year average margins/returns instead of the latest year.
"""

PROFILES = {
    "general": {
        "label": "General (non-financial)",
        "model": "fcff",
        "max_growth": 0.15,
        "terminal_spread": 0.02,   # terminal ROIC = WACC + spread (capped at current ROIC)
        "normalize": False,
        "note": "Standard FCFF: growth = reinvestment rate x ROIC, fading to stable growth.",
    },
    "technology": {
        "label": "Technology",
        "model": "fcff",
        "max_growth": 0.25,
        "terminal_spread": 0.04,
        "normalize": False,
        "note": "Higher growth ceiling; moats allow terminal ROIC to stay above WACC.",
    },
    "consumer_defensive": {
        "label": "Consumer staples / brands",
        "model": "fcff",
        "max_growth": 0.15,
        "terminal_spread": 0.04,
        "normalize": False,
        "note": "Brand moats sustain excess returns in stable growth.",
    },
    "healthcare": {
        "label": "Healthcare / pharma",
        "model": "fcff",
        "max_growth": 0.18,
        "terminal_spread": 0.03,
        "normalize": False,
        "note": "Patents/R&D give moderate long-run excess returns.",
    },
    "cyclical": {
        "label": "Cyclical / commodity",
        "model": "fcff",
        "max_growth": 0.10,
        "terminal_spread": 0.0,
        "normalize": True,
        "note": "Normalised margins and ROIC (multi-year average); no excess returns in stable growth.",
    },
    "utility": {
        "label": "Regulated utility",
        "model": "fcff",
        "max_growth": 0.08,
        "terminal_spread": 0.0,
        "normalize": True,
        "note": "Regulated returns ~ cost of capital; low growth.",
    },
    "real_estate": {
        "label": "Real estate",
        "model": "fcff",
        "max_growth": 0.12,
        "terminal_spread": 0.0,
        "normalize": True,
        "note": "Lumpy project cash flows are normalised over available years.",
    },
    "financial": {
        "label": "Financial (bank / insurer / NBFC)",
        "model": "fcfe",
        "max_growth": 0.15,
        "terminal_spread": 0.01,   # terminal ROE = cost of equity + spread
        "normalize": False,
        "note": "FCFE at cost of equity: growth = retention x ROE; debt/deposits are operating, not financing.",
    },
}

_FINANCIAL_KEYWORDS = (
    "bank", "insurance", "credit services", "capital markets", "asset management",
    "financial conglomerates", "mortgage finance", "shell companies",
)
# Financial Services industries that are fee businesses and fine with FCFF
_FINANCIAL_FCFF_EXCEPTIONS = ("insurance brokers", "financial data", "stock exchanges")

_CYCLICAL_INDUSTRIES = (
    "steel", "aluminum", "copper", "other industrial metals", "gold", "silver",
    "other precious metals", "coking coal", "thermal coal", "uranium",
    "oil & gas", "chemicals", "specialty chemicals", "agricultural inputs",
    "marine shipping", "auto manufacturers", "auto parts", "building materials",
    "paper & paper products", "lumber", "airlines", "farm & heavy construction",
    "metal fabrication", "textile manufacturing",
)

_SECTOR_MAP = {
    "Technology": "technology",
    "Communication Services": "general",
    "Consumer Defensive": "consumer_defensive",
    "Consumer Cyclical": "general",
    "Healthcare": "healthcare",
    "Industrials": "general",
    "Basic Materials": "cyclical",
    "Energy": "cyclical",
    "Utilities": "utility",
    "Real Estate": "real_estate",
    "Financial Services": "general",
}


def classify(sector, industry):
    """Map a yfinance sector/industry pair to a profile key."""
    s = (sector or "").strip()
    i = (industry or "").strip().lower()

    if s == "Financial Services" or any(k in i for k in _FINANCIAL_KEYWORDS):
        if not any(k in i for k in _FINANCIAL_FCFF_EXCEPTIONS):
            return "financial"
    if any(k in i for k in _CYCLICAL_INDUSTRIES):
        return "cyclical"
    return _SECTOR_MAP.get(s, "general")


def get_profile(key):
    if key not in PROFILES:
        raise ValueError(f"Unknown industry profile '{key}'")
    return dict(PROFILES[key], key=key)


def profile_choices():
    return [(k, v["label"]) for k, v in PROFILES.items()]


# ---------------------------------------------------------------------------
# Bottom-up betas
# ---------------------------------------------------------------------------
# Approximate unlevered (cash-corrected) betas by industry, in the spirit of
# Damodaran's annual "Betas by Sector" dataset (pages.stern.nyu.edu/~adamodar).
# These are rounded starting points -- refresh them from his latest dataset.
# For financials these are LEVERED equity betas (D/E is not meaningful).
UNLEVERED_BETA_BY_INDUSTRY = {
    "software": 1.20, "semiconductor": 1.40, "information technology services": 1.00,
    "consumer electronics": 1.10, "communication equipment": 1.00,
    "telecom services": 0.60, "internet content": 1.10,
    "steel": 0.95, "aluminum": 1.00, "chemicals": 0.90, "agricultural inputs": 0.80,
    "oil & gas": 0.85, "coal": 0.85, "marine shipping": 0.70, "auto manufacturers": 0.85,
    "auto parts": 0.90, "building materials": 0.80, "airlines": 0.80,
    "drug manufacturers": 0.90, "biotechnology": 1.10, "medical": 0.85,
    "tobacco": 0.60, "packaged foods": 0.60, "household": 0.65, "beverages": 0.65,
    "utilities": 0.45, "real estate": 0.65, "engineering & construction": 0.85,
    "aerospace & defense": 0.85,
    # financials (levered)
    "bank": 0.95, "insurance": 0.85, "credit services": 1.10,
    "capital markets": 1.10, "asset management": 1.00,
}

UNLEVERED_BETA_BY_SECTOR = {
    "Technology": 1.10, "Communication Services": 0.80, "Consumer Defensive": 0.65,
    "Consumer Cyclical": 0.90, "Healthcare": 0.90, "Industrials": 0.90,
    "Basic Materials": 0.90, "Energy": 0.85, "Utilities": 0.45,
    "Real Estate": 0.65, "Financial Services": 1.00,
}


def industry_beta(sector, industry):
    """Return (beta, source_label) or (None, None) if no match."""
    i = (industry or "").lower()
    for key, beta in UNLEVERED_BETA_BY_INDUSTRY.items():
        if key in i:
            return beta, f"industry '{industry}'"
    if sector in UNLEVERED_BETA_BY_SECTOR:
        return UNLEVERED_BETA_BY_SECTOR[sector], f"sector '{sector}'"
    return None, None

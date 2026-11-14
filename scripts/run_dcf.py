"""Value one or more tickers from the command line.

    python -m scripts.run_dcf INFY.NS HDFCBANK.NS
    python -m scripts.run_dcf INFY.NS --growth 0.08 --wacc 0.11

Omitted assumptions use the model defaults.
"""
import argparse

from analytics.dcf import DCF_Valuation, DCFError, print_valuation
from analytics.industry import PROFILES
from analytics.market_data import DataError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--industry", choices=sorted(PROFILES), help="override the auto-detected model")
    ap.add_argument("--wacc", type=float, dest="discount_rate", help="discount rate, e.g. 0.11")
    ap.add_argument("--growth", type=float, dest="growth_rate")
    ap.add_argument("--reinvestment", type=float, dest="reinvestment_rate")
    ap.add_argument("--terminal-growth", type=float, dest="terminal_growth")
    ap.add_argument("--terminal-return", type=float, dest="terminal_return")
    ap.add_argument("--high-years", type=int, dest="high_growth_years")
    ap.add_argument("--fade-years", type=int, dest="fade_years")
    args = vars(ap.parse_args())

    tickers, industry = args.pop("tickers"), args.pop("industry")
    for tk in tickers:
        try:
            print_valuation(DCF_Valuation(tk, industry=industry).run(args))
        except (DCFError, DataError) as exc:
            print(f"{tk}: {exc}")


if __name__ == "__main__":
    main()

"""Screen a list of tickers by DCF upside.  Run: python -m scripts.dcf_screener"""
from analytics.market_data import DataError
from analytics.dcf import DCF_Valuation, DCFError

nifty_500_full = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "NESTLEIND.NS", "HCLTECH.NS", "DMART.NS", "ULTRACEMCO.NS",
    "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "ASHOKLEY.NS", "TATAMOTORS.NS",
    "JSWSTEEL.NS", "WIPRO.NS", "BAJFINANCE.NS", "COALINDIA.NS", "HINDALCO.NS",
    "GRASIM.NS", "LTIM.NS", "TATASTEEL.NS", "CIPLA.NS", "ONGC.NS",
    "EICHERMOT.NS", "BAJAJFINSV.NS", "TATACONSUM.NS", "BRITANNIA.NS",
    "TRENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "BPCL.NS",
    "DRREDDY.NS", "KOTAKBANK.NS", "HEROMOTOCO.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS",
    "ADANIENT.NS", "UPL.NS", "SHRIRAMFIN.NS", "PIDILITIND.NS", "M&M.NS",
    "GODREJCP.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SRTRANSFIN.NS",
    "TATACOMM.NS", "HAVELLS.NS", "CHOLAFIN.NS", "TORNTPOWER.NS", "AMBUJACEM.NS",
    "HAL.NS", "ZYDUSLIFE.NS", "VARUNBEV.NS", "ASTRAL.NS", "POLYCAB.NS",
    "BEL.NS", "OBEROIRLTY.NS", "DABUR.NS", "MPHASIS.NS", "SAIL.NS",
    "ACC.NS", "LTTS.NS", "IOC.NS", "PNB.NS", "CANBK.NS",
    "PERSISTENT.NS", "DLF.NS", "NAUKRI.NS", "INDIGO.NS", "IDFCFIRSTB.NS",
    "LICI.NS", "BHARATFORG.NS", "JINDALSTEL.NS", "ABB.NS", "COLPAL.NS",
    "GAIL.NS", "COROMANDEL.NS", "APOLLOTYRE.NS", "LUPIN.NS", "JUBLFOOD.NS",
    "PATANJALI.NS", "AUBANK.NS", "MINDTREE.NS", "KPITTECH.NS", "METROPOLIS.NS"
]

nifty_smallcap_250 = ["BLS.NS", "SHARDAMOTR.NS", "NATIONALUM.NS", "KEI.NS", "JYOTHYLAB.NS"]

us_stocks_dcf = [
    # S&P 500 Magnificent 7 + Classics
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    
    # **Analyst Undervalued (Dec 2025)** [web:165][web:166]
    "PYPL", "PFE", "CMCSA", "CVS", "ZM", "AMP", "FI", "ADBE",
    
    # Value + Dividend Plays
    "T", "VZ", "IBM", "INTC", "CSCO", "ORCL", "WMT", "KO",
    
    # Financials + Energy
    "JPM", "BAC", "WFC", "GS", "XOM", "CVX", "SLB",
    
    # Healthcare + Consumer
    "JNJ", "PFE", "ABBV", "MRK", "NKE", "SBUX", "MCD",
    
    # Tech Value
    "CRM", "NOW", "SNPS", "CDNS", "PLTR", "CRWD"
]

if __name__ == "__main__":
    print(f"US Stocks Array: {len(us_stocks_dcf)} ready for DCF screening")
    print(f"NIFTY Smallcap 250 Array: {len(nifty_smallcap_250)} stocks ready for DCF screening")


def screen(tickers):
    upside, downside = [], []
    for ticker in tickers:
        try:
            res = DCF_Valuation(ticker).run()
        except (DCFError, DataError, ValueError) as exc:
            print(f"{ticker}: skipped ({exc})")
            continue
        if res["upside"] is None:
            continue
        row = (ticker, res["currency"], res["per_share_value"], res["upside"], res["profile"]["label"])
        (upside if res["upside"] > 0 else downside).append(row)
    upside.sort(key=lambda x: x[3], reverse=True)
    downside.sort(key=lambda x: x[3])
    return upside, downside


if __name__ == "__main__":
    buys, sells = screen(us_stocks_dcf[:5])
    print("TOP 10 BUYS:")
    for ticker, ccy, target, up, label in buys[:10]:
        print(f"{ticker:<12} {ccy} {target:,.2f} ({up:+.0%})  [{label}]")
    print("TOP 10 SELLS:")
    for ticker, ccy, target, up, label in sells[:10]:
        print(f"{ticker:<12} {ccy} {target:,.2f} ({up:+.0%})  [{label}]")

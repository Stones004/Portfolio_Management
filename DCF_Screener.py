import WACC as wacc
import mydcf as mydcf
import yfinance as yf

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

print(f"US Stocks Array: {len(us_stocks_dcf)} ready for DCF screening")

print(f"NIFTY Smallcap 250 Array: {len(nifty_smallcap_250)} stocks ready for DCF screening")


portfolio_upside = []
portfolio_downside = []
for ticker in us_stocks_dcf[:2]:  # Test first 50
    try:
        model = mydcf.DCF_Valuation(ticker)
        wacc_val = wacc.WACCModel(ticker).wacc()
        ev, _ = model.project_fcff(wacc_val)
        if ev:
            eq, ps = model.equity_value(ev)
            market_price = yf.Ticker(ticker).info.get('currentPrice', 0)
            upside = (ps - market_price) / market_price * 100 if market_price > 0 else 0
            downside = (market_price - ps) / market_price * 100 if market_price > 0 else 0
            if upside > 0:  # Top buys only
                portfolio_upside.append((ticker, ps, upside))
            elif downside > 0:  # Top sells only
                portfolio_downside.append((ticker, ps, downside))
    except:
        continue

# Sort by upside
portfolio_upside.sort(key=lambda x: x[2], reverse=True)
print("TOP 10 BUYS:")
for ticker, target, upside in portfolio_upside[:10]:
    print(f"{ticker}: ₹{target:.0f} (+{upside:.0f}%)")

# Sort by downside
portfolio_downside.sort(key=lambda x: x[2], reverse=True)
print("TOP 10 SELLS:")
for ticker, target, downside in portfolio_downside[:10]:
    print(f"{ticker}: ₹{target:.0f} (-{downside:.0f}%)")

# views.py (fixed)
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import StockFormSet
import yfinance as yf


from .models import Portfolio, Holding
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login

def _ensure_df_close(data):
    """
    Ensure yf.download output has a DataFrame with 'Close' columns.
    yfinance may return a Series for single ticker - normalize to DataFrame.
    Return: DataFrame of close prices (columns: tickers)
    """
    if data is None:
        return pd.DataFrame()
    # if 'Close' present as level in MultiIndex columns (when download with multiple fields)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        # sometimes data has single-level columns where 'Close' is a column name
        close = data['Close']
    else:
        # If user passed tickers and we get a DataFrame with columns already being tickers (most likely)
        # Try to extract close prices using column name 'Close' in case of multi-level
        try:
            close = data['Close']
        except Exception:
            # If data itself is the Close prices (common)
            close = data

    # If it's a Series (single ticker), convert to DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame(name=str(close.name) if close.name else '0')

    # Drop any columns that are not numeric
    close = close.select_dtypes(include=[np.number])
    return close.dropna(how='all')


def analyze_portfolio(stocks, weights):
    """
    Fetch data, compute returns, volatility, sharpe and histogram image (base64).
    """
    try:
        raw = yf.download(stocks, period="5y", progress=False)
        data = _ensure_df_close(raw)
        if data.empty:
            raise ValueError("No closing price data returned for the given tickers.")
    except Exception as e:
        raise ValueError(f"Error fetching data from yfinance: {e}")

    # Align stocks order to data columns (in case some tickers were missing)
    available_tickers = list(data.columns)
    if not available_tickers:
        raise ValueError("No valid tickers found in the fetched data.")

    # Build arrays for only the available tickers
    try:
        # Map requested stocks to available columns (case-insensitive)
        col_map = {c.upper(): c for c in data.columns}
        chosen_cols = []
        chosen_weights = []
        for s, w in zip(stocks, weights):
            if s.upper() in col_map:
                chosen_cols.append(col_map[s.upper()])
                chosen_weights.append(w)
            else:
                # skip missing ticker
                print(f"Ticker {s} not found in fetched data — skipping.")
        if not chosen_cols:
            raise ValueError("None of the requested tickers returned data.")
        data = data[chosen_cols]
        weights = np.array(chosen_weights, dtype=float)
    except Exception as e:
        raise ValueError(f"Error aligning tickers to data: {e}")

    # Normalize weights safely
    sum_w = np.sum(weights)
    if sum_w == 0 or np.isnan(sum_w):
        raise ValueError("Sum of weights is zero or invalid.")
    weights = weights / sum_w

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError("Not enough price history to compute returns.")

    # Annualize
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252

    # Portfolio return & volatility
    portfolio_return = float(np.dot(weights, mean_returns))
    portfolio_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    # Sharpe (assume risk-free = 5)
    sharpe_ratio = float((portfolio_return - 0.05) / portfolio_volatility) if portfolio_volatility != 0 else 0.0

    # Portfolio daily returns for histogram and risk metrics
    portfolio_daily_returns = np.dot(daily_returns.values, weights)
    # handle if portfolio_daily_returns empty
    if portfolio_daily_returns.size == 0:
        var_95 = 0.0
        cvar_95 = 0.0
    else:
        # Value at Risk (VaR) at 95% confidence - potential loss (positive number)
        var_95 = float(-np.percentile(portfolio_daily_returns, 5) * 100)
        # Conditional VaR (CVaR/Expected Shortfall) - average loss beyond VaR
        worst_5_percent = portfolio_daily_returns[portfolio_daily_returns <= np.percentile(portfolio_daily_returns, 5)]
        cvar_95 = float(-np.mean(worst_5_percent) * 100) if len(worst_5_percent) > 0 else 0.0

    # Generate histogram image (base64)
    plt.figure(figsize=(8, 4))
    plt.hist(portfolio_daily_returns, bins=50, color='#00ffcc', edgecolor='black', alpha=0.8)
    plt.title('Portfolio Daily Returns Distribution', fontsize=12, color='white')
    plt.xlabel('Daily Return', color='white')
    plt.ylabel('Frequency', color='white')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca().set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')
    plt.tick_params(colors='white')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=plt.gcf().get_facecolor())
    plt.close()
    buf.seek(0)
    histogram_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Prepare details for each stock
    details = []
    for i, col in enumerate(data.columns):
        mean_pct = float(mean_returns.iloc[i] * 100) if i < len(mean_returns) else 0.0
        vol_pct = float(np.sqrt(cov_matrix.iloc[i, i]) * 100) if i < cov_matrix.shape[0] else 0.0
        details.append({
            "symbol": col,
            "weight": round(float(weights[i]), 4),
            "mean": round(mean_pct, 2),
            "vol": round(vol_pct, 2)
        })

    return {
        "portfolio_return": round(portfolio_return * 100, 2),
        "portfolio_volatility": round(portfolio_volatility * 100, 2),
        "sharpe": round(sharpe_ratio, 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "details": details,
        "histogram": histogram_base64,
    }

@login_required
def home(request):
    user = request.user
    portfolios = Portfolio.objects.filter(user=user)

    # ----------------------------
    # Portfolio selection (GET)
    # ----------------------------
    portfolio_id = request.GET.get("portfolio_id")
    selected_portfolio = None

    if portfolio_id:
        selected_portfolio = Portfolio.objects.filter(
            id=portfolio_id,
            user=user
        ).first()

    # ----------------------------
    # Default formset
    # ----------------------------
    if selected_portfolio:
        initial_data = [
            {"symbol": h.symbol, "weight": h.weight}
            for h in selected_portfolio.holdings.all()
        ]
        formset = StockFormSet(initial=initial_data)
    else:
        formset = StockFormSet()

    # ----------------------------
    # GET → render page
    # ----------------------------
    if request.method == "GET":
        return render(request, "stockapp/index.html", {
            "formset": formset,
            "portfolios": portfolios,
            "selected_portfolio": selected_portfolio,
        })

    # ----------------------------
    # POST actions
    # ----------------------------
    action = request.POST.get("action")

    # 🟩 Upload CSV / Excel
    if action == "upload_file":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return render(request, "stockapp/index.html", {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "error": "No file selected."
            })

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file type")

            df.columns = df.columns.str.lower().str.strip()
            if not {"symbol", "weight"}.issubset(df.columns):
                raise ValueError("CSV must contain symbol and weight")

            initial_data = [
                {"symbol": str(r.symbol).upper(), "weight": float(r.weight)}
                for r in df.itertuples()
            ]

            formset = StockFormSet(initial=initial_data)

            return render(request, "stockapp/index.html", {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "message": f"Loaded {len(initial_data)} stocks from file."
            })

        except Exception as e:
            return render(request, "stockapp/index.html", {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "error": str(e)
            })

    # 🟦 Analyze portfolio
    elif action == "analyze":
        formset = StockFormSet(request.POST)

        if not formset.is_valid():
            return render(request, "stockapp/index.html", {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "error": "Invalid input."
            })

        stocks, weights = [], []
        for f in formset:
            if f.cleaned_data:
                stocks.append(f.cleaned_data["symbol"].upper())
                weights.append(f.cleaned_data["weight"])

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        try:
            results = analyze_portfolio(stocks, weights)

            context = {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "symbols": ",".join(stocks),
                "weights": ",".join(map(str, weights)),
            }
            context.update(results)

            return render(request, "stockapp/index.html", context)

        except Exception as e:
            return render(request, "stockapp/index.html", {
                "formset": formset,
                "portfolios": portfolios,
                "selected_portfolio": selected_portfolio,
                "error": str(e)
            })

    elif action == "save_portfolio":
        portfolio_name = request.POST.get("portfolio_name")
        symbols = request.POST.get("symbols")
        weights = request.POST.get("weights")

        if not portfolio_name or not symbols or not weights:
            return render(request, "stockapp/index.html", {
            "formset": formset,
            "portfolios": portfolios,
            "error": "Missing portfolio data."
        })

        symbols = symbols.split(",")
        weights = list(map(float, weights.split(",")))

        # ✅ Create NEW portfolio (never overwrite silently)
        portfolio = Portfolio.objects.create(
            user=request.user,
            name=portfolio_name
        )

        for s, w in zip(symbols, weights):
            Holding.objects.create(
                portfolio=portfolio,
                symbol=s,
                weight=w
            )

        return redirect("home")

    # 🟨 Fallback
    return render(request, "stockapp/index.html", {
        "formset": formset,
        "portfolios": portfolios,
        "selected_portfolio": selected_portfolio,
        "error": "Unknown action."
    })




@login_required
def efficient_frontier(request):
    """
    Efficient frontier view. Accepts POST with 'symbols' & 'weights' OR file upload.
    """
    context = {}
    user = request.user
    context["portfolios"] = Portfolio.objects.filter(user=user)

    # Case 1: From 'Optimizer' with symbols & weights
    if request.method == "POST" and request.POST.get("symbols") and request.POST.get("weights") and not request.FILES.get("file"):
        symbols = request.POST.get("symbols")
        weights = request.POST.get("weights")
        try:
            risk_free_rate = float(request.POST.get("risk_free_rate", 0.05))
        except Exception:
            risk_free_rate = 0.05

        try:
            stocks = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            weights = [float(w.strip()) for w in weights.split(",") if w.strip()]

            raw = yf.download(stocks, period="5y", progress=False)
            data = _ensure_df_close(raw)
            if data.empty:
                context["error"] = "No valid stock data returned."
                return render(request, "stockapp/efficient_frontier.html", context)

            new_returns = data.pct_change().dropna()
            opt_returns = new_returns  # no need to drop 'Daily_returns' here; defensive below

            mean_returns = opt_returns.mean() * 252
            cov_matrix = opt_returns.cov() * 252
            num_assets = len(opt_returns.columns)
            num_portfolios = 50000

            # Pre-alloc
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                ex_weights = np.random.random(num_assets)
                ex_weights /= np.sum(ex_weights)
                port_return = np.dot(ex_weights, mean_returns)
                port_std = np.sqrt(np.dot(ex_weights.T, np.dot(cov_matrix, ex_weights)))
                # guard against zero volatility
                sharpe = (port_return - risk_free_rate) / port_std if port_std != 0 else 0
                results[0, i] = port_return
                results[1, i] = port_std
                results[2, i] = sharpe
                weights_record.append(ex_weights)

            weights_record = np.array(weights_record)
            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0, max_sharpe_idx]
            max_sharpe_std = results[1, max_sharpe_idx]
            max_sharpe_ratio = results[2, max_sharpe_idx]
            best_weights = weights_record[max_sharpe_idx]

            best_portfolio = pd.DataFrame(best_weights, index=opt_returns.columns, columns=['Weight']).sort_values(by='Weight', ascending=False)

            original_weights_dict = dict(zip(stocks, weights))
            weight_changes = []
            for symbol in opt_returns.columns:
                old_w = original_weights_dict.get(symbol, 0)
                new_w = float(best_portfolio.loc[symbol, "Weight"])
                change = new_w - old_w
                weight_changes.append({
                    "symbol": symbol,
                    "old_weight": round(old_w, 4),
                    "new_weight": round(new_w, 4),
                    "change": round(change, 4)
                })
            weight_changes = sorted(weight_changes, key=lambda x: x["new_weight"], reverse=True)

            # Plot efficient frontier
            plt.figure(figsize=(10, 6))
            plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Risk (Volatility)')
            plt.ylabel('Expected Return')
            plt.title('Efficient Frontier')
            plt.scatter(max_sharpe_std, max_sharpe_return, c='red', marker='*', s=120)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frontier_plot = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

            context.update({
                "frontier_plot": frontier_plot,
                "best_portfolio": best_portfolio.to_dict(orient='index'),
                "weight_changes": weight_changes,
                "max_sharpe_return": round(float(max_sharpe_return * 100), 2),
                "max_sharpe_std": round(float(max_sharpe_std * 100), 2),
                "max_sharpe_ratio": round(float(max_sharpe_ratio), 3),
                "symbols": symbols,
                "weights": weights,
                "risk_free_rate": risk_free_rate,
            })
        except Exception as e:
            context["error"] = f"Error generating efficient frontier: {e}"
            print("Efficient frontier error:", e)

        return render(request, "stockapp/efficient_frontier.html", context)

    # Case 2: GET -> show page
    if request.method == "GET":
        return render(request, "stockapp/efficient_frontier.html", context)

    # Case 3: File upload for frontier
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                context["error"] = "Unsupported file type. Please upload CSV or Excel."
                return render(request, "stockapp/efficient_frontier.html", context)

            df.columns = df.columns.str.lower().str.strip()
            if not {"symbol", "weight"}.issubset(df.columns):
                context["error"] = "CSV must contain 'symbol' and 'weight' columns."
                return render(request, "stockapp/efficient_frontier.html", context)

            context["symbols"] = ",".join(df["symbol"].astype(str))
            context["weights"] = ",".join(df["weight"].astype(str))
            context["message"] = f"Loaded {len(df)} symbols from file. Click Analyze to optimize."

        except Exception as e:
            context["error"] = f"Error reading file: {e}"

        return render(request, "stockapp/efficient_frontier.html", context)

    return render(request, "stockapp/efficient_frontier.html", context)


def register(request):
    """
    Simple registration view using Django's built-in UserCreationForm.
    """
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # auto login after register
            return redirect("home")
    else:
        form = UserCreationForm()
    return render(request, "stockapp/register.html", {"form": form})

def dcf_view(request):

    return render(request, "stockapp/dcf.html")

from mydcf import DCF_Valuation
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import WACC as wacc


def dcf_view(request):
    result = None
    error = None
    ticker = ""

    show_inputs = False
    input_params = {}

    # 👇 pull originals from session (if they exist)
    original_params = request.session.get("original_params", {})

    if request.method == "POST":
        try:
            ticker = request.POST.get("ticker")
            model = DCF_Valuation(ticker)

            # ====================================================
            # FIRST RUN → MAIN DCF
            # ====================================================
            if "wacc" not in request.POST:
                wacc_model = wacc.WACCModel(ticker=ticker)
                wacc_value = wacc_model.wacc()

                ev, proj, params = model.project_fcff(wacc_value)
                eq_val, per_share = model.equity_value(ev)

                result = {
                    "enterprise_value": ev,
                    "equity_value": eq_val,
                    "per_share_value": per_share,
                    "dcf_table": proj
                }

                # ORIGINAL (model-derived, frozen)
                original_params = {
                    "wacc": round(params["wacc"] * 100, 2),
                    "growth_rate": round(params["growth_rate"] * 100, 2),
                    "reinvestment_rate": round(params["reinvestment_rate"] * 100, 2),
                    "terminal_growth": round(params["terminal_growth"] * 100, 2),
                    "years": params["years"]
                }

                # 🔒 store originals in session
                request.session["original_params"] = original_params

                # Editable inputs start equal to originals
                input_params = original_params.copy()

                show_inputs = True

            # ====================================================
            # SUBSEQUENT RUNS → RAW DCF
            # ====================================================
            else:
                input_params = {
                    "wacc": float(request.POST.get("wacc")),
                    "growth_rate": float(request.POST.get("growth_rate")),
                    "reinvestment_rate": float(request.POST.get("reinvestment_rate")),
                    "terminal_growth": float(request.POST.get("terminal_growth")),
                    "years": int(request.POST.get("years"))
                }

                result = model.run_raw_dcf(
                    wacc=input_params["wacc"] / 100,
                    growth_rate=input_params["growth_rate"] / 100,
                    reinvestment_rate=input_params["reinvestment_rate"] / 100,
                    terminal_growth=input_params["terminal_growth"] / 100,
                    years=input_params["years"]
                )

                show_inputs = True

        except Exception as e:
            error = str(e)

    return render(
        request,
        "stockapp/dcf.html",
        {
            "ticker": ticker,
            "result": result,
            "error": error,
            "show_inputs": show_inputs,
            "input_params": input_params,
            "original_params": original_params
        }
    )

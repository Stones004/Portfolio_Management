import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from .forms import StockFormSet
import yfinance as yf

def analyze_portfolio(stocks, weights):
    try:
        # Fetch stock data for the past 5 years
        data = yf.download(stocks, period="5y")['Close']
        if data.empty:
            raise ValueError("No data returned for given tickers.")
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}")

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Normalize weights to ensure they sum to 1
    weights = np.array(weights)
    weights /= np.sum(weights)  # Ensuring weights sum to 1

    # Calculate mean returns and covariance matrix (annualized)
    mean_returns = daily_returns.mean() * 252  # annualize mean returns
    cov_matrix = daily_returns.cov() * 252  # annualize covariance matrix

    # Calculate portfolio return and volatility
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0

    # Calculate portfolio daily returns
    portfolio_daily_returns = np.dot(daily_returns, weights)
    max_daily_loss = np.percentile(portfolio_daily_returns, 5) * 100  # 5th percentile for max loss

    # Generate histogram for daily returns distribution
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

    # Prepare details for each stock in the portfolio
    details = []
    for i, s in enumerate(stocks):
        details.append({
            "symbol": s,
            "weight": round(weights[i], 4),
            "mean": round(mean_returns.iloc[i] * 100, 2),
            "vol": round(np.sqrt(cov_matrix.iloc[i, i]) * 100, 2)
        })

    return {
        "portfolio_return": round(portfolio_return * 100, 2),
        "portfolio_volatility": round(portfolio_volatility * 100, 2),
        "sharpe": round(sharpe_ratio, 2),
        "max_daily_loss": round(max_daily_loss, 2),
        "details": details,
        "histogram": histogram_base64
    }

def home(request):
    # Handle GET request (show empty formset)
    if request.method == "GET":
        formset = StockFormSet()
        return render(request, "stockapp/index.html", {"formset": formset})

    # Handle POST request for file upload or portfolio analysis
    action = request.POST.get("action")
    print(action)

    if action == "upload_file":
        formset = StockFormSet()  # Blank formset for now (will be replaced if file is loaded)
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return render(request, "stockapp/index.html", {"formset": formset, "error": "No file selected."})

        try:
            # Read file content
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                return render(request, "stockapp/index.html", {"formset": formset, "error": "Unsupported file type. Use .csv/.xlsx."})

            # Normalize column names and ensure the required columns are present
            df.columns = df.columns.str.lower().str.strip()
            if not {"symbol", "weight"}.issubset(df.columns):
                return render(request, "stockapp/index.html", {"formset": formset, "error": "File needs 'symbol' and 'weight' columns."})

            # Process rows and append valid ones
            initial_data = []
            for _, row in df.iterrows():
                try:
                    initial_data.append({
                        "symbol": str(row["symbol"]).strip().upper(),
                        "weight": float(row["weight"])
                    })
                except ValueError:
                    # Skip rows with invalid data
                    continue

            if not initial_data:
                return render(request, "stockapp/index.html", {"formset": formset, "error": "No valid rows in file."})

            formset = StockFormSet(initial=initial_data)
            return render(request, "stockapp/index.html", {"formset": formset, "message": f"Loaded {len(initial_data)} stocks from file."})

        except Exception as e:
            return render(request, "stockapp/index.html", {"formset": formset, "error": f"Error reading file: {e}"})

    elif action == "analyze":
        # Process form submission for portfolio analysis
        formset = StockFormSet(request.POST)
        if not formset.is_valid():
            errors = []
            for i, f in enumerate(formset):
                if f.errors:
                    errors.append(f"Row {i + 1}: {f.errors.as_text()}")
            # Debug output for errors
            print(errors)
            return render(request, "stockapp/index.html", {"formset": formset, "error": "Invalid input. Please fix highlighted fields.", "details_errors": errors})

        # Extract stock symbols and weights from the formset
        stocks = []
        weights = []
        for form in formset:
            symbol = form.cleaned_data.get("symbol")
            weight = form.cleaned_data.get("weight")
            if symbol and weight is not None:
                print(f"Extracted symbol: {symbol}, weight: {weight}")  # Debug output
                stocks.append(symbol.upper())
                weights.append(weight)

        if not stocks:
            return render(request, "stockapp/index.html", {"formset": formset, "error": "Please add at least one stock."})

        total_weight = sum(weights)
        if total_weight <= 0:
            return render(request, "stockapp/index.html", {"formset": formset, "error": "Total weight must be > 0."})

        # Normalize weights
        weights = [w / total_weight for w in weights]

        # Analyze portfolio and render results
        try:
            results = analyze_portfolio(stocks, weights)
            context = {
                "formset": formset,
                "symbols": ",".join(stocks),
                "weights": ",".join(str(w) for w in weights),
            }
            context.update(results)
            return render(request, "stockapp/index.html", context)
        except Exception as e:
            return render(request, "stockapp/index.html", {"formset": formset, "error": f"Error analyzing portfolio: {e}"})

    else:
        # If action is unknown
        formset = StockFormSet(request.POST or None)
        return render(request, "stockapp/index.html", {"formset": formset, "error": "Unknown action."})


def efficient_frontier(request):
    context = {}

    # 🟩 Case 1: Coming from "Optimizer" button (POST with symbols & weights)
    if request.method == "POST" and request.POST.get("symbols") and request.POST.get("weights"):
        symbols = request.POST.get("symbols")
        weights = request.POST.get("weights")
        risk_free_rate = float(request.POST.get("risk_free_rate", 0.05))

        try:
            stocks = [s.strip().upper() for s in symbols.split(",")]
            weights = [float(w.strip()) for w in weights.split(",")]

            # ✅ Fetch stock data
            data = yf.download(stocks, period="5y")['Close']
            if data.empty:
                context["error"] = "No valid stock data returned."
                return render(request, "stockapp/efficient_frontier.html", context)

            new_returns = data.pct_change().dropna()
            opt_returns = new_returns.drop(columns=['Daily_returns'], errors='ignore')

            mean_returns = opt_returns.mean() * 252
            cov_matrix = opt_returns.cov() * 252
            num_assets = len(opt_returns.columns)
            num_portfolios = 50000

            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                ex_weights = np.random.random(num_assets)
                ex_weights /= np.sum(ex_weights)
                port_return = np.dot(ex_weights, mean_returns)
                port_std = np.sqrt(np.dot(ex_weights.T, np.dot(cov_matrix, ex_weights)))
                sharpe = (port_return - risk_free_rate) / port_std

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

            # 🟩 Create optimized weight DataFrame
            best_portfolio = pd.DataFrame(best_weights, index=opt_returns.columns, columns=['Weight']).sort_values(by='Weight', ascending=False)

            # 🟩 Compare old vs new weights
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

            # 🟩 Plot Efficient Frontier
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

            # 🟩 Update context
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
            print("Error:", e)

        return render(request, "stockapp/efficient_frontier.html", context)

    # 🟨 Case 2: User visits page directly (GET request)
    if request.method == "GET":
        return render(request, "stockapp/efficient_frontier.html", context)

    # 🟦 Case 3: Manual CSV Upload
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


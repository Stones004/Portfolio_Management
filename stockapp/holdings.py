"""Parsing holdings (symbol, weight) from uploads and form fields."""
import pandas as pd

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB is plenty for a holdings list


def read_holdings_file(uploaded_file):
    """Return (symbols, weights) from a CSV/Excel file with 'symbol' and 'weight' columns."""
    if uploaded_file.size and uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large (max 2 MB).")
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload a .csv, .xls or .xlsx file.")

    df.columns = df.columns.astype(str).str.lower().str.strip()
    if not {"symbol", "weight"}.issubset(df.columns):
        raise ValueError("File must contain 'symbol' and 'weight' columns.")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["symbol", "weight"])
    df = df[(df["symbol"] != "") & (df["weight"] > 0)]
    if df.empty:
        raise ValueError("No valid rows found in the file.")
    # combine duplicate symbols
    df = df.groupby("symbol", sort=False, as_index=False)["weight"].sum()
    return df["symbol"].tolist(), df["weight"].tolist()


def parse_csv_fields(symbols_text, weights_text):
    """Parse the hidden 'AAPL,MSFT' / '0.5,0.5' fields passed between pages."""
    symbols = [s.strip().upper() for s in (symbols_text or "").split(",") if s.strip()]
    try:
        weights = [float(w) for w in (weights_text or "").split(",") if w.strip()]
    except ValueError:
        raise ValueError("Weights must be numbers.")
    if not symbols or len(symbols) != len(weights):
        raise ValueError("Symbols and weights don't match.")
    return symbols, weights


def normalise(weights):
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must add up to more than zero.")
    return [w / total for w in weights]

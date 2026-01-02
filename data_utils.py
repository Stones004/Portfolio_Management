import numpy as np
import pandas as pd

def ensure_df_close(data):
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        close = data['Close']
    else:
        try:
            close = data['Close']
        except Exception:
            close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=str(close.name) if close.name else '0')

    close = close.select_dtypes(include=[np.number])
    return close.dropna(how='all')

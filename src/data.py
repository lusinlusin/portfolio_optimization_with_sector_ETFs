from pathlib import Path

import pandas as pd
import yfinance as yf

ASSETS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# Start one month early (Dec 2018) so the first monthly return is Jan 2019:
# pct_change drops the first month, and we want returns to span Jan 2019–Dec 2025
# (a full first calendar year, so the sample is exactly 84 monthly returns).
START = '2018-12-01'
END   = '2025-12-31'


def download_data(ticker: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            prices_daily = df["Close"].copy()
        else:
            prices_daily = df["Adj Close"].copy()
    else:
        prices_daily = df.copy()

    prices_daily = prices_daily.dropna(how="all").sort_index().ffill()
    prices_daily.index = pd.to_datetime(prices_daily.index)
    prices_daily.index.name = "date"

    # The project works at monthly frequency, so we keep the last trading day of each month.
    prices_monthly = prices_daily.resample("ME").last()

    # yfinance returns columns alphabetically; restore the requested ASSETS order so
    # downstream estimators / weight vectors stay aligned.
    if isinstance(ticker, (list, tuple)):
        prices_monthly = prices_monthly.reindex(columns=list(ticker))

    returns_monthly = prices_monthly.pct_change()

    return returns_monthly

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    returns = download_data(ASSETS, START, END)
    returns = returns.dropna(how="all")  # drop the leading all-NaN month from pct_change
    print(returns.tail())
    returns.to_csv(data_dir / "monthly_returns.csv")

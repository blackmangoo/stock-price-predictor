# =============================================
# src/data_loader.py — Fetch & Preprocess Stock Data
# =============================================
# This module handles everything related to GETTING the data:
#   1. Downloading stock data from Yahoo Finance API
#   2. Saving it locally as CSV (so we don't re-download every time)
#   3. Basic preprocessing (cleaning, handling missing values)
#
# KEY CONCEPT: yfinance
# Yahoo Finance provides FREE historical stock data.
# The 'yfinance' library is a Python wrapper around their API.
# It returns data as a pandas DataFrame — the most common
# data structure in data science.
# =============================================

import yfinance as yf      # Yahoo Finance API wrapper
import pandas as pd        # Data manipulation library (think: Excel on steroids)
import os                  # For file path operations (checking if file exists, etc.)


def fetch_stock_data(ticker: str, start: str, end: str, save_path: str = None) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    PARAMETERS:
    -----------
    ticker : str
        The stock symbol (e.g., 'AAPL' for Apple, 'TSLA' for Tesla).
        These are unique identifiers for companies on the stock exchange.
        
    start : str
        Start date in 'YYYY-MM-DD' format (e.g., '2020-01-01').
        
    end : str
        End date in 'YYYY-MM-DD' format (e.g., '2024-12-31').
        
    save_path : str, optional
        If provided, saves the data as a CSV file at this path.
        This avoids re-downloading data every time you run the code.
    
    RETURNS:
    --------
    pd.DataFrame
        A DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        
        WHAT THESE COLUMNS MEAN:
        - Open:      Price when market opened that day
        - High:      Highest price reached during that day
        - Low:       Lowest price reached during that day
        - Close:     Price when market closed that day (THIS IS WHAT WE PREDICT)
        - Volume:    Number of shares traded that day
        - Adj Close: Close price adjusted for dividends/splits
    
    EXAMPLE:
    --------
    >>> df = fetch_stock_data('AAPL', '2022-01-01', '2024-12-31')
    >>> print(df.head())
    """
    
    print(f"📊 Downloading {ticker} stock data from {start} to {end}...")
    
    # ---- Download data from Yahoo Finance ----
    # yf.download() sends a request to Yahoo Finance's servers
    # and returns the data as a pandas DataFrame.
    # 'auto_adjust=False' keeps both Close and Adj Close columns.
    stock_data = yf.download(
        tickers=ticker,     # Which stock to download
        start=start,        # From when
        end=end,            # Until when
        auto_adjust=False,  # Keep original Close price column
        progress=True       # Show download progress bar
    )
    
    # ---- Validate: Did we get any data? ----
    # Sometimes the ticker is wrong or the date range has no data
    if stock_data.empty:
        raise ValueError(
            f"❌ No data returned for ticker '{ticker}'. "
            f"Check if the ticker symbol is correct."
        )
    
    # ---- Handle MultiIndex columns ----
    # When downloading a single ticker, yfinance sometimes returns
    # MultiIndex columns like ('Close', 'AAPL'). We flatten them.
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # ---- Print basic info about the downloaded data ----
    # This is good practice to verify your data looks correct
    print(f"✅ Downloaded {len(stock_data)} trading days of data")
    print(f"📅 Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    print(f"📋 Columns: {list(stock_data.columns)}")
    
    # ---- Save to CSV if path is provided ----
    # CSV (Comma-Separated Values) is a simple text format for tabular data.
    # Saving locally means we can load the data faster next time
    # without hitting the API again.
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stock_data.to_csv(save_path)
        print(f"💾 Data saved to {save_path}")
    
    return stock_data


def load_stock_data(filepath: str) -> pd.DataFrame:
    """
    Load previously saved stock data from a CSV file.
    
    WHY LOAD FROM CSV?
    ------------------
    - Faster than downloading from API every time
    - Works offline (no internet needed)
    - Reproducible: same data every time you run your code
    
    PARAMETERS:
    -----------
    filepath : str
        Path to the CSV file (e.g., 'data/AAPL_stock.csv')
    
    RETURNS:
    --------
    pd.DataFrame with Date as the index
    """
    
    # ---- Check if file exists ----
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ File not found: {filepath}. "
            f"Download data first using fetch_stock_data()."
        )
    
    # ---- Read CSV with Date as index ----
    # 'index_col=0' means the first column (Date) becomes the index
    # 'parse_dates=True' converts date strings to datetime objects
    # so we can do date arithmetic (e.g., filter by date range)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"📂 Loaded {len(df)} rows from {filepath}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the stock data.
    
    WHAT IS PREPROCESSING?
    ----------------------
    Raw data is rarely perfect. It might have:
    - Missing values (NaN) — market was closed, data gap, etc.
    - Duplicate rows
    - Wrong data types
    
    Preprocessing fixes these issues BEFORE we feed data to our model.
    Think of it as "cleaning the kitchen before cooking."
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Raw stock data from fetch_stock_data() or load_stock_data()
    
    RETURNS:
    --------
    pd.DataFrame — cleaned and ready for feature engineering
    """
    
    # ---- Make a copy to avoid modifying the original ----
    # In Python, DataFrames are passed by REFERENCE, not by value.
    # Without .copy(), changes here would change the original df too!
    df = df.copy()
    
    # ---- Step 1: Check for missing values ----
    # .isnull().sum() counts NaN values in each column
    missing = df.isnull().sum()
    if missing.any():
        print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
        
        # ---- Fill missing values using forward fill ----
        # 'ffill' = Forward Fill: use the previous day's value
        # This makes sense for stock data because if a value is missing,
        # the price was likely the same as the previous close.
        # 
        # Example: If Wednesday's Close is NaN, use Tuesday's Close.
        # This is better than using the MEAN (which would mix future data).
        df = df.ffill()
        
        # ---- Backward fill any remaining NaN at the start ----
        # If the FIRST row has NaN, forward fill can't help (no previous value).
        # Backward fill uses the NEXT available value instead.
        df = df.bfill()
        
        print(f"✅ Missing values filled using forward/backward fill")
    else:
        print(f"✅ No missing values found")
    
    # ---- Step 2: Remove duplicate rows ----
    # Sometimes API returns duplicate dates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Removing {duplicates} duplicate rows")
        df = df[~df.index.duplicated(keep='first')]  # Keep first occurrence
    
    # ---- Step 3: Sort by date (ascending) ----
    # Ensure data is in chronological order (oldest first).
    # This is CRITICAL for time series data!
    df = df.sort_index()
    
    # ---- Step 4: Verify data types ----
    # All price/volume columns should be numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✅ Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
    print(f"📋 Data shape: {df.shape}")
    
    return df


# =============================================
# MAIN — Run this file directly to test it
# =============================================
# The 'if __name__ == "__main__"' block only runs when you execute
# this file directly (python src/data_loader.py), NOT when you
# import it from another file (from src.data_loader import ...).
# This is a common Python pattern for testing modules.
# =============================================
if __name__ == "__main__":
    # Test with Apple stock
    df = fetch_stock_data(
        ticker="AAPL",
        start="2022-01-01",
        end="2024-12-31",
        save_path="data/AAPL_stock.csv"
    )
    
    # Preprocess
    df = preprocess_data(df)
    
    # Show first 5 rows
    print("\n📊 First 5 rows:")
    print(df.head())
    
    # Show basic statistics
    print("\n📈 Basic Statistics:")
    print(df.describe())

# =============================================
# src/feature_engineering.py — Create Features for ML Model
# =============================================
# WHAT IS FEATURE ENGINEERING?
# ----------------------------
# Raw stock data only has: Open, High, Low, Close, Volume.
# That's just 5 columns — not enough for a good prediction.
#
# Feature engineering is the art of CREATING NEW COLUMNS
# from existing data that help the model learn patterns.
#
# Example: Instead of just giving the model today's price,
# we also tell it:
#   - What was the price 1, 2, 3 days ago? (lag features)
#   - What's the 7-day average trend? (moving averages)
#   - Is the stock overbought or oversold? (RSI)
#   - How much did volume change? (volume change)
#
# WHY THIS MATTERS:
# The model itself can't look at previous days' data.
# It sees each row independently. So WE create features
# that encode the "history" into each row.
# =============================================

import pandas as pd
import numpy as np      # NumPy = Numerical Python, for math operations
from ta.momentum import RSIIndicator  # Technical Analysis library for RSI


def add_lag_features(df: pd.DataFrame, column: str = 'Close', lags: list = [1, 2, 3, 5, 7]) -> pd.DataFrame:
    """
    Create LAG FEATURES — previous days' values as new columns.
    
    WHAT ARE LAG FEATURES?
    ----------------------
    A lag feature is simply "what was the value N days ago?"
    
    Example for Close price with lag=1:
    | Date       | Close | Close_lag_1 |
    |------------|-------|-------------|
    | 2024-01-01 | 100   | NaN         |  ← No previous day
    | 2024-01-02 | 102   | 100         |  ← Yesterday's close
    | 2024-01-03 | 99    | 102         |  ← Yesterday's close
    
    WHY?
    The model can't "look back in time" on its own.
    By adding lag features, we TELL the model what happened
    in the past, so it can learn patterns like:
    "If the price dropped 3 days in a row, it might bounce back."
    
    PARAMETERS:
    -----------
    df : pd.DataFrame — stock data with a 'Close' column
    column : str — which column to create lags for (default: 'Close')
    lags : list of int — how many days back to look (default: [1,2,3,5,7])
    
    RETURNS:
    --------
    pd.DataFrame with new lag columns added
    """
    df = df.copy()
    
    for lag in lags:
        # .shift(n) moves the entire column DOWN by n rows
        # This effectively gives us the value from n days ago
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
    print(f"✅ Added {len(lags)} lag features for '{column}': {lags}")
    return df


def add_moving_averages(df: pd.DataFrame, column: str = 'Close', windows: list = [7, 21, 50]) -> pd.DataFrame:
    """
    Create MOVING AVERAGE features — smoothed trend indicators.
    
    WHAT IS A MOVING AVERAGE?
    -------------------------
    A moving average smooths out daily price fluctuations
    to show the overall TREND.
    
    SMA_7  = Average of the last  7 days → Short-term trend
    SMA_21 = Average of the last 21 days → Medium-term trend  
    SMA_50 = Average of the last 50 days → Long-term trend
    
    HOW IT WORKS (SMA = Simple Moving Average):
    For SMA_7 on day 8:
        SMA_7 = (day1 + day2 + day3 + day4 + day5 + day6 + day7) / 7
    On day 9:
        SMA_7 = (day2 + day3 + day4 + day5 + day6 + day7 + day8) / 7
    The "window" slides forward one day at a time.
    
    WHY MOVING AVERAGES HELP:
    - If price > SMA_50: stock is in an UPTREND
    - If price < SMA_50: stock is in a DOWNTREND
    - When SMA_7 crosses above SMA_21: bullish signal ("golden cross")
    - When SMA_7 crosses below SMA_21: bearish signal ("death cross")
    
    PARAMETERS:
    -----------
    df : pd.DataFrame — stock data
    column : str — column to calculate MA on (default: 'Close')
    windows : list of int — window sizes (default: [7, 21, 50])
    """
    df = df.copy()
    
    for window in windows:
        # .rolling(n) creates a sliding window of n days
        # .mean() calculates the average within that window
        df[f'SMA_{window}'] = df[column].rolling(window=window).mean()
        
    print(f"✅ Added {len(windows)} moving averages: SMA_{', SMA_'.join(map(str, windows))}")
    return df


def add_rsi(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index) — momentum indicator.
    
    WHAT IS RSI?
    ------------
    RSI measures the SPEED and MAGNITUDE of price changes.
    It ranges from 0 to 100:
    
    - RSI > 70 → OVERBOUGHT (price went up too fast, might drop)
    - RSI < 30 → OVERSOLD  (price went down too fast, might rise)
    - RSI ≈ 50 → NEUTRAL
    
    HOW RSI IS CALCULATED (simplified):
    1. For each day, calculate the price change (today - yesterday)
    2. Separate into gains (positive changes) and losses (negative changes)
    3. Average gain over last 14 days / Average loss over last 14 days = RS
    4. RSI = 100 - (100 / (1 + RS))
    
    WHY 14 DAYS?
    14 is the standard window used by most traders.
    Shorter windows (e.g., 7) are more sensitive to price changes.
    Longer windows (e.g., 21) are smoother but slower to react.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
    column : str — price column (default: 'Close')
    window : int — RSI period (default: 14)
    """
    df = df.copy()
    
    # Using the 'ta' library's RSI calculation
    # This handles all the math internally
    rsi_indicator = RSIIndicator(close=df[column], window=window)
    df['RSI'] = rsi_indicator.rsi()
    
    print(f"✅ Added RSI with window={window}")
    return df


def add_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PRICE CHANGE features — how much did the price move?
    
    FEATURES CREATED:
    -----------------
    1. daily_return: Percentage change from yesterday
       Formula: ((today - yesterday) / yesterday) * 100
       Example: 100 → 103 = +3% return
       
    2. daily_range: How volatile was the day? (High - Low)
       A large range means high volatility (big price swings)
       
    3. open_close_diff: Difference between Open and Close
       Positive = price went UP during the day (bullish)
       Negative = price went DOWN during the day (bearish)
    
    WHY THESE MATTER:
    - Returns are SCALE-FREE (works across different stock prices)
    - Volatility often precedes big moves
    - Open-close patterns reveal market sentiment
    """
    df = df.copy()
    
    # ---- Percentage change from previous day ----
    # .pct_change() calculates (current - previous) / previous
    # Multiply by 100 to get percentage
    df['daily_return'] = df['Close'].pct_change() * 100
    
    # ---- Daily price range (High - Low) ----
    # This measures VOLATILITY: how much the price swung during the day
    df['daily_range'] = df['High'] - df['Low']
    
    # ---- Open to Close difference ----
    # Positive = bullish day (price went up)
    # Negative = bearish day (price went down)
    df['open_close_diff'] = df['Close'] - df['Open']
    
    print(f"✅ Added price change features: daily_return, daily_range, open_close_diff")
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VOLUME-BASED features — trading activity indicators.
    
    WHAT IS VOLUME?
    ---------------
    Volume = number of shares traded in a day.
    High volume = lots of buying/selling (strong conviction)
    Low volume  = few trades (weak conviction)
    
    FEATURES CREATED:
    -----------------
    1. volume_change: % change in volume from yesterday
    2. volume_sma_7: 7-day average volume (smoothed)
    3. volume_ratio: Today's volume / 7-day average
       > 1 means above-average trading activity (something's happening!)
    """
    df = df.copy()
    
    # ---- Volume percentage change ----
    df['volume_change'] = df['Volume'].pct_change() * 100
    
    # ---- 7-day average volume ----
    df['volume_sma_7'] = df['Volume'].rolling(window=7).mean()
    
    # ---- Volume ratio: today vs average ----
    # A ratio > 1 means higher-than-normal trading activity
    # This often signals important news or events affecting the stock
    df['volume_ratio'] = df['Volume'] / df['volume_sma_7']
    
    print(f"✅ Added volume features: volume_change, volume_sma_7, volume_ratio")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the TARGET variable — what we want to PREDICT.
    
    OUR GOAL: Predict TOMORROW'S closing price.
    
    HOW?
    We shift the Close column BACKWARD by 1 day.
    This means each row's target is the NEXT day's Close price.
    
    | Date       | Close | target (next_day_close) |
    |------------|-------|-------------------------|
    | 2024-01-01 | 100   | 102                     | ← Tomorrow's close
    | 2024-01-02 | 102   | 99                      | ← Tomorrow's close
    | 2024-01-03 | 99    | NaN                     | ← Unknown (the future!)
    
    The LAST ROW will have NaN as the target because we don't know
    tomorrow's price yet. We'll drop this row before training.
    
    IMPORTANT DISTINCTION:
    - Features (X) = what we KNOW (today's data + historical patterns)
    - Target (y) = what we PREDICT (tomorrow's close price)
    """
    df = df.copy()
    
    # shift(-1) moves data UP by 1 row
    # So each row now contains TOMORROW's close as the target
    df['target'] = df['Close'].shift(-1)
    
    print(f"✅ Created target variable: 'target' (next day's close price)")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MASTER FUNCTION: Run all feature engineering steps in order.
    
    This is a convenience function that chains all the individual
    feature engineering functions together. Call this one function
    instead of calling each one separately.
    
    PIPELINE:
    1. Add lag features (past prices)
    2. Add moving averages (trends)
    3. Add RSI (momentum)
    4. Add price changes (daily movements)
    5. Add volume features (trading activity)
    6. Create target variable (what to predict)
    7. Drop rows with NaN (from feature creation)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame — preprocessed stock data
    
    RETURNS:
    --------
    pd.DataFrame — fully featured data ready for modeling
    """
    print("\n🔧 === FEATURE ENGINEERING PIPELINE ===")
    print(f"Starting with {len(df)} rows, {len(df.columns)} columns\n")
    
    # ---- Step 1: Lag features ----
    df = add_lag_features(df, column='Close', lags=[1, 2, 3, 5, 7])
    
    # ---- Step 2: Moving averages ----
    df = add_moving_averages(df, column='Close', windows=[7, 21, 50])
    
    # ---- Step 3: RSI ----
    df = add_rsi(df, column='Close', window=14)
    
    # ---- Step 4: Price changes ----
    df = add_price_changes(df)
    
    # ---- Step 5: Volume features ----
    df = add_volume_features(df)
    
    # ---- Step 6: Target variable ----
    df = create_target(df)
    
    # ---- Step 7: Drop NaN rows ----
    # Feature creation introduces NaN values:
    # - Lag features: first N rows have NaN (no previous data)
    # - Moving averages: first N rows have NaN (not enough data for window)
    # - RSI: first 14 rows have NaN
    # - Target: last row has NaN (don't know tomorrow's price)
    #
    # We MUST drop these rows before training the model,
    # because ML models can't handle NaN values.
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    
    print(f"\n🗑️  Dropped {rows_dropped} rows with NaN values")
    print(f"✅ Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"📋 All features: {list(df.columns)}")
    
    return df


# =============================================
# MAIN — Test feature engineering
# =============================================
if __name__ == "__main__":
    from data_loader import fetch_stock_data, preprocess_data
    
    # Load data
    df = fetch_stock_data("AAPL", "2022-01-01", "2024-12-31")
    df = preprocess_data(df)
    
    # Build features
    df = build_features(df)
    
    print("\n📊 Sample of featured data:")
    print(df.head())
    print(f"\n📋 Total features: {len(df.columns)}")

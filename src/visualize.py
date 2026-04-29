# =============================================
# src/visualize.py — Plotting & Visualization Functions
# =============================================
# This module creates all the charts and plots.
# We use TWO libraries:
#   1. matplotlib + seaborn — for static plots (good for notebooks/papers)
#   2. plotly — for interactive plots (good for dashboards, hover to see values)
#
# WHY VISUALIZATION MATTERS:
# - Humans are visual — a chart tells more than 1000 numbers
# - Helps spot patterns, outliers, and errors in your data
# - Essential for presenting results to non-technical stakeholders
# - Makes your GitHub repo and LinkedIn post look professional
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     # The OG Python plotting library
import matplotlib.dates as mdates   # For formatting dates on x-axis
import seaborn as sns               # Built on matplotlib, prettier by default
import plotly.graph_objects as go    # Interactive plots
from plotly.subplots import make_subplots  # Multiple plots in one figure
import os


# ---- Set a consistent style for all plots ----
# This makes every matplotlib plot look clean and professional
# without setting style in every function.
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams['figure.figsize'] = (14, 7)     # Default figure size
plt.rcParams['figure.dpi'] = 100              # Resolution
plt.rcParams['font.size'] = 12                # Default font size


def plot_stock_history(df: pd.DataFrame, ticker: str = "Stock", save_path: str = None):
    """
    Plot the stock price history with volume bars.
    
    Creates a 2-panel chart:
    - Top panel: Close price over time (line chart)
    - Bottom panel: Trading volume (bar chart)
    
    This is the most basic stock chart — every financial website shows this.
    It helps you visually understand the stock's overall trend.
    """
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1,                          # 2 rows, 1 column
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]},  # Top chart is 3x taller
        sharex=True                    # Share the same x-axis (dates)
    )
    
    # ---- Top panel: Price ----
    ax1.plot(df.index, df['Close'], color='#2196F3', linewidth=1.5, label='Close Price')
    ax1.set_title(f'{ticker} Stock Price History', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ---- Bottom panel: Volume ----
    # Color volume bars green (up days) and red (down days)
    colors = ['#4CAF50' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
              else '#F44336' for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=1)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates nicely
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_stock_interactive(df: pd.DataFrame, ticker: str = "Stock"):
    """
    Create an INTERACTIVE stock chart with Plotly.
    
    WHY INTERACTIVE?
    - Hover over any point to see exact values
    - Zoom in/out on specific date ranges
    - Perfect for Streamlit dashboards
    
    This creates a CANDLESTICK chart — the standard chart used by traders.
    Each "candle" shows 4 values for one day:
    - Green candle: Close > Open (price went UP)
    - Red candle:   Close < Open (price went DOWN)
    - Top of body:  max(Open, Close)
    - Bottom of body: min(Open, Close)
    - Top wick:     High price
    - Bottom wick:  Low price
    """
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price', 'Volume')
    )
    
    # ---- Candlestick chart ----
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#4CAF50',  # Green for up days
            decreasing_line_color='#F44336'   # Red for down days
        ),
        row=1, col=1
    )
    
    # ---- Add Moving Averages if they exist ----
    ma_colors = {'SMA_7': '#FF9800', 'SMA_21': '#9C27B0', 'SMA_50': '#00BCD4'}
    for ma_col, color in ma_colors.items():
        if ma_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[ma_col],
                    name=ma_col, line=dict(color=color, width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # ---- Volume bars ----
    colors = ['#4CAF50' if row['Close'] >= row['Open'] else '#F44336' 
              for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # ---- Layout styling ----
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        xaxis_rangeslider_visible=False,    # Hide the range slider
        template='plotly_dark',              # Dark theme
        height=700,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name: str = "Model", 
                             dates=None, save_path: str = None):
    """
    Plot ACTUAL vs PREDICTED prices — the most important chart!
    
    This chart directly shows how well your model performs:
    - If the lines overlap perfectly → great model
    - If the lines diverge → model is making errors
    
    We create TWO views:
    1. Time series view: actual and predicted prices over time
    2. Scatter plot: each point = one prediction vs actual value
       - Points on the diagonal line = perfect predictions
       - Points far from the diagonal = bad predictions
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ---- Left: Time Series View ----
    x_axis = dates if dates is not None else range(len(y_test))
    
    ax1.plot(x_axis, y_test, color='#2196F3', linewidth=1.5, label='Actual', alpha=0.8)
    ax1.plot(x_axis, y_pred, color='#FF5722', linewidth=1.5, label='Predicted', 
             alpha=0.8, linestyle='--')
    ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date' if dates is not None else 'Sample Index')
    ax1.set_ylabel('Price ($)')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # ---- Right: Scatter Plot ----
    ax2.scatter(y_test, y_pred, alpha=0.5, color='#9C27B0', edgecolors='white', s=40)
    
    # Draw the "perfect prediction" diagonal line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_title(f'{model_name}: Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str = "Model",
                            top_n: int = 15, save_path: str = None):
    """
    Plot feature importance as a horizontal bar chart.
    
    Shows which features the model considers most important.
    Longer bars = more important features.
    
    This helps you understand:
    - What drives predictions (e.g., "yesterday's close is the strongest predictor")
    - Which features you could remove (unimportant ones)
    - Whether the model learned something sensible
    """
    
    if importance_df is None:
        print("⚠️  No feature importance data available")
        return
    
    # Take top N features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart (easier to read feature names)
    bars = ax.barh(
        range(len(top_features)),
        top_features['importance'].values,
        color=sns.color_palette("viridis", len(top_features)),
        edgecolor='white'
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Most important at top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance ({model_name})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_test, y_pred, model_name: str = "Model", save_path: str = None):
    """
    Plot RESIDUALS — the prediction errors.
    
    WHAT ARE RESIDUALS?
    -------------------
    Residual = Actual - Predicted
    
    If residual = +5:  model predicted $5 too LOW
    If residual = -5:  model predicted $5 too HIGH
    If residual = 0:   perfect prediction!
    
    WHY ANALYZE RESIDUALS?
    - A good model should have residuals randomly scattered around 0
    - If residuals show a PATTERN, the model is missing something
    - If residuals grow over time, the model degrades on newer data
    
    We create TWO views:
    1. Residuals over time: should be random, no trend
    2. Residual distribution: should be bell-shaped (normal distribution)
    """
    
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---- Left: Residuals vs Predicted ----
    ax1.scatter(y_pred, residuals, alpha=0.5, color='#673AB7', edgecolors='white', s=30)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Price ($)', fontsize=12)
    ax1.set_ylabel('Residual ($)', fontsize=12)
    ax1.set_title(f'{model_name}: Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ---- Right: Residual Distribution ----
    ax2.hist(residuals, bins=40, color='#673AB7', edgecolor='white', alpha=0.7, density=True)
    
    # Overlay a normal distribution curve for comparison
    from scipy import stats
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
             color='red', linewidth=2, label='Normal Distribution')
    
    ax2.set_xlabel('Residual ($)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'{model_name}: Residual Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Create a side-by-side comparison chart of all models.
    
    This is the "money shot" — the chart that tells you
    at a glance which model performed best on each metric.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparison_df.plot(kind='bar', ax=ax, rot=0, colormap='Set2', edgecolor='white')
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()

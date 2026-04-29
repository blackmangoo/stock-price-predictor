# =============================================
# app/streamlit_app.py — Interactive Stock Dashboard
# =============================================
# WHAT IS STREAMLIT?
# ------------------
# Streamlit turns Python scripts into web applications.
# No HTML, CSS, or JavaScript needed!
# You write Python → Streamlit renders it as a beautiful web app.
#
# HOW TO RUN:
# streamlit run app/streamlit_app.py
#
# This opens a web browser at http://localhost:8501
# =============================================

import streamlit as st                 # The UI framework
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# ---- Add the project root to Python's path ----
# This allows us to import from the 'src' folder
# regardless of where we run the streamlit command from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import fetch_stock_data, preprocess_data
from src.feature_engineering import build_features
from src.model import split_time_series, train_and_evaluate_all


# =============================================
# PAGE CONFIGURATION
# =============================================
# Must be the FIRST Streamlit command in the script!
# Sets the page title, icon, and layout.
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide",              # Use full screen width
    initial_sidebar_state="expanded"
)


# =============================================
# CUSTOM CSS STYLING
# =============================================
# Streamlit allows injecting custom CSS for advanced styling.
# This gives the app a premium, professional look.
st.markdown("""
<style>
    /* ---- Main background ---- */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
    }
    
    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 40, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* ---- Custom header ---- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    
    /* ---- Divider ---- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# HEADER
# =============================================
st.markdown('<p class="main-header">📈 Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered next-day stock price prediction using Machine Learning</p>', unsafe_allow_html=True)


# =============================================
# SIDEBAR — User Controls
# =============================================
# The sidebar is where users configure parameters.
# It keeps the main area clean and focused on results.
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # ---- Stock Ticker Selection ----
    # Popular stocks with their full names for clarity
    popular_stocks = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet (Google)',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'META': 'Meta (Facebook)',
        'NVDA': 'NVIDIA',
        'NFLX': 'Netflix'
    }
    
    ticker = st.selectbox(
        "🏢 Select Stock",
        options=list(popular_stocks.keys()),
        format_func=lambda x: f"{x} — {popular_stocks[x]}",
        index=0  # Default: AAPL
    )
    
    # ---- Or enter a custom ticker ----
    custom_ticker = st.text_input("✏️ Or enter custom ticker", placeholder="e.g., AMD, UBER")
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    st.markdown("---")
    
    # ---- Date Range Selection ----
    st.markdown("### 📅 Date Range")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*3),  # 3 years ago
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    st.markdown("---")
    
    # ---- Model Settings ----
    st.markdown("### 🤖 Model Settings")
    test_size = st.slider(
        "Test split (%)",
        min_value=10, max_value=40, value=20,
        help="Percentage of data used for testing (remaining used for training)"
    ) / 100
    
    st.markdown("---")
    
    # ---- Action Button ----
    run_prediction = st.button("🚀 Run Prediction", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.markdown(
        "**Built by** [Ammar Akbar](https://github.com/blackmangoo)\n\n"
        "*DevelopersHub AI/ML Internship*"
    )


# =============================================
# MAIN CONTENT — Runs when user clicks "Run Prediction"
# =============================================
if run_prediction:
    
    # ---- STEP 1: Fetch Data ----
    with st.spinner(f"📊 Downloading {ticker} data..."):
        try:
            df = fetch_stock_data(
                ticker=ticker,
                start=str(start_date),
                end=str(end_date)
            )
            df = preprocess_data(df)
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.stop()  # Stop execution if data fetch fails
    
    # ---- Display Key Metrics ----
    st.markdown("---")
    st.markdown("### 📊 Stock Overview")
    
    # Calculate summary stats
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    daily_change = latest_close - prev_close
    daily_change_pct = (daily_change / prev_close) * 100
    high_52w = df['Close'].tail(252).max()   # 252 trading days ≈ 1 year
    low_52w = df['Close'].tail(252).min()
    avg_volume = df['Volume'].mean()
    
    # Display metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Close", f"${latest_close:.2f}", f"{daily_change_pct:+.2f}%")
    with col2:
        st.metric("52-Week High", f"${high_52w:.2f}")
    with col3:
        st.metric("52-Week Low", f"${low_52w:.2f}")
    with col4:
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    # ---- STEP 2: Interactive Chart ----
    st.markdown("---")
    st.markdown("### 📈 Price Chart")
    
    # Create candlestick chart
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25],
        subplot_titles=(f'{ticker} Price', 'Volume')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#4CAF50', decreasing_line_color='#F44336'
        ), row=1, col=1
    )
    
    colors = ['#4CAF50' if row['Close'] >= row['Open'] else '#F44336' 
              for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- STEP 3: Feature Engineering & Model Training ----
    st.markdown("---")
    st.markdown("### 🤖 Model Training & Prediction")
    
    with st.spinner("🔧 Engineering features & training models..."):
        # Build features
        df_featured = build_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_time_series(df_featured, test_size=test_size)
        
        # Train and evaluate
        results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    st.success("✅ Models trained successfully!")
    
    # ---- Display Model Comparison ----
    st.markdown("#### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Linear Regression**")
        lr = results['lr_results']
        st.metric("MAE", f"${lr['mae']:.2f}")
        st.metric("RMSE", f"${lr['rmse']:.2f}")
        st.metric("R²", f"{lr['r2']:.4f}")
    
    with col2:
        st.markdown("**Random Forest**")
        rf = results['rf_results']
        st.metric("MAE", f"${rf['mae']:.2f}")
        st.metric("RMSE", f"${rf['rmse']:.2f}")
        st.metric("R²", f"{rf['r2']:.4f}")
    
    st.info(f"🏆 **Best Model: {results['best_model_name']}** (lowest MAE)")
    
    # ---- STEP 4: Prediction Chart ----
    st.markdown("---")
    st.markdown("### 📈 Actual vs Predicted Prices")
    
    # Use the better model's predictions
    if results['best_model_name'] == "Linear Regression":
        best_results = results['lr_results']
    else:
        best_results = results['rf_results']
    
    # Create prediction comparison chart
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Scatter(
        x=X_test.index, y=best_results['actuals'],
        mode='lines', name='Actual Price',
        line=dict(color='#2196F3', width=2)
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=X_test.index, y=best_results['predictions'],
        mode='lines', name='Predicted Price',
        line=dict(color='#FF5722', width=2, dash='dash')
    ))
    
    fig_pred.update_layout(
        title=f'{ticker} — Actual vs Predicted ({results["best_model_name"]})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # ---- STEP 5: Feature Importance ----
    st.markdown("---")
    st.markdown("### 🏆 Feature Importance")
    
    imp_df = results['rf_importance']
    if imp_df is not None:
        top_features = imp_df.head(15)
        
        fig_imp = go.Figure(go.Bar(
            x=top_features['importance'].values[::-1],
            y=top_features['feature'].values[::-1],
            orientation='h',
            marker_color='#a78bfa',
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        ))
        
        fig_imp.update_layout(
            title='Top 15 Most Important Features (Random Forest)',
            xaxis_title='Importance Score',
            template='plotly_dark',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    # ---- Default state: Instructions ----
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Configure
        Select a stock ticker and date range from the sidebar.
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Run
        Click **"🚀 Run Prediction"** to start the analysis.
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Analyze
        View predictions, compare models, and explore feature importance.
        """)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color: rgba(255,255,255,0.5);'>"
        "Built with ❤️ using Python, scikit-learn, and Streamlit — DevelopersHub AI/ML Internship"
        "</p>",
        unsafe_allow_html=True
    )

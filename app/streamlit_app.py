# =============================================
# app/streamlit_app.py — Premium Stock Dashboard
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import fetch_stock_data, preprocess_data
from src.feature_engineering import build_features
from src.model import split_time_series, train_and_evaluate_all

# ---- Page Config ----
st.set_page_config(page_title="StockSense AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---- Premium CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #1a1a3e 40%, #24243e 70%, #0f0c29 100%);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b24 0%, #151340 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    background: linear-gradient(90deg, #a78bfa, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(59,130,246,0.08) 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px; padding: 20px;
    box-shadow: 0 4px 30px rgba(139,92,246,0.06);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(139,92,246,0.4);
    box-shadow: 0 8px 40px rgba(139,92,246,0.12);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; letter-spacing: 0.5px; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; }

/* Hero */
.hero-title {
    font-size: 3.2rem; font-weight: 900; text-align: center; margin-bottom: 4px; line-height: 1.1;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 25%, #c084fc 50%, #f472b6 75%, #fb923c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite alternate;
}
@keyframes shimmer { 0%{filter:brightness(1)} 100%{filter:brightness(1.3)} }
.hero-sub { text-align:center; color:rgba(255,255,255,0.5); font-size:1.05rem; margin-bottom:2.5rem; font-weight:300; }

/* Section headers */
.section-header {
    font-size: 1.5rem; font-weight: 700; margin: 1.5rem 0 1rem 0;
    background: linear-gradient(90deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: flex; align-items: center; gap: 10px;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 28px; margin: 16px 0;
    backdrop-filter: blur(20px);
}
.glass-card h3 { color: #c4b5fd; margin-top: 0; font-weight: 600; }
.glass-card p { color: rgba(255,255,255,0.65); line-height: 1.6; }

/* Winner banner */
.winner-banner {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(16,185,129,0.12) 100%);
    border: 1px solid rgba(34,197,94,0.3); border-radius: 16px;
    padding: 20px 28px; text-align: center; margin: 20px 0;
}
.winner-banner .trophy { font-size: 2rem; }
.winner-banner .winner-text { color: #4ade80; font-size: 1.3rem; font-weight: 700; }
.winner-banner .winner-sub { color: rgba(255,255,255,0.5); font-size: 0.9rem; }

/* Landing cards */
.landing-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.06) 0%, rgba(59,130,246,0.06) 100%);
    border: 1px solid rgba(139,92,246,0.15); border-radius: 20px;
    padding: 32px 24px; text-align: center; transition: all 0.3s ease;
    min-height: 200px;
}
.landing-card:hover { border-color: rgba(139,92,246,0.4); transform: translateY(-4px); box-shadow: 0 12px 40px rgba(139,92,246,0.1); }
.landing-card .icon { font-size: 2.5rem; margin-bottom: 12px; }
.landing-card h3 { color: #c4b5fd; font-weight: 700; margin: 8px 0; }
.landing-card p { color: rgba(255,255,255,0.5); font-size: 0.9rem; line-height: 1.5; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(139,92,246,0.08); border-radius: 12px; border: 1px solid rgba(139,92,246,0.15);
    color: rgba(255,255,255,0.6); padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2)) !important;
    border-color: rgba(139,92,246,0.4) !important; color: #fff !important;
}

/* Footer */
.footer { text-align:center; color:rgba(255,255,255,0.3); padding:2rem 0 1rem 0; font-size:0.85rem; }
.footer a { color: #a78bfa; text-decoration: none; }

/* Divider */
hr { border:none; height:1px; background:linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent); margin:2rem 0; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- HERO HEADER ----
st.markdown('<p class="hero-title">🧠 StockSense AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Next-day stock price prediction powered by Machine Learning</p>', unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    popular_stocks = {
        'AAPL': '🍎 Apple Inc.', 'GOOGL': '🔍 Alphabet', 'MSFT': '💻 Microsoft',
        'TSLA': '⚡ Tesla', 'AMZN': '📦 Amazon', 'META': '👤 Meta',
        'NVDA': '🎮 NVIDIA', 'NFLX': '🎬 Netflix'
    }
    ticker = st.selectbox("Select Stock", list(popular_stocks.keys()),
                          format_func=lambda x: f"{x} — {popular_stocks[x]}", index=0)

    custom = st.text_input("Or enter custom ticker", placeholder="e.g., AMD, UBER")
    if custom: ticker = custom.upper()

    st.markdown("---")
    st.markdown("#### 📅 Date Range")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start", value=datetime.now() - timedelta(days=365*3), max_value=datetime.now())
    with c2: end_date = st.date_input("End", value=datetime.now(), max_value=datetime.now())

    st.markdown("---")
    st.markdown("#### ⚙️ Model Config")
    test_pct = st.slider("Test Split %", 10, 40, 20, help="% of data for testing") / 100

    st.markdown("---")
    run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:10px;'>
        <p style='color:rgba(255,255,255,0.4); font-size:0.8rem;'>Built by</p>
        <p style='color:#a78bfa; font-weight:600;'>Ammar Akbar</p>
        <p style='color:rgba(255,255,255,0.3); font-size:0.75rem;'>DevelopersHub AI/ML Internship</p>
    </div>
    """, unsafe_allow_html=True)

# ---- MAIN CONTENT ----
if run:
    # STEP 1: Fetch data
    with st.spinner(f"📡 Fetching {ticker} market data..."):
        try:
            df = fetch_stock_data(ticker=ticker, start=str(start_date), end=str(end_date))
            df = preprocess_data(df)
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            st.stop()

    # ---- Stock Overview Metrics ----
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Market Overview</p>', unsafe_allow_html=True)

    latest = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    change_pct = ((latest - prev) / prev) * 100
    high_52w = df['Close'].tail(252).max()
    low_52w = df['Close'].tail(252).min()
    avg_vol = df['Volume'].mean()
    total_return = ((latest - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Current Price", f"${latest:.2f}", f"{change_pct:+.2f}%")
    with c2: st.metric("52W High", f"${high_52w:.2f}")
    with c3: st.metric("52W Low", f"${low_52w:.2f}")
    with c4: st.metric("Avg Volume", f"{avg_vol/1e6:.1f}M")
    with c5: st.metric("Total Return", f"{total_return:+.1f}%")

    # ---- Tabs for organized content ----
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🤖 Model Results", "📊 Predictions", "🏆 Feature Analysis"])

    with tab1:
        # Candlestick + Volume + Moving Averages
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC', increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
            increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef4444'
        ), row=1, col=1)

        # Add moving averages
        for w, c, n in [(7,'#fbbf24','7D MA'), (21,'#a78bfa','21D MA'), (50,'#3b82f6','50D MA')]:
            ma = df['Close'].rolling(w).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma, name=n, line=dict(color=c, width=1.2), opacity=0.8), row=1, col=1)

        vol_colors = ['#22c55e' if r['Close']>=r['Open'] else '#ef4444' for _,r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=vol_colors, opacity=0.5), row=2, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=False, template='plotly_dark', height=650, showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)),
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis2=dict(gridcolor='rgba(255,255,255,0.05)'),
            xaxis2=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)

    # STEP 2: Train models
    with st.spinner("🧠 Engineering features & training models..."):
        df_feat = build_features(df)
        X_train, X_test, y_train, y_test = split_time_series(df_feat, test_size=test_pct)
        results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    lr, rf = results['lr_results'], results['rf_results']
    best_name = results['best_model_name']
    best_res = lr if best_name == "Linear Regression" else rf

    with tab2:
        st.markdown('<p class="section-header">🤖 Model Performance Comparison</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            is_winner_lr = best_name == "Linear Regression"
            st.markdown(f"""
            <div class="glass-card" style="{'border-color: rgba(34,197,94,0.3);' if is_winner_lr else ''}">
                <h3>{'🏆 ' if is_winner_lr else ''}Linear Regression</h3>
                <p style="font-size:2.2rem; font-weight:800; color:#fff; margin:8px 0;">${lr['mae']:.2f}</p>
                <p style="color:rgba(255,255,255,0.4); margin-top:-8px;">Mean Absolute Error</p>
                <hr style="margin:12px 0;">
                <p>RMSE: <strong style="color:#fff;">${lr['rmse']:.2f}</strong></p>
                <p>R² Score: <strong style="color:#fff;">{lr['r2']:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            is_winner_rf = best_name == "Random Forest"
            st.markdown(f"""
            <div class="glass-card" style="{'border-color: rgba(34,197,94,0.3);' if is_winner_rf else ''}">
                <h3>{'🏆 ' if is_winner_rf else ''}Random Forest</h3>
                <p style="font-size:2.2rem; font-weight:800; color:#fff; margin:8px 0;">${rf['mae']:.2f}</p>
                <p style="color:rgba(255,255,255,0.4); margin-top:-8px;">Mean Absolute Error</p>
                <hr style="margin:12px 0;">
                <p>RMSE: <strong style="color:#fff;">${rf['rmse']:.2f}</strong></p>
                <p>R² Score: <strong style="color:#fff;">{rf['r2']:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="winner-banner">
            <span class="trophy">🏆</span><br>
            <span class="winner-text">{best_name} wins!</span><br>
            <span class="winner-sub">Lowest prediction error (MAE: ${best_res['mae']:.2f})</span>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<p class="section-header">📊 Actual vs Predicted Prices</p>', unsafe_allow_html=True)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=X_test.index, y=best_res['actuals'], mode='lines', name='Actual',
            line=dict(color='#3b82f6', width=2.5),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.05)'
        ))
        fig_pred.add_trace(go.Scatter(
            x=X_test.index, y=best_res['predictions'], mode='lines', name='Predicted',
            line=dict(color='#f472b6', width=2.5, dash='dot')
        ))
        fig_pred.update_layout(
            template='plotly_dark', height=500,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='Price ($)', xaxis_title='Date',
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Scatter plot
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=best_res['actuals'], y=best_res['predictions'], mode='markers',
            marker=dict(color='#a78bfa', size=8, opacity=0.6, line=dict(width=1, color='rgba(255,255,255,0.2)')),
            name='Predictions'
        ))
        mn, mx = min(best_res['actuals'].min(), best_res['predictions'].min()), max(best_res['actuals'].max(), best_res['predictions'].max())
        fig_sc.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines', name='Perfect', line=dict(color='#22c55e', dash='dash', width=2)))
        fig_sc.update_layout(
            template='plotly_dark', height=450, xaxis_title='Actual ($)', yaxis_title='Predicted ($)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab4:
        st.markdown('<p class="section-header">🏆 Feature Importance Analysis</p>', unsafe_allow_html=True)
        imp_df = results['rf_importance']
        if imp_df is not None:
            top = imp_df.head(12)
            colors = [f'rgba(139,92,246,{0.4 + 0.6*(1-i/len(top))})' for i in range(len(top))]
            fig_imp = go.Figure(go.Bar(
                x=top['importance'].values[::-1], y=top['feature'].values[::-1],
                orientation='h', marker=dict(color=colors[::-1], line=dict(width=0)),
            ))
            fig_imp.update_layout(
                template='plotly_dark', height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), xaxis_title='Importance Score',
                yaxis=dict(gridcolor='rgba(255,255,255,0.03)'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<p class="footer">Built with ❤️ by <a href="https://github.com/blackmangoo">Ammar Akbar</a> · DevelopersHub AI/ML Internship · Python · scikit-learn · Streamlit</p>', unsafe_allow_html=True)

else:
    # ---- Landing Page ----
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="landing-card">
            <div class="icon">📡</div>
            <h3>Real-Time Data</h3>
            <p>Fetches live stock data from Yahoo Finance for any ticker worldwide</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="landing-card">
            <div class="icon">🧠</div>
            <h3>AI Prediction</h3>
            <p>Compares Linear Regression vs Random Forest to find the best model</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="landing-card">
            <div class="icon">📊</div>
            <h3>Deep Analysis</h3>
            <p>Interactive charts, feature importance, and model performance metrics</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="text-align:center; max-width:600px; margin:0 auto;">
        <h3 style="color:#a78bfa;">👈 Get Started</h3>
        <p>Select a stock from the sidebar and click <strong style="color:#a78bfa;">Run Prediction</strong> to begin</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="footer">Built with ❤️ by <a href="https://github.com/blackmangoo">Ammar Akbar</a> · DevelopersHub AI/ML Internship</p>', unsafe_allow_html=True)

# 🧠 StockSense AI — Stock Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> An end-to-end machine learning pipeline that predicts next-day stock closing prices using historical market data, technical indicators, and regression models. Features an interactive Streamlit dashboard with premium UI.

## 📋 Task Objective

Build a complete ML pipeline that:
- Fetches real-time stock data from **Yahoo Finance**
- Engineers **15+ technical features** (SMA, RSI, lag features, volume ratios)
- Trains and compares **Linear Regression** vs **Random Forest** models
- Deploys an interactive **Streamlit dashboard** with live predictions
- Containerized with **Docker** for easy deployment

## 📁 Project Structure

```
stock-price-predictor/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker containerization
├── .gitignore
├── app/
│   └── streamlit_app.py             # 🎨 Premium Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Fetch & preprocess stock data
│   ├── feature_engineering.py       # Technical indicators & features
│   ├── model.py                     # Train & evaluate ML models
│   └── visualize.py                 # Plotting functions
├── notebooks/
│   └── stock_prediction.ipynb       # Full analysis notebook
├── data/                            # Downloaded stock data (gitignored)
├── models/                          # Saved trained models
└── results/                         # Plots and evaluation metrics
```

## 🧠 Pipeline Steps

```
1. Data Loading (yfinance API)
   ↓
2. Data Preprocessing (missing values, duplicates, sorting)
   ↓
3. Feature Engineering
   • Lag features (1, 2, 3, 5, 7 days)
   • Moving Averages (SMA_7, SMA_21, SMA_50)
   • RSI (Relative Strength Index, 14-day)
   • Daily returns, price range, open-close diff
   • Volume change, volume ratio
   ↓
4. Chronological Train/Test Split (80/20)
   ↓
5. Model Training
   • Linear Regression (with StandardScaler)
   • Random Forest Regressor (100 trees)
   ↓
6. Evaluation (MAE, RMSE, R²)
   ↓
7. Interactive Dashboard (Streamlit)
```

## 📊 Results

### Model Performance (AAPL, 2022-2024)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Linear Regression** ✅ | **$3.09** | **$3.89** | **0.886** |
| Random Forest | $35.53 | $37.09 | -9.37 |

**🏆 Winner: Linear Regression** — On average, predictions are only **$3.09** off from the actual price.

### Key Insights

1. **Linear Regression outperformed Random Forest** on this time-series task because lag features contain absolute price values that drift over time — LR's linear weights can extrapolate, but RF's tree splits cannot
2. **Close_lag_1** (yesterday's close) is the strongest predictor — stock prices are highly autocorrelated
3. **SMA_7 and daily_return** are the next most important features, capturing short-term momentum
4. **RSI** contributes meaningful signal about overbought/oversold conditions

## 🚀 Quick Start

### Local Setup
```bash
# Clone the repository
git clone https://github.com/blackmangoo/stock-price-predictor.git
cd stock-price-predictor

# Create environment & install dependencies
conda create -n stock-predictor python=3.11 -y
conda activate stock-predictor
pip install -r requirements.txt

# Run the dashboard
streamlit run app/streamlit_app.py
```

### Docker
```bash
docker build -t stock-predictor .
docker run -p 8501:8501 stock-predictor
# Access at http://localhost:8501
```

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| yfinance | Stock data API |
| pandas / numpy | Data manipulation |
| scikit-learn | Machine Learning |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| Docker | Containerization |
| ta (Technical Analysis) | RSI calculation |

## 📄 License

This project is licensed under the MIT License.

---

**Internship Task 2** — DevelopersHub Corporation AI/ML Engineering Internship | May 2026

Built by [Ammar Akbar](https://github.com/blackmangoo)
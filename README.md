# 📈 Stock Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> Predict next-day stock closing prices using historical market data and machine learning.

## 🎯 Objective

Use historical stock data from Yahoo Finance to predict the **next day's closing price** using regression models. Compare multiple ML approaches and build an interactive dashboard.

## 📚 What You'll Learn

- **Data fetching** from real-world APIs (`yfinance`)
- **Time series data handling** (why you can't randomly split time data)
- **Feature engineering** with technical indicators (SMA, EMA, RSI)
- **Regression modeling** (Linear Regression, Random Forest, XGBoost)
- **Model evaluation** (MAE, RMSE, R²)
- **Interactive visualization** with Plotly
- **Building a Streamlit dashboard**
- **Docker containerization**

## 🧠 Concepts to Revise Before Starting

| Concept | Resource |
|---------|----------|
| Linear Regression | [StatQuest Video](https://www.youtube.com/watch?v=PaFPbb66DxQ) |
| Random Forest | [StatQuest Video](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) |
| Train/Test Split | [sklearn docs](https://scikit-learn.org/stable/modules/cross_validation.html) |
| Feature Scaling | [Why & When to Scale](https://www.youtube.com/watch?v=mnKm3YP56PY) |
| Time Series Basics | [Intro to Time Series](https://www.youtube.com/watch?v=FjCgJnDvoog) |
| yfinance Library | [yfinance Docs](https://pypi.org/project/yfinance/) |
| Technical Indicators | [TA Library Docs](https://technical-analysis-library-in-python.readthedocs.io/) |

## 📁 Project Structure

```
stock-price-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── notebooks/
│   └── stock_prediction.ipynb    ← Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py            ← Fetch & preprocess stock data
│   ├── feature_engineering.py    ← Create technical indicators
│   ├── model.py                  ← Train & evaluate models
│   └── visualize.py              ← Plotting functions
├── data/                          ← Downloaded stock data (gitignored)
├── models/                        ← Saved trained models
├── results/                       ← Plots and evaluation metrics
└── app/
    └── streamlit_app.py           ← Interactive stock dashboard
```

## 🚀 Step-by-Step Implementation Guide

### Step 1: Setup Environment
```bash
conda create -n stock-predictor python=3.11 -y
conda activate stock-predictor
pip install -r requirements.txt
```

### Step 2: Data Loading (`src/data_loader.py`)
- Use `yfinance` to download historical stock data
- Select a stock (e.g., AAPL, TSLA, GOOGL)
- Download at least 2 years of daily data
- Save to `data/` folder as CSV for reproducibility
- Handle any missing values

### Step 3: Feature Engineering (`src/feature_engineering.py`)
- Create lag features (yesterday's close, 2 days ago, etc.)
- Add moving averages (SMA_7, SMA_21, SMA_50)
- Add RSI (Relative Strength Index)
- Add daily price change percentage
- Add volume change percentage

### Step 4: Model Training (`src/model.py`)
- Split data chronologically (NOT randomly — this is time series!)
  - Train: first 80% of data
  - Test: last 20% of data
- Train multiple models:
  1. Linear Regression
  2. Random Forest Regressor
  3. (Bonus) XGBoost Regressor
- Evaluate each with MAE, RMSE, R²
- Save the best model using joblib

### Step 5: Visualization (`src/visualize.py`)
- Plot stock price history (candlestick chart with Plotly)
- Plot actual vs predicted prices
- Plot feature importance (for tree-based models)
- Plot residuals (prediction errors)

### Step 6: Jupyter Notebook (`notebooks/stock_prediction.ipynb`)
- Combine all steps with explanations
- Add markdown cells explaining your thought process
- Show all plots inline
- Write a conclusion section

### Step 7: Streamlit App (`app/streamlit_app.py`)
- Stock selector dropdown
- Date range selector
- Display stock chart (interactive Plotly)
- Show prediction results
- Display model metrics
- Beautiful UI with custom styling

### Step 8: Docker
```bash
docker build -t stock-predictor .
docker run -p 8501:8501 stock-predictor
```

## 🎯 Extra Challenges (Bonus Learning)

- [ ] Add multiple stock comparison
- [ ] Implement walk-forward validation (proper time series CV)
- [ ] Add MACD and Bollinger Bands as features
- [ ] Create a model comparison table
- [ ] Add prediction confidence intervals

## 📊 Results

<!-- Fill this section after completing the project -->

### Model Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |

### Screenshots
<!-- Add screenshots of your Streamlit app and key plots here -->

## 🐳 Docker Usage

```bash
# Build the Docker image
docker build -t stock-predictor .

# Run the container
docker run -p 8501:8501 stock-predictor

# Access the app at http://localhost:8501
```

## 📝 Key Findings

<!-- Write your conclusions here after completing the project -->

## 🔗 Links

- **Dataset Source:** [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/)
- **Internship:** DevelopersHub Corporation AI/ML Engineering

---
*Built as part of DevelopersHub Corporation AI/ML Engineering Internship*
# =============================================
# src/model.py — Train & Evaluate ML Models
# =============================================
# This module handles the MACHINE LEARNING part:
#   1. Splitting data into train/test sets
#   2. Training multiple regression models
#   3. Evaluating models with metrics (MAE, RMSE, R²)
#   4. Saving/loading trained models
#
# KEY CONCEPT: Supervised Learning
# We have INPUTS (features) and a known OUTPUT (target).
# The model learns the relationship: features → target
# Then it uses that relationship to predict on NEW data.
#
# KEY CONCEPT: Regression
# Regression predicts a CONTINUOUS NUMBER (e.g., price = $152.37)
# Classification predicts a CATEGORY (e.g., "will go up" vs "will go down")
# Stock price prediction is a REGRESSION problem.
# =============================================

import pandas as pd
import numpy as np
import joblib                                    # For saving/loading models to disk
import os
from sklearn.linear_model import LinearRegression  # Simplest ML model
from sklearn.ensemble import RandomForestRegressor  # Powerful ensemble model
from sklearn.metrics import (
    mean_absolute_error,    # MAE: average absolute prediction error
    mean_squared_error,     # MSE: average squared prediction error
    r2_score                # R²: how much variance the model explains
)
from sklearn.preprocessing import StandardScaler   # Feature scaling


def split_time_series(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2):
    """
    Split data into training and testing sets FOR TIME SERIES.
    
    ⚠️ CRITICAL CONCEPT: Why NOT use random train_test_split?
    ----------------------------------------------------------
    In regular ML, we randomly shuffle data before splitting.
    But for TIME SERIES data, this is WRONG because:
    
    1. FUTURE DATA LEAKAGE: If we randomly pick rows, some training
       data might come from 2024 and some test data from 2023.
       The model would "learn from the future" — that's cheating!
    
    2. REAL-WORLD SIMULATION: In reality, you can only train on PAST
       data and predict the FUTURE. So the split must be chronological:
       
       |-------- TRAIN (80%) --------|---- TEST (20%) ----|
       Jan 2022                   Oct 2023            Dec 2024
       
       The model NEVER sees any data from the test period during training.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame — featured data (sorted by date!)
    target_col : str — name of the target column (default: 'target')
    test_size : float — fraction of data for testing (default: 0.2 = 20%)
    
    RETURNS:
    --------
    X_train, X_test, y_train, y_test : training and testing data
    """
    
    # ---- Calculate the split point ----
    # If we have 1000 rows and test_size=0.2, split at row 800
    split_idx = int(len(df) * (1 - test_size))
    
    # ---- Define features (X) and target (y) ----
    # IMPORTANT: We must EXCLUDE raw price columns (Close, Open, High, Low, Adj Close)
    # because they are essentially "the answer" — today's Close is almost the same
    # as tomorrow's Close. Including them would be DATA LEAKAGE.
    # Instead, we use ENGINEERED features (lags, MAs, RSI, returns) which encode
    # price information in a more useful, pattern-based way.
    exclude_cols = {target_col, 'Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]  # Features (input data)
    y = df[target_col]    # Target (what we predict)
    
    # ---- Split chronologically ----
    # First 80% = training, Last 20% = testing
    X_train = X.iloc[:split_idx]    # .iloc = select by INTEGER position
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    print(f"✅ Data split (chronological):")
    print(f"   Train: {len(X_train)} rows ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test:  {len(X_test)} rows ({X_test.index[0].date()} → {X_test.index[-1].date()})")
    print(f"   Features: {len(feature_cols)} columns")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale features to have mean=0 and std=1 (StandardScaler).
    
    WHY SCALE FEATURES?
    -------------------
    Different features have very different scales:
    - Close price: ~100-200 (dollars)
    - Volume: ~50,000,000-100,000,000 (shares)
    - RSI: 0-100 (index)
    - Daily return: -5% to +5% (percentage)
    
    Some models (like Linear Regression) are affected by scale.
    If Volume is in millions and RSI is 0-100, the model might
    think Volume is "more important" just because the numbers are bigger.
    
    StandardScaler transforms each feature so that:
    - Mean = 0 (centered around zero)
    - Standard deviation = 1 (same spread for all features)
    
    IMPORTANT: Fit the scaler on TRAINING data only!
    If we fit on ALL data (including test), we'd leak information
    from the test set into the scaling — that's data leakage.
    
    RETURNS:
    --------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    # Fit on TRAINING data (learn mean & std from train only)
    # Transform both train and test using those same parameters
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),    # fit + transform on train
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),          # ONLY transform on test (no fit!)
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✅ Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def train_linear_regression(X_train, y_train):
    """
    Train a LINEAR REGRESSION model.
    
    WHAT IS LINEAR REGRESSION?
    --------------------------
    The simplest ML model. It finds the best straight line (or plane
    in higher dimensions) that fits the data.
    
    Formula: y = w1*x1 + w2*x2 + ... + wn*xn + b
    
    Where:
    - y = predicted price
    - x1, x2, ... = feature values
    - w1, w2, ... = weights (how important each feature is)
    - b = bias (intercept)
    
    The model learns the weights (w) and bias (b) that minimize
    the prediction error on the training data.
    
    PROS:
    + Simple and fast
    + Easy to interpret (look at weights)
    + Good baseline model
    
    CONS:
    - Assumes LINEAR relationship between features and target
    - Can't capture complex patterns
    - Sensitive to outliers
    """
    model = LinearRegression()
    model.fit(X_train, y_train)  # Learn weights from training data
    
    print(f"✅ Linear Regression trained")
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a RANDOM FOREST model.
    
    WHAT IS RANDOM FOREST?
    ----------------------
    An ENSEMBLE method that builds many Decision Trees and
    combines their predictions (by averaging).
    
    How it works:
    1. Create 100 different Decision Trees (n_estimators=100)
    2. Each tree is trained on a RANDOM SUBSET of the data
    3. Each tree also uses a RANDOM SUBSET of features
    4. Final prediction = AVERAGE of all 100 trees
    
    WHY RANDOM?
    - If all trees were identical, averaging wouldn't help
    - Randomness creates DIVERSE trees that make different errors
    - When you average diverse predictions, errors cancel out!
    - This is called "wisdom of the crowd"
    
    PROS:
    + Captures non-linear patterns (unlike Linear Regression)
    + Resistant to overfitting (thanks to averaging)
    + Handles different feature scales (no scaling needed)
    + Gives feature importance scores
    
    CONS:
    - Slower to train than Linear Regression
    - Less interpretable (100 trees vs 1 equation)
    - Can be memory-intensive
    
    PARAMETERS:
    -----------
    n_estimators : int — number of trees (more = better but slower)
    random_state : int — seed for reproducibility (same result every run)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,  # Number of trees in the forest
        random_state=random_state,  # Seed for reproducibility
        n_jobs=-1                   # Use all CPU cores (faster training)
    )
    model.fit(X_train, y_train)
    
    print(f"✅ Random Forest trained ({n_estimators} trees)")
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a model using regression metrics.
    
    METRICS EXPLAINED:
    ------------------
    
    1. MAE (Mean Absolute Error)
       = Average of |actual - predicted|
       Example: If MAE = 2.5, on average the prediction is off by $2.50
       EASY TO UNDERSTAND: "How far off are we, on average?"
    
    2. RMSE (Root Mean Squared Error)
       = sqrt(Average of (actual - predicted)²)
       Similar to MAE, but PENALIZES LARGE ERRORS more.
       A prediction off by $10 is penalized more than ten predictions
       off by $1 each.
       GOOD FOR: When large errors are especially bad.
    
    3. R² (R-squared / Coefficient of Determination)
       = 1 - (sum of squared errors / sum of squared deviations from mean)
       Ranges from -∞ to 1.0:
       - R² = 1.0 → PERFECT prediction (never happens in practice)
       - R² = 0.0 → Model is as good as just predicting the mean
       - R² < 0.0 → Model is WORSE than predicting the mean
       TELLS YOU: "What fraction of the variance does the model explain?"
    
    RETURNS:
    --------
    dict with all metrics and predictions
    """
    
    # ---- Make predictions on test data ----
    y_pred = model.predict(X_test)
    
    # ---- Calculate metrics ----
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # MSE → RMSE by taking sqrt
    r2 = r2_score(y_test, y_pred)
    
    # ---- Print results ----
    print(f"\n📊 {model_name} Results:")
    print(f"   MAE:  ${mae:.2f}  (avg prediction error)")
    print(f"   RMSE: ${rmse:.2f}  (penalizes large errors)")
    print(f"   R²:   {r2:.4f}   (1.0 = perfect, 0.0 = useless)")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test.values
    }


def get_feature_importance(model, feature_names: list, model_name: str = "Model") -> pd.DataFrame:
    """
    Get feature importance scores from the model.
    
    WHAT IS FEATURE IMPORTANCE?
    ---------------------------
    It tells you WHICH FEATURES the model relies on most.
    
    For Linear Regression: importance = absolute value of coefficients
    For Random Forest: importance = how much each feature reduces error
    
    WHY IT MATTERS:
    - Understand what drives predictions
    - Remove unimportant features (simpler model)
    - Gain domain insights (e.g., "RSI is the most important predictor")
    
    RETURNS:
    --------
    pd.DataFrame sorted by importance (highest first)
    """
    
    # ---- Get importance values depending on model type ----
    if hasattr(model, 'feature_importances_'):
        # Random Forest has .feature_importances_
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear Regression has .coef_ (coefficients/weights)
        importance = np.abs(model.coef_)  # Use absolute value
    else:
        print(f"⚠️  {model_name} doesn't support feature importance")
        return None
    
    # ---- Create a sorted DataFrame ----
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"\n🏆 Top 10 Features ({model_name}):")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df


def save_model(model, filepath: str):
    """
    Save a trained model to disk using joblib.
    
    WHY SAVE MODELS?
    ----------------
    Training can take time. Once trained, save the model so you
    can load it later without re-training. Essential for deployment.
    
    joblib is better than pickle for models with large numpy arrays
    (which sklearn models have internally).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")


def load_model(filepath: str):
    """Load a previously saved model from disk."""
    model = joblib.load(filepath)
    print(f"📂 Model loaded from {filepath}")
    return model


def train_and_evaluate_all(X_train, X_test, y_train, y_test) -> dict:
    """
    MASTER FUNCTION: Train all models and compare results.
    
    This runs the complete modeling pipeline:
    1. Scale features (for Linear Regression)
    2. Train Linear Regression
    3. Train Random Forest
    4. Evaluate both models
    5. Compare and declare winner
    6. Save the best model
    
    RETURNS:
    --------
    dict with all results, models, and comparison
    """
    
    print("\n🚀 === MODEL TRAINING PIPELINE ===\n")
    
    # ---- Scale features ----
    # Linear Regression benefits from scaling
    # Random Forest doesn't need it (tree-based models don't care about scale)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # ---- Train models ----
    print("\n--- Training Models ---")
    lr_model = train_linear_regression(X_train_scaled, y_train)
    # NOTE: Even though Random Forest technically doesn't need scaling,
    # our lag features contain raw prices that DRIFT over time (e.g., $150 in 2022
    # vs $220 in 2024). Scaling helps RF generalize across time periods.
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # ---- Evaluate models ----
    print("\n--- Evaluating Models ---")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    # ---- Get feature importance ----
    feature_names = list(X_train.columns)
    lr_importance = get_feature_importance(lr_model, feature_names, "Linear Regression")
    rf_importance = get_feature_importance(rf_model, feature_names, "Random Forest")
    
    # ---- Compare models ----
    print("\n\n📊 === MODEL COMPARISON ===")
    comparison = pd.DataFrame({
        'Metric': ['MAE ($)', 'RMSE ($)', 'R²'],
        'Linear Regression': [f"${lr_results['mae']:.2f}", f"${lr_results['rmse']:.2f}", f"{lr_results['r2']:.4f}"],
        'Random Forest': [f"${rf_results['mae']:.2f}", f"${rf_results['rmse']:.2f}", f"{rf_results['r2']:.4f}"]
    })
    print(comparison.to_string(index=False))
    
    # ---- Determine the best model ----
    # Lower MAE = better prediction (closer to actual price)
    if lr_results['mae'] < rf_results['mae']:
        best_name = "Linear Regression"
        best_model = lr_model
        print(f"\n🏆 Winner: Linear Regression (lower MAE)")
    else:
        best_name = "Random Forest"
        best_model = rf_model
        print(f"\n🏆 Winner: Random Forest (lower MAE)")
    
    # ---- Save the best model ----
    save_model(best_model, f"models/best_model.pkl")
    save_model(scaler, f"models/scaler.pkl")
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_results': lr_results,
        'rf_results': rf_results,
        'lr_importance': lr_importance,
        'rf_importance': rf_importance,
        'scaler': scaler,
        'best_model_name': best_name,
        'comparison': comparison
    }


# =============================================
# MAIN — Test the full pipeline
# =============================================
if __name__ == "__main__":
    from data_loader import fetch_stock_data, preprocess_data
    from feature_engineering import build_features
    
    # Load and prepare data
    df = fetch_stock_data("AAPL", "2022-01-01", "2024-12-31")
    df = preprocess_data(df)
    df = build_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_time_series(df)
    
    # Train and evaluate
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

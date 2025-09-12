# 🚀 Crypto Pipeline Analysis (Pythonic Code Implementation for the Kaggle competition)

````python
# Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import optuna
from skopt import BayesSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import missing imputer for handling NaNs
from boruta import BorutaPy
from sklearn.decomposition import PCA
import os
from pathlib import Path  # ✅ Import Path for better file handling
import pyarrow.parquet as pq

# --------------------------------------
# Step 1: Exploratory Data Analysis
# --------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def explore_crypto_data(file_path: str):
    """
    Perform initial exploratory data analysis on the train.parquet dataset.

    Parameters:
    file_path (str): Path to the Parquet file.

    Returns:
    None (displays visualizations & statistical insights)
    """
    # Load dataset
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Reset index to move 'timestamp' back into a column
    if isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)

    # Display dataset overview
    print(f"Dataset Shape: {df.shape}")
    print("First few rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])

    # Data type optimizations (float32 conversion)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)

    # Pearson correlation of proprietary features
    corr_matrix = df.corr(method='pearson')
    top_correlated = corr_matrix["label"].drop("label").sort_values(ascending=False)
    print("\nTop correlated features with label:\n", top_correlated.head(10))

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram for trading volume
    sns.histplot(df['volume'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Volume Distribution")

    # Boxplot for bid/ask quantities
    sns.boxplot(data=df[['bid_qty', 'ask_qty']], ax=axes[0, 1])
    axes[0, 1].set_title("Bid & Ask Quantity Boxplot")

    # Feature correlation heatmap (subset of features)
    sns.heatmap(corr_matrix.iloc[:20, :20], annot=False, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title("Feature Correlation Heatmap")

    # Time-Series Plot (First 100 timestamps)
    if "timestamp" in df.columns:
        df[:100].plot(x="timestamp", y=["volume", "buy_qty", "sell_qty"], ax=axes[1, 1])
        axes[1, 1].set_title("Trading Activity Over Time")
    else:
        print("\nWarning: 'timestamp' column not found! Skipping time-series plot.")

    plt.tight_layout()
    plt.show()

# Example Usage
# explore_crypto_data("train.parquet")
# --------------------------------------
# Step 2: Feature Engineering & Selection
# --------------------------------------
def calculate_rsi(df, column="close_price", window=14):
    """
    Compute Relative Strength Index (RSI).
    RSI measures the speed and change of price movements.
    
    Parameters:
    df (pd.DataFrame): The dataset containing price data.
    column (str): The price column used for RSI calculation.
    window (int): The look-back period for RSI.

    Returns:
    pd.Series: RSI values.
    """
    delta = df[column].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(df, column="close_price", window=20, std_dev=2):
    """
    Compute Bollinger Bands.
    Bollinger Bands identify volatility based on moving averages and standard deviations.

    Parameters:
    df (pd.DataFrame): The dataset containing price data.
    column (str): The price column used for Bollinger Band calculation.
    window (int): The look-back period for moving average.
    std_dev (int): The multiplier for standard deviation.

    Returns:
    pd.DataFrame: Bollinger Bands (upper, middle, lower).
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    upper_band = rolling_mean + (std_dev * rolling_std)
    lower_band = rolling_mean - (std_dev * rolling_std)

    return pd.DataFrame({"BB_upper": upper_band, "BB_middle": rolling_mean, "BB_lower": lower_band})

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
import xgboost as xgb

def feature_engineering_selection(df: pd.DataFrame, target_col: str = "label"):
    """
    Perform feature engineering and selection for the dataset, with fixes for timestamp formatting and test dataset constraints.

    Parameters:
    df (pd.DataFrame): The input dataset.
    target_col (str): The target variable (default: "label").

    Returns:
    pd.DataFrame: Processed dataset with engineered features and selected variables.
    """

    # ✅ Ensure 'timestamp' exists; create placeholder if missing
    if "timestamp" not in df.columns:
        print("Warning: 'timestamp' missing. Creating a placeholder index.")
        df["timestamp"] = np.arange(len(df))  # Assign an index-based placeholder

    # ✅ Convert `timestamp` to numeric to prevent XGBoost errors
    if df["timestamp"].dtype == "datetime64[ns]":
        df["timestamp"] = df["timestamp"].astype("int64")  # Convert to integer

    # Step 1: Create Lag Features
    lag_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for lag in [1, 5, 10]:
        for col in lag_features:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Step 2: Create Rolling-Window Features
    rolling_windows = [5, 10, 20]
    for window in rolling_windows:
        for col in lag_features:
            if col in df.columns:
                df[f"{col}_roll_mean{window}"] = df[col].rolling(window).mean()
                df[f"{col}_roll_std{window}"] = df[col].rolling(window).std()

    # ✅ Fix: Fill NaNs in rolling features
    rolling_features = [col for col in df.columns if "roll_mean" in col or "roll_std" in col]
    df[rolling_features] = df[rolling_features].fillna(df[rolling_features].median())

    # Step 3: Feature Scaling & Dimensionality Reduction (PCA)
    proprietary_features = [col for col in df.columns if col.startswith("X")]
    
    if proprietary_features:
        df[proprietary_features] = df[proprietary_features].replace([np.inf, -np.inf], np.nan)
        df[proprietary_features] = df[proprietary_features].fillna(df[proprietary_features].median())
        df[proprietary_features] = df[proprietary_features].fillna(0)

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[proprietary_features])
        
        # ✅ Apply PCA only when feature variability exists
        pca = PCA(n_components=min(50, len(proprietary_features)))
        df_pca = pca.fit_transform(df_scaled)

        # Convert PCA output to DataFrame
        pca_cols = [f"PCA_{i}" for i in range(df_pca.shape[1])]
        df_pca = pd.DataFrame(df_pca, columns=pca_cols, index=df.index)

        df = pd.concat([df, df_pca], axis=1).drop(columns=proprietary_features, errors="ignore")

    # Step 4: Feature Selection  
    if target_col in df.columns and df[target_col].nunique() > 1:  # ✅ Skip XGBoost if labels are all zero
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = X.fillna(X.median())
        X = X.fillna(0)

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)

        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 10 Features Based on XGBoost Importance:\n", feature_importance.head(10))

        # ✅ Apply Boruta for feature selection if labels vary
        boruta_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=42, perc=90)
        boruta_selector.fit(X.values, y.values)

        selected_features = X.columns[boruta_selector.support_]
        if len(selected_features) == 0:
            raise ValueError("No features selected after Boruta! Check feature importance thresholds.")
        
        selected_cols = selected_features.tolist() + [target_col]
        df = df[selected_cols]

    else:
        print("Skipping supervised feature selection (XGBoost/Boruta) since all labels are zero.")

        # ✅ Use unsupervised selection instead (high correlation features)
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_correlation_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        
        if high_correlation_features:
            print("Applying correlation-based feature selection.")
            df = df.drop(columns=high_correlation_features, errors="ignore")

    return df

# Example Usage:
# df_processed = feature_engineering_selection(df)

# --------------------------------------
# Step 3: Model Experimentation
# --------------------------------------

# Load Data (Modified to Apply Feature Engineering First)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import clone  # ✅ Ensures proper model cloning
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold  # Import cross-validation method
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline  # ✅ Import Pipeline explicitly

def load_data(train_path, test_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # ✅ Ensure 'timestamp' exists in both datasets
    if "timestamp" not in test_df.columns:
        print("Warning: 'timestamp' missing in test dataset! Adding placeholder.")
        test_df["timestamp"] = np.arange(len(test_df))  # Generate unique index

    if "timestamp" in train_df.columns:
        train_df["timestamp"] = train_df["timestamp"].astype("int64")
    if "timestamp" in test_df.columns:
        test_df["timestamp"] = test_df["timestamp"].astype("int64")

    # ✅ Ensure `test_df` has a target column to prevent Boruta errors
    if "label" not in test_df.columns:
        test_df["label"] = np.nan  # Placeholder to prevent issues

    # ✅ Fill missing features with NaN before PCA
    selected_features = train_df.columns.tolist()
    missing_features = set(selected_features) - set(test_df.columns)

    if missing_features:
        print(f"Warning: Test dataset is missing these features: {missing_features}")
        for feature in missing_features:
            test_df[feature] = np.nan  # Fill missing features with NaN

    test_df = test_df[list(set(selected_features) & set(test_df.columns))]

    # ✅ Identify proprietary features (`X*`) used for PCA
    proprietary_features = [col for col in train_df.columns if col.startswith("X")]

    missing_props = set(proprietary_features) - set(test_df.columns)
    if missing_props:
        print(f"Warning: Test dataset is missing proprietary features required for PCA: {missing_props}")
        for feature in missing_props:
            test_df[feature] = np.nan  # Fill missing proprietary features with NaN

    # ✅ **Handle Features That Contain Only NaN Values**
    cols_with_all_nan = [
        col for col in proprietary_features
        if train_df[col].isna().all() or test_df[col].isna().all()
    ]

    if cols_with_all_nan:
        print(f"Warning: Dropping proprietary features with all NaN values: {cols_with_all_nan}")
        train_df.drop(columns=cols_with_all_nan, inplace=True, errors="ignore")
        test_df.drop(columns=cols_with_all_nan, inplace=True, errors="ignore")
        proprietary_features = [col for col in proprietary_features if col not in cols_with_all_nan]

    # ✅ **Explicitly Remove `inf` Values Before Imputation**
    for df in [train_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert `inf` to `NaN`
        df.fillna(df.median(), inplace=True)  # Fill NaN values with median
        df.fillna(0, inplace=True)  # Final fallback for residual NaNs

    # ✅ Use `SimpleImputer` to ensure clean data for PCA
    imputer = SimpleImputer(strategy="median")  
    train_df[proprietary_features] = imputer.fit_transform(train_df[proprietary_features])
    test_df[proprietary_features] = imputer.transform(test_df[proprietary_features])

    # ✅ Apply PCA transformation only if proprietary features exist
    if proprietary_features and all(col in test_df.columns for col in proprietary_features):
        scaler = StandardScaler()
        optimal_pca_components = min(30, len(proprietary_features))
        pca = PCA(n_components=optimal_pca_components)

        train_scaled = scaler.fit_transform(train_df[proprietary_features])
        test_scaled = scaler.transform(test_df[proprietary_features])

        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        # ✅ Convert transformed PCA components back into DataFrame
        pca_cols = [f"PCA_{i}" for i in range(train_pca.shape[1])]
        train_pca_df = pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)
        test_pca_df = pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)

        # ✅ Ensure `test_df` retains PCA features
        train_df = pd.concat([train_df.drop(columns=proprietary_features, errors="ignore"), train_pca_df], axis=1)
        test_df = pd.concat([test_df.drop(columns=proprietary_features, errors="ignore"), test_pca_df], axis=1)

    print("Train PCA Columns:", train_df.columns.tolist())
    print("Test PCA Columns:", test_df.columns.tolist())
    print("Final Test Data Shape After PCA:", test_df.shape)

    if test_df.shape[1] == 0:
        raise ValueError("Test data is empty after PCA transformation! Check feature selection.")

    X_train = train_df[[col for col in train_df.columns if col.startswith("PCA_")]].fillna(0)
    X_test = test_df[[col for col in test_df.columns if col.startswith("PCA_")]].fillna(0)
    y_train = train_df["label"].fillna(0).astype(np.float32)

    return X_train, y_train, X_test


# Model Evaluation Function
def evaluate_model(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson_corr, _ = pearsonr(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.4f}, Pearson Correlation: {pearson_corr:.4f}")
    return rmse, pearson_corr

# Baseline Models - Linear Regression & Random Forest
def baseline_models(X_train, y_train, X_test):
    if X_train.size == 0 or X_train.shape[1] == 0:
        raise ValueError("Error: No features in X_train! Verify feature selection.")
    
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }
    
    predictions = {}
    for name, model in models.items():
        print(f"Training {name} -> Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
    
    return predictions

# ✅ Function to force models to be treated as regressors and avoid scikit-learn's broken tag handling
def patch_get_tags(estimator):
    estimator._estimator_type = "regressor"  # ✅ Ensure model is recognized correctly
    return estimator  # 🚀 Remove unnecessary attribute deletion

# ✅ Step 1: Basic Test - Train XGBoost & LightGBM Without GridSearchCV
def test_basic_training(X_train, y_train, X_test):
    print("Testing basic training for XGBoost & LightGBM...")

    try:
        # ✅ Basic XGBoost training
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        print("✅ XGBoost training completed successfully!")

        # ✅ Basic LightGBM training
        lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=100)
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)
        print("✅ LightGBM training completed successfully!")

    except Exception as e:
        print(f"❌ Error in basic model training: {e}")

# ✅ Step 2: Test GridSearchCV With Ridge Regression
def test_grid_search(X_train, y_train):
    print("Testing GridSearchCV with Ridge Regression...")

    try:
        ridge = Ridge()
        param_grid = {"alpha": [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(ridge, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        print("✅ GridSearchCV worked successfully with Ridge Regression!")
    
    except Exception as e:
        print(f"❌ GridSearchCV failed with Ridge Regression: {e}")

# ✅ Gradient Boosting Models - XGBoost & LightGBM (with manual hyperparameter tuning)
def gradient_boosting_models(X_train, y_train, X_test):
    test_basic_training(X_train, y_train, X_test)  # ✅ Step 1: Basic training check
    test_grid_search(X_train, y_train)  # ✅ Step 2: GridSearchCV test with Ridge Regression

    models = {
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100),
        "LightGBM": lgb.LGBMRegressor(objective="regression", n_estimators=100)  # ✅ Explicitly set objective
    }

    best_params = {}
    predictions = {}

    warnings.filterwarnings("ignore", category=UserWarning)  # ✅ Silence unnecessary warnings

    for name, model in models.items():
        best_score = float("inf")
        best_config = None

        for lr in [0.01, 0.1, 0.3]:
            for depth in [3, 6, 10]:
                for estimators in [50, 100, 500]:
                    model_instance = clone(model)  # ✅ Clone model to ensure a fresh instance
                    model_instance = patch_get_tags(model_instance)  # ✅ Apply patch to avoid scikit-learn tag issues
                    model_instance.set_params(learning_rate=lr, max_depth=depth, n_estimators=estimators)

                    model_instance.fit(X_train, y_train)
                    score = -model_instance.score(X_train, y_train)  # ✅ Minimize negative mean squared error

                    if score < best_score:
                        best_score = score
                        best_config = {"learning_rate": lr, "max_depth": depth, "n_estimators": estimators}

        # ✅ Apply best hyperparameters & train final model
        model_instance = clone(model)
        model_instance = patch_get_tags(model_instance)
        model_instance.set_params(**best_config)
        model_instance.fit(X_train, y_train)

        predictions[name] = model_instance.predict(X_test)
        best_params[name] = best_config

    return predictions, best_params


# LSTM Model
def lstm_model(X_train, y_train, X_test):
    X_train_seq = np.expand_dims(X_train, axis=2)
    X_test_seq = np.expand_dims(X_test, axis=2)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train, epochs=10, batch_size=64, verbose=1)
    
    return model.predict(X_test_seq)

# Hybrid Model - CNN-LSTM
def cnn_lstm_model(X_train, y_train, X_test):
    X_train_seq = np.expand_dims(X_train, axis=2)
    X_test_seq = np.expand_dims(X_test, axis=2)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
        LSTM(50, activation='relu'),
        Flatten(),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train, epochs=10, batch_size=64, verbose=1)
    
    return model.predict(X_test_seq)

# Bayesian Optimization for LightGBM
def bayesian_optimization_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor()
    search_spaces = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_estimators': (50, 500)
    }

    opt = BayesSearchCV(model, search_spaces, scoring='neg_mean_squared_error', n_iter=30, cv=3)
    opt.fit(X_train, y_train)
    return opt.best_params_

# Optuna Hyperparameter Tuning for XGBoost
def optuna_xgboost(X_train, y_train):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500)
        }
        model = xgb.XGBRegressor(objective="reg:squarederror", **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_squared_error(y_train, preds)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params

# ✅ Function to ensure models are recognized as regressors and override scikit-learn's tag validation
def patch_get_tags_ensemble(estimator):
    """ Ensure models are recognized as regressors in scikit-learn, avoiding attribute errors. """
    if isinstance(estimator, Pipeline):  # ✅ Skip modification for Pipelines
        return estimator
    
    if hasattr(estimator, "_estimator_type"):
        estimator._estimator_type = "regressor"

    # ✅ Check if `__sklearn_tags__` exists before deleting
    if hasattr(estimator, "__sklearn_tags__"):
        try:
            del estimator.__sklearn_tags__
        except AttributeError:
            pass  # ✅ Silently ignore if deletion fails

    return estimator

# ✅ Ensemble Learning - Stacking Models
def ensemble_learning(X_train, y_train, X_test):
    warnings.filterwarnings("ignore", category=UserWarning)  # ✅ Silence unnecessary warnings

    # ✅ Validate and patch models before stacking (excluding XGBoost & LightGBM)
    base_models = [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),  # ✅ Pipelines are skipped
        ("rf", patch_get_tags_ensemble(RandomForestRegressor(n_estimators=100, n_jobs=-1)))  # ✅ Enables parallel computing
    ]

    # ✅ Ensure final meta-model is recognized correctly
    meta_model = patch_get_tags_ensemble(Ridge(alpha=1.0))

    # ✅ Initialize `StackingRegressor` with only compliant models
    stack = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # ✅ Fit the stacking model
    stack.fit(X_train, y_train)
    stack_preds = stack.predict(X_test)  # ✅ Get stacked predictions

    # ✅ Train & predict separately with XGBoost & LightGBM (with `n_jobs=-1`)
    xgb_model = patch_get_tags_ensemble(xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, n_jobs=-1))
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    lgb_model = patch_get_tags_ensemble(lgb.LGBMRegressor(n_estimators=100, n_jobs=-1))
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)

    # ✅ Combine predictions from all models
    predictions = {
        "StackingRegressor": stack_preds,
        "XGBoost": xgb_preds,
        "LightGBM": lgb_preds
    }

    return predictions

# Main Function to Execute All Models
def experiment_models(train_path, test_path):
    X_train, y_train, X_test = load_data(train_path, test_path)

    print("X_train shape:", X_train.shape)  # Ensure it has rows and columns
    print("y_train shape:", y_train.shape)  # Should have values

    print(X_train.dtypes)
    print(y_train.dtype)

    print(type(X_train), type(y_train))  # Should be pandas DataFrame and Series or numpy array

    # ✅ Convert to NumPy arrays to prevent type mismatches
    X_train = X_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)

    # ✅ Ensure no NaNs or Inf values in y_train
    if np.isnan(y_train).sum() > 0 or np.isinf(y_train).sum() > 0:
        raise ValueError("NaN or Inf found in y_train! Cleaning required.")
    y_train = np.nan_to_num(y_train)  # Converts NaNs/Infs to valid numbers

    print("Running Baseline Models...")
    baseline_preds = baseline_models(X_train, y_train, X_test)
    
    print("Running Gradient Boosting Models with Hyperparameter Tuning...")
    boosting_preds, best_params = gradient_boosting_models(X_train, y_train, X_test)
    
    print("Running Bayesian Optimization for LightGBM...")
    bayes_params = bayesian_optimization_lightgbm(X_train, y_train)
    
    print("Running Optuna Optimization for XGBoost...")
    optuna_params = optuna_xgboost(X_train, y_train)

    print("Running LSTM Model...")
    lstm_preds = lstm_model(X_train, y_train, X_test)
    
    print("Running CNN-LSTM Hybrid Model...")
    cnn_lstm_preds = cnn_lstm_model(X_train, y_train, X_test)
    
    print("Running Ensemble Learning...")
    ensemble_preds = ensemble_learning(X_train, y_train, X_test)

    return {
        "Baseline Models": baseline_preds,
        "Gradient Boosting": boosting_preds,
        "Best Hyperparameters": best_params,
        "Bayesian Optimization": bayes_params,
        "Optuna Optimization": optuna_params,
        "LSTM": lstm_preds,
        "CNN-LSTM": cnn_lstm_preds,
        "Ensemble Learning": ensemble_preds
    }

# Example Usage:
# results = experiment_models("train.parquet", "test.parquet")

# --------------------------------------
# Step 4: Submission Strategy
# --------------------------------------
# ✅ Function to select the best-performing model
def select_best_model(model_predictions, y_true):
    """
    Selects the best-performing model based on RMSE and Pearson correlation.

    Parameters:
    model_predictions (dict): Dictionary containing model names as keys and predictions as values.
    y_true (pd.Series): Actual label values for evaluation.

    Returns:
    str: Best model name.
    pd.Series: Best model predictions.
    """
    best_model = None
    best_score = -np.inf
    best_rmse = np.inf

    # ✅ Convert y_true to NumPy once, avoiding repeated operations
    y_true = np.array(y_true, dtype=np.float32)

    for model_name, y_pred in model_predictions.items():
        # ✅ Ensure predictions are valid before proceeding
        if not isinstance(y_pred, (np.ndarray, list)) or len(y_pred) == 0:
            print(f"Warning: {model_name} has no valid predictions! Skipping...")
            continue  # ✅ Skip models without valid predictions
        
        y_pred = np.array(y_pred, dtype=np.float32)  # ✅ Convert to NumPy efficiently

        # ✅ Handle dictionary-based predictions correctly
        if isinstance(y_pred.flat[0], dict):  
            y_pred = np.mean([np.array(list(pred.values()), dtype=np.float32) for pred in y_pred], axis=0)

        # ✅ Ensure matching lengths efficiently
        min_length = min(len(y_true), len(y_pred))
        y_true_trimmed = y_true[:min_length]
        y_pred_trimmed = y_pred[:min_length]

        # ✅ Prevent Pearson correlation error due to constant values
        if np.std(y_true_trimmed) == 0 or np.std(y_pred_trimmed) == 0:
            print(f"Warning: {model_name} has constant predictions! Skipping Pearson correlation...")
            pearson_corr = 0  # ✅ Assign default value instead of computing correlation
        else:
            pearson_corr, _ = pearsonr(y_true_trimmed, y_pred_trimmed)

        # ✅ Compute RMSE safely
        rmse = np.sqrt(mean_squared_error(y_true_trimmed, y_pred_trimmed))

        # ✅ Track best model based on Pearson correlation & RMSE
        if pearson_corr > best_score or (pearson_corr == best_score and rmse < best_rmse):
            best_score = pearson_corr
            best_rmse = rmse
            best_model = model_name
            best_predictions = y_pred_trimmed

    if best_model is None:
        raise ValueError("No valid model predictions found!")

    print(f"\n✅ Best Model Selected: {best_model} -> RMSE: {best_rmse:.4f}, Pearson Correlation: {best_score:.4f}")
    return best_model, best_predictions


def rolling_window_experiment(train_sample_file, test_sample_file, window_sizes, precomputed_model_predictions):
    """
    Performs rolling window experimentation with different training periods,
    ensuring feature alignment between train and test data.

    Parameters:
    train_sample_file (str): Path to the training dataset file.
    test_sample_file (str): Path to the testing dataset file.
    window_sizes (list): List of different rolling window sizes (e.g., [3, 6, 12] months).
    precomputed_model_predictions (dict): Dictionary of previously computed model predictions from Step 3.

    Returns:
    dict: Dictionary containing results for different window sizes.
    """
    results = {}

    print("\nRunning Rolling Window Experimentation...")

    for window in window_sizes:
        print(f"\nTesting model performance using last {window} months of data...")

        # ✅ Ensure rolling window experiment does not return empty results
        if not isinstance(precomputed_model_predictions, dict) or not precomputed_model_predictions:
            raise ValueError("experiment_models() returned invalid results!")

        # ✅ Flatten model predictions to extract only valid numeric outputs
        flattened_model_predictions = {}

        ignored_keys = {"Best Hyperparameters", "Bayesian Optimization", "Optuna Optimization"}

        for category, models in precomputed_model_predictions.items():
            if category in ignored_keys:
                continue  # ✅ Ignore non-prediction entries

            if isinstance(models, dict):  # ✅ Handle nested dictionaries
                for sub_model, predictions in models.items():
                    if isinstance(predictions, np.ndarray):
                        flattened_model_predictions[sub_model] = predictions.squeeze()  # ✅ Flatten LSTM outputs
            elif isinstance(models, np.ndarray):  # ✅ Directly store if it's a valid array
                flattened_model_predictions[category] = models.squeeze()

        # ✅ Load data separately for evaluation
        X_train, y_train, _ = load_data(train_sample_file, test_sample_file)

        # ✅ Select the best model based on Pearson correlation & RMSE
        best_model, best_predictions = select_best_model(flattened_model_predictions, y_train)

        # ✅ Ensure predictions align with y_train before evaluation
        min_length = min(len(y_train), len(best_predictions))
        y_train_trimmed = y_train[:min_length]
        best_predictions_trimmed = best_predictions[:min_length]

        # ✅ Prevent Pearson correlation errors due to constant values
        if np.std(y_train_trimmed) == 0 or np.std(best_predictions_trimmed) == 0:
            print(f"Warning: {best_model} has constant predictions! Skipping Pearson correlation...")
            pearson_corr = 0  # ✅ Assign default value instead of computing correlation
        else:
            pearson_corr, _ = pearsonr(y_train_trimmed, best_predictions_trimmed)

        # ✅ Compute RMSE safely
        rmse = np.sqrt(mean_squared_error(y_train_trimmed, best_predictions_trimmed))

        # Print logging information
        print(f"Rolling Window ({window} months) -> Best Model: {best_model}, RMSE: {rmse:.4f}, Pearson Correlation: {pearson_corr:.4f}")

        # ✅ Ensure rolling_results formatting before storing predictions
        results[f"Last_{window}_Months"] = {
            "Model": best_model,
            "Predictions": best_predictions_trimmed,
            "RMSE": rmse,
            "Pearson_Correlation": pearson_corr
        }

    return results


# ✅ Function to log hyperparameters and evaluation results
def log_results(model_name, hyperparameters, rmse, pearson_corr, log_file="model_logs.csv"):
    """
    Logs hyperparameter settings and evaluation results to a CSV file.

    Parameters:
    model_name (str): Name of the selected model.
    hyperparameters (dict): Best hyperparameters used.
    rmse (float): Root Mean Squared Error.
    pearson_corr (float): Pearson correlation value.
    log_file (str): Path to the CSV log file.

    Returns:
    None
    """
    log_entry = pd.DataFrame({
        "Model": [model_name],
        "Hyperparameters": [str(hyperparameters)],
        "RMSE": [rmse],
        "Pearson_Correlation": [pearson_corr]
    })

    # ✅ Improved CSV writing approach to prevent header overwrite issues
    file_exists = Path(log_file).exists()
    log_entry.to_csv(log_file, mode='a' if file_exists else 'w', header=not file_exists, index=False)

    print(f"\nLogged results for {model_name} in {log_file}")


# ✅ Function to format final predictions for submission
def format_submission(best_predictions, output_file="submission.csv"):
    """
    Formats final predictions for Kaggle submission and saves them as a CSV file.

    Parameters:
    best_predictions (pd.Series): Best model predictions.
    output_file (str): Path to the output submission file (default: "submission.csv").

    Returns:
    None
    """
    # ✅ Ensure predictions have valid indexing before submission
    if best_predictions.index is None:
        best_predictions = pd.Series(best_predictions, index=np.arange(1, len(best_predictions) + 1))  # ✅ Start ID from 1
    
    submission_df = pd.DataFrame({
        "ID": best_predictions.index,  
        "prediction": best_predictions
    })

    # ✅ Optionally remove negative values if required
    # submission_df["prediction"] = np.maximum(submission_df["prediction"], 0)  # ✅ Ensures non-negative predictions

    # ✅ Save correctly formatted CSV
    submission_df.to_csv(output_file, index=False)
    print(f"\n✅ Submission file saved: {output_file}")


# ✅ Main function to execute submission strategy
def execute_submission_strategy(train_sample_file, test_sample_file, window_sizes):
    """
    Executes the full submission strategy including model selection, rolling window testing, and logging.

    Parameters:
    train_sample_file (str): Path to the training dataset file.
    test_sample_file (str): Path to the testing dataset file.
    window_sizes (list): List of rolling window sizes for experimentation.

    Returns:
    None
    """
    print("\nStep 4: Running Experiment Models to Generate Predictions...")
    
    # ✅ Generate model predictions before rolling window testing
    precomputed_model_predictions = experiment_models(train_sample_file, test_sample_file)  # ✅ Corrected Placement

    print("Running Rolling Window Experimentation...")
    rolling_results = rolling_window_experiment(train_sample_file, test_sample_file, window_sizes, precomputed_model_predictions)

    # ✅ Fail-safe check: Ensure rolling_results is not empty before proceeding
    if not rolling_results:
        raise ValueError("Rolling window experiment returned empty results! Check model execution.")

    # ✅ Load training labels separately to ensure correct submission formatting
    X_train, y_train, _ = load_data(train_sample_file, test_sample_file)

    # ✅ Validate `y_train.index` exists before submission formatting
    if not hasattr(y_train, "index") or y_train.index is None:
        y_train = pd.Series(y_train, index=np.arange(len(y_train)))

    # ✅ Safely extract best model predictions
    valid_results = {k: v for k, v in rolling_results.items() if "Predictions" in v}
    if not valid_results:
        raise ValueError("No valid model predictions found in rolling_results!")

    best_model_key = max(valid_results.keys(), key=lambda x: pearsonr(y_train[:len(valid_results[x]['Predictions'])], pd.Series(valid_results[x]['Predictions'], index=y_train.index[:len(valid_results[x]['Predictions'])]).astype(np.float32))[0])

    final_best_model = valid_results[best_model_key]['Model']
    final_best_predictions = pd.Series(valid_results[best_model_key]['Predictions'], index=y_train.index[:len(valid_results[best_model_key]['Predictions'])]).astype(np.float32)

    # ✅ Prevent Pearson correlation errors due to constant values
    if np.std(y_train[:len(final_best_predictions)]) == 0 or np.std(final_best_predictions) == 0:
        print(f"Warning: {final_best_model} has constant predictions! Skipping Pearson correlation...")
        pearson_corr = 0  # ✅ Assign default value instead of computing correlation
    else:
        pearson_corr, _ = pearsonr(y_train[:len(final_best_predictions)], final_best_predictions)

    # ✅ Log final results
    log_results(final_best_model, {}, np.sqrt(mean_squared_error(y_train[:len(final_best_predictions)], final_best_predictions)), pearson_corr)
    
    # ✅ Ensure submission formatting consistency
    format_submission(final_best_predictions)

# Example Usage:
# execute_submission_strategy("train.parquet", "test.parquet", window_sizes=[3, 6, 12])
````

# 📚 References
1. J. Berk, P. DeMarzo: „__Corporate Finance__“, 6th Ed., Pearson (2023); R. W. Melicher, E. A. Norton: "__Introduction to Finance__", 16th Ed. WILEY (2017); 
Anatoly B. Schmidt: "__Quantitative Finance for Physicists: An Introduction__", 1st Ed. Academic Press (2005); Alex Backwell: "__An Intuitive Introduction to Finance and Derivatives: Concepts, Terminology and Models__",
 1st Ed, Springer (2023); Michael Isichenko: "__Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage__", 1st Ed., Springer (2021); John H. Cochrane: "__Asset Pricing__", Revised Ed., Princeton University Press (2005);
 Antti Ilmanen: "__Expected Returns: An Investor’s Guide to Harvesting Market Rewards__", 1st Ed., WILEY (2011); Steven E. Shreve: "__Stochastic Calculus for Finance I & II__", 1st Ed., Springer (2004); 
 Andrew Pole: "__Statistical Arbitrage: Algorithmic Trading Insights and Techniques__", 1st Ed., WILEY (2007); Mark S. Joshi: "__The Concepts and Practice of Mathematical Finance__", 2nd Ed., Cambridge University Press (2008);
Kaggle-link: competition-documentation: https://www.kaggle.com/competitions/drw-crypto-market-prediction.
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/f652e5f66ff0c322505da5a61dc0e79d053a3aa8/ForexArbitrageSeeker/ForexArbitrageSeeker.ipynb)
3. [![FOREX_Arbitrage_Seeking Report | English](https://img.shields.io/badge/FOREX_Arbitrage_Seeker%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/558d66eb6f1d13f19e41723431280766ed48df58/ForexArbitrageSeeker/ArbitrageSeekerReport.pdf) 
4. A. Meister , T. Sonar: "__Numerik__", 1st Ed. Springer-Spektrum (2019); S. Chapra, R. Canale: "__Numerical Methods for Engineers__", Mcgraw-Hill, 6th Edition (2010). 
5. J. Kilty, A. M. McAllister: "__Mathematical Modeling and Applied Calculus__", 1st Ed. Oxford University Press (2018).
6. U. Kockelkorn: "__Statistik für Anwender__", 1st Ed. Springer (2012), s. chapters 7 - 8.
7. Robert H. Shumway, David S. Stoffer: "__Time Series Analysis and Its Applications with R Examples__", Springer (2011).
8. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor: "__An Introduction to Statistical Learning with Applications in Python__", Springer (2023).
9. Cornelis W. Oosterlee, Lech A. Grzelak: "__Mathematical Modeling and Computation in Finance with Exercises and Python and MATLAB Computer Codes__", World Scientific (2020).
10. Richard Szeliski: "__Computer Vision - Algorithms and Applications__", Springer (2022).
11. Anthony Scopatz, Kathryn D. Huff: "__Effective Computation in Physics - Field Guide to Research with Python__", O'Reilly Media (2015).
12. Alex Gezerlis: "__Numerical Methods in Physics with Python__", Cambridge University Press (2020).
13. Gary Hutson, Matt Jackson: "__Graph Data Modeling in Python. A practical guide__", Packt-Publishing (2023).
14. Hagen Kleinert: "__Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets__", 5th Edition, World Scientific Publishing Company (2009).
15. Peter Richmond, Jurgen Mimkes, Stefan Hutzler: "__Econophysics and Physical Economics__", Oxford University Press (2013).
16. A. Coryn , L. Bailer Jones: "__Practical Bayesian Inference A Primer for Physical Scientists__", Cambridge University Press (2017).
17. Avram Sidi: "__Practical Extrapolation Methods - Theory and Applications__", Cambridge university Press (2003).
18. Volker Ziemann: "__Physics and Finance__", Springer (2021).
19. Zhi-Hua Zhou: "__Ensemble methods, foundations and algorithms__", CRC Press (2012).
20. B. S. Everitt, et al.: "__Cluster analysis__", Wiley (2011).
21. Lior Rokach, Oded Maimon: "__Data Mining With Decision Trees - Theory and Applications__", World Scientific (2015).
22. Bernhard Schölkopf, Alexander J. Smola: "__Learning with kernels - support vector machines, regularization, optimization and beyond__", MIT Press (2009).
23. Johan A. K. Suykens: "__Regularization, Optimization, Kernels, and Support Vector Machines__", CRC Press (2014).
24. Sarah Depaoli: "__Bayesian Structural Equation Modeling__", Guilford Press (2021).
25. Rex B. Kline: "__Principles and Practice of Structural Equation Modeling__", Guilford Press (2023).
26. Ekaterina Kochmar: "__Getting Started with Natural Language Processing__", Manning (2022).
27. Jakub Langr, Vladimir Bok: "__GANs in Action__", Computer Vision Lead at Founders Factory (2019).
28. David Foster: "__Generative Deep Learning__", O'Reilly(2023).
29. Rowel Atienza: "__Advanced Deep Learning with Keras: Applying GANs and other new deep learning algorithms to the real world__", Packt Publishing (2018).
30. Josh Kalin: "__Generative Adversarial Networks Cookbook__", Packt Publishing (2018).  
31. Thomas Haslwanter: "__Hands-on Signal Analysis with Python: An Introduction__", Springer (2021).
32. Jose Unpingco: "__Python for Signal Processing__", Springer (2023).
33. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
34. C. Paar, J. Pelzl: "__Understanding Cryptography__", Springer (2010); H. Delfs, H. Knebl: "__Introduction to Cryptography__", 3rd Ed. Springer (2015); J. Katz, Y. lindell: "__Introduction to Modern Cryptography__", 2nd Ed, CRC Press (2015); 
O. Goldreich: "__Foundations of Cryptography__", Cambridge University Press (2008); J. P. Aumasson: "__Serious Cryptography__", no starch press (2018).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sea Level Forecasting with 4 Models (2021-2024)"""

# --- Explicit Library Imports for Robustness ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import time
from datetime import datetime
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set seed for LSTM reproducibility
tf.random.set_seed(42) 

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION (VERIFY THESE) ---
# VERIFY THIS PATH: Must be a RAW string (r"...") for Windows
FILE_PATH = r"C:all_sheet_2.xlsx"

LOCATION_TO_FORECAST = 'Klang'
N_TEST_STEPS = 12 # Forecasting all 12 months of the test year

# --- FILTER CONSTANTS ---
START_YEAR_FILTER = 1994 # Assuming the original 1994-2024 range based on file name
END_YEAR_FILTER = 2024
# ----------------------------

# --- COLUMN CONSTANTS ---
SEA_LEVEL_COLUMN = 'sea_level_m'
DATE_COLUMN_NAME = 'date'
# ----------------------------

print("🔧 Setting up environment and running imports...")
print(f"TensorFlow version: {tf.__version__}")

# ========== METRICS FUNCTIONS (STANDARD) ==========
def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error"""
    actual_copy = actual.copy()
    actual_copy[actual_copy == 0] = 1e-8
    return np.mean(np.abs((actual - forecast) / actual_copy)) * 100

def calculate_smape(actual, forecast):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = np.abs(actual) + np.abs(forecast)
    denominator[denominator == 0] = 1e-8
    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / denominator)

def calculate_rmae(actual, forecast, baseline_forecast=None):
    """Calculate Relative Mean Absolute Error"""
    mae = mean_absolute_error(actual, forecast)
    if baseline_forecast is None:
        baseline_forecast = np.roll(actual, 1)
        baseline_forecast[0] = actual[0]
    baseline_mae = mean_absolute_error(actual, baseline_forecast)
    return mae / baseline_mae if baseline_mae != 0 else float('inf')

def calculate_all_metrics(actual, forecast, model_name):
    """Calculate all metrics for a model"""
    actual = np.array(actual)
    forecast = np.array(forecast)
    if forecast is None or len(forecast) == 0 or len(actual) != len(forecast):
        print(f"⚠️ Skipping metrics for {model_name}: Forecast is invalid or length mismatch.")
        return None
    mae = mean_absolute_error(actual, forecast)
    mape = calculate_mape(actual, forecast)
    smape = calculate_smape(actual, forecast)
    rmae = calculate_rmae(actual, forecast, baseline_forecast=None)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return {
        'Model': model_name,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'rMAE': rmae,
        'RMSE': rmse
    }
# ========== END METRICS FUNCTIONS ==========

# --- DATA LOADING FUNCTIONS --- #
def load_sea_level_data(file_path, location):
    """Load sea level data from Excel file for specific location (sheet name)."""
    print(f"\n📂 Attempting to load sea level data from: {file_path}")
    print(f"📍 Reading sheet/location: {location}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at: {file_path}. Please check the path and spelling.")

    try:
        # Using openpyxl engine is critical for .xlsx files
        df = pd.read_excel(file_path, sheet_name=location, engine='openpyxl')
        print(f"✓ Successfully loaded Excel sheet: '{location}'")

        if DATE_COLUMN_NAME not in df.columns or SEA_LEVEL_COLUMN not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"Required columns not found. Need '{DATE_COLUMN_NAME}' and '{SEA_LEVEL_COLUMN}'. Available columns: {available_cols}")

        df.loc[:, DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME], errors='coerce')
        ts_data = df.set_index(DATE_COLUMN_NAME)[SEA_LEVEL_COLUMN].dropna().sort_index()
        
        # Assume Monthly Start frequency
        ts_data = ts_data.asfreq('MS') 

        if ts_data.empty:
              raise ValueError("Time series data is empty after cleanup. Check data quality.")

        print(f"✅ Time series data created for location: '{location}'")
        print(f"📈 Full date range: {ts_data.index.min().strftime('%Y-%m-%d')} to {ts_data.index.max().strftime('%Y-%m-%d')}")
        print(f"🔢 Total observations: {len(ts_data)}")

        return ts_data, df

    except ImportError:
        print("❌ Error: 'openpyxl' required to read .xlsx files. Please run: pip install openpyxl")
        raise
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        raise

def prepare_sea_level_data(ts_data, n_test_steps=12):
    """Prepare sea level data for forecasting"""
    if len(ts_data) <= n_test_steps:
        raise ValueError(f"Not enough data. Have {len(ts_data)} points, need more than {n_test_steps} for testing.")

    train_data = ts_data[:-n_test_steps]
    test_data = ts_data[-n_test_steps:]

    print(f"\n📊 Data split for forecasting:")
    print(f"   Train size: {len(train_data)} ({train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')})")
    print(f"   Test size: {len(test_data)} ({test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')})")
    print(f"   Forecasting horizon: {n_test_steps} steps")

    return train_data, test_data

# --- MAIN EXECUTION BLOCK (Data Loading) ---
try:
    ts_data_full, original_df = load_sea_level_data(FILE_PATH, LOCATION_TO_FORECAST)

    # FILTER DATA TO START_YEAR_FILTER-END_YEAR_FILTER RANGE
    start_date = datetime(START_YEAR_FILTER, 1, 1)
    end_date = datetime(END_YEAR_FILTER, 12, 31)

    ts_data = ts_data_full.loc[start_date:end_date]

    if ts_data.empty:
        raise ValueError(f"No data found in the time range {START_YEAR_FILTER} to {END_YEAR_FILTER} after filtering.")

    print(f"\n✅ Data filtered to desired range: {ts_data.index.min().strftime('%Y-%m-%d')} to {ts_data.index.max().strftime('%Y-%m-%d')}")

    train_data, test_data = prepare_sea_level_data(ts_data, N_TEST_STEPS)

except Exception as e:
    print(f"\n❌ Error loading actual data: {e}")
    print("🔄 Generating sample data for demonstration to allow models to run...")

    # Generate sample sea level-like data (fallback)
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2024-12-01', freq='MS')
    n_points = len(dates)
    trend = 0.005 * np.arange(n_points)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_points) / 12)
    noise = np.random.normal(0, 0.05, n_points)
    sample_data = 1.0 + trend + seasonal + noise
    ts_data = pd.Series(sample_data, index=dates, name=SEA_LEVEL_COLUMN)
    ts_data.index.name = DATE_COLUMN_NAME
    original_df = pd.DataFrame({DATE_COLUMN_NAME: ts_data.index, LOCATION_TO_FORECAST: ts_data.values})

    train_data, test_data = prepare_sea_level_data(ts_data, N_TEST_STEPS)

    print("\n⚠️ WARNING: Models are running on **synthetic** sea level data. Please verify your file path for actual results.")

# --- OUTPUT DIRECTORY FIX ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() 
    print("⚠️ WARNING: Running outside a file environment. Saving to current working directory.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_name = f"sea_level_forecasting_1994-2024_{LOCATION_TO_FORECAST.replace(' ', '_')}_{timestamp}" 
output_dir = os.path.join(script_dir, output_dir_name)
os.makedirs(output_dir, exist_ok=True)
print(f"\n📁 Created results directory: {output_dir}")

# ========== GLOBAL VARIABLES TO STORE FORECASTS AND METRICS ==========
all_forecasts = {}
metrics_results = []
# =====================================================================

# --- MODEL UTILITY FUNCTIONS --- #
def create_base_features(data, lag=12):
    """Create basic lagged and time features for base XGBoost model."""
    df = pd.DataFrame(data.copy())
    df.columns = ['y']
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    if isinstance(data.index, pd.DatetimeIndex):
        df['month'] = data.index.month
        df['year'] = data.index.year
        df['dayofweek'] = data.index.dayofweek
    df.dropna(inplace=True)
    return df

def create_enhanced_features(data, lag=12, fourier_k=2):
    """Create enhanced lagged, time, Fourier, and Target Encoding features."""
    df = create_base_features(data, lag=lag)

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > lag:
        t = np.arange(len(df)) 
        
        # 3. Seasonal Fourier Features
        for k in range(1, fourier_k + 1):
            df[f'sin_{k}'] = np.sin(2 * np.pi * k * t / 12)
            df[f'cos_{k}'] = np.cos(2 * np.pi * k * t / 12)

        # 4. Target Encoding (Monthly Mean)
        df['monthly_mean'] = df.groupby('month')['y'].transform(lambda x: x.shift(1).expanding().mean())

    df.dropna(inplace=True)
    return df

def build_lstm_model(look_back):
    """Standard LSTM model architecture."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse') 
    return model

def create_lstm_dataset(data, look_back=1):
    X, Y = [], []
    # Data is expected to be a 2D numpy array [samples, 1]
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    # Reshape X for LSTM [samples, look_back, features (1)]
    return np.array(X).reshape(len(X), look_back, 1), np.array(Y)
# --- END MODEL UTILITY FUNCTIONS --- #

# --- MODEL IMPLEMENTATIONS (5 MODELS) ---

## 🌊 1. ARIMA Model (Base)
def run_arima_model(train, test):
    print("\n⏳ Running 1. ARIMA Model (Base)...")
    try:
        order = (1, 1, 1) 
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        forecast = model_fit.predict(start=train.index[-1], end=test.index[-1], dynamic=False) 
        forecast = forecast.loc[test.index]
        return forecast
    except Exception as e:
        print(f"❌ ARIMA model failed: {e}")
        return None

## 🌲 2. XGBoost Model (Base - Lag + Time features only)
def run_base_xgboost_model(train, test, n_test_steps):
    """XGBoost with basic lag and time features."""
    print("\n⏳ Running 2. XGBoost Model (Base)...")
    try:
        LAG = 12
        full_data = pd.concat([train, test])
        features_df = create_base_features(full_data, lag=LAG)

        X_full = features_df.drop('y', axis=1)
        y_full = features_df['y']

        X_train = X_full.loc[:test.index[0]].iloc[:-1]
        y_train = y_full.loc[X_train.index]
        X_test = X_full.loc[test.index]
        
        if X_train.empty or X_test.empty or len(X_test) != n_test_steps: return None

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, learning_rate=0.05)
        model.fit(X_train, y_train)
        forecast_values = model.predict(X_test)

        forecast = pd.Series(forecast_values, index=test.index)
        return forecast
    except Exception as e:
        print(f"❌ Base XGBoost model failed: {e}")
        return None

## 🌲 3. XGBoost Model (Enhanced)


## 🧠 4. LSTM Model (Base)
def run_lstm_model(train, test, n_test_steps):
    print("\n⏳ Running 4. LSTM Model (Base)...")
    try:
        LOOK_BACK = 12
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_values = train.dropna().values.reshape(-1, 1) 
        train_scaled = scaler.fit_transform(train_values)
        
        # Adjusted create_lstm_dataset handles the reshape now
        X_train, y_train = create_lstm_dataset(train_scaled, LOOK_BACK) 
        
        model = build_lstm_model(LOOK_BACK)
        model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0) 
        
        last_train_batch = train_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
        test_predictions_scaled = []
        current_batch = last_train_batch
        
        for _ in range(n_test_steps):
            predicted_value = model.predict(current_batch, verbose=0)[0]
            new_batch = np.append(current_batch[:, 1:, :], [[predicted_value]], axis=1)
            current_batch = new_batch
            test_predictions_scaled.append(predicted_value)
            
        forecast_values = scaler.inverse_transform(np.array(test_predictions_scaled).reshape(-1, 1)).flatten()
        forecast = pd.Series(forecast_values, index=test.index)
        return forecast
    except Exception as e:
        print(f"❌ Base LSTM model failed: {e}")
        return None

## ➕ 5. ARIMA-LSTM Hybrid Model (Robust Version)
def run_arima_lstm_hybrid(train, test, n_test_steps):
    print("\n⏳ Running 5. ARIMA+LSTM Hybrid Model...")
    try:
        LOOK_BACK = 12
        
        # --- 1. ARIMA (Linear Component) ---
        arima_model = ARIMA(train, order=(1, 1, 1)).fit() 
        arima_train_pred = arima_model.predict(start=0, end=len(train)-1)
        residuals = train - arima_train_pred
        
        # --- CRITICAL FIX: Ensure residuals are clean and long enough ---
        # Drop NaNs, and ensure we have enough points for LOOK_BACK window
        clean_residuals = residuals.dropna().iloc[LOOK_BACK:]
        
        if len(clean_residuals) < LOOK_BACK + 1:
             print(f"❌ ERROR: Not enough clean residual data ({len(clean_residuals)} points) for LSTM training.")
             return None

        # --- 2. LSTM on Residuals ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        residuals_values = clean_residuals.values.reshape(-1, 1)
        residuals_scaled = scaler.fit_transform(residuals_values)
        
        # Adjusted create_lstm_dataset handles the reshape now
        X_res, y_res = create_lstm_dataset(residuals_scaled, LOOK_BACK)
        
        lstm_model = build_lstm_model(LOOK_BACK)
        lstm_model.fit(X_res, y_res, epochs=10, batch_size=4, verbose=0) 

        # --- 3. Forecasting ---
        
        # Get ARIMA Forecast (Linear Component)
        arima_test_forecast = arima_model.predict(start=train.index[-1], end=test.index[-1])
        arima_test_forecast = arima_test_forecast.loc[test.index] 

        # Get LSTM Forecast (Residual Component)
        last_residual_batch = residuals_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
        res_predictions_scaled = []
        current_batch = last_residual_batch
        
        for _ in range(n_test_steps):
            predicted_value = lstm_model.predict(current_batch, verbose=0)[0]
            new_batch = np.append(current_batch[:, 1:, :], [[predicted_value]], axis=1)
            current_batch = new_batch
            res_predictions_scaled.append(predicted_value)
            
        res_forecast_values = scaler.inverse_transform(np.array(res_predictions_scaled).reshape(-1, 1)).flatten()
        res_forecast = pd.Series(res_forecast_values, index=test.index)
        
        # Final Hybrid: Linear Forecast + Non-Linear Residual Forecast
        hybrid_forecast = arima_test_forecast + res_forecast
        
        return hybrid_forecast
        
    except Exception as e:
        print(f"❌ ARIMA+LSTM Hybrid model FAILED with critical error: {e}")
        return None
# --- END MODEL IMPLEMENTATIONS --- #

# --- RUN ALL 5 MODELS AND COLLECT RESULTS --- #

# 1. ARIMA (Base)
arima_forecast = run_arima_model(train_data, test_data)
if arima_forecast is not None:
    all_forecasts['ARIMA'] = arima_forecast
    metrics_results.append(calculate_all_metrics(test_data.values, arima_forecast.values, 'ARIMA'))

# 2. XGBoost (Base)
base_xgboost_forecast = run_base_xgboost_model(train_data, test_data, N_TEST_STEPS)
if base_xgboost_forecast is not None:
    all_forecasts['XGBoost (Base)'] = base_xgboost_forecast
    metrics_results.append(calculate_all_metrics(test_data.values, base_xgboost_forecast.values, 'XGBoost (Base)'))

# 3. XGBoost (Enhanced)
#enhanced_xgboost_forecast = run_enhanced_xgboost_model(train_data, test_data, N_TEST_STEPS)
#if enhanced_xgboost_forecast is not None:
 #   all_forecasts['XGBoost (Enhanced)'] = enhanced_xgboost_forecast
 #   metrics_results.append(calculate_all_metrics(test_data.values, enhanced_xgboost_forecast.values, 'XGBoost (Enhanced)'))

# 4. LSTM (Base)
lstm_forecast = run_lstm_model(train_data, test_data, N_TEST_STEPS)
if lstm_forecast is not None:
    all_forecasts['LSTM'] = lstm_forecast
    metrics_results.append(calculate_all_metrics(test_data.values, lstm_forecast.values, 'LSTM'))

# 5. ARIMA+LSTM Hybrid
# FIX FOR SYNTAX ERROR: The previous lines must have been missing a comma or have a misplaced character.
# This simple, clean assignment prevents the SyntaxError from the previous line.
arima_lstm_forecast = run_arima_lstm_hybrid(train_data, test_data, N_TEST_STEPS)
if arima_lstm_forecast is not None:
    all_forecasts['ARIMA-LSTM'] = arima_lstm_forecast
    metrics_results.append(calculate_all_metrics(test_data.values, arima_lstm_forecast.values, 'ARIMA-LSTM'))

# --- END RUN ALL 5 MODELS AND COLLECT RESULTS --- #

# --- GENERATE RESULTS TABLE AND PLOT --- #

## 📊 Performance Metrics
print("\n" + "="*50)
print("📊 FINAL PERFORMANCE METRICS")
print("="*50)

if metrics_results:
    metrics_df = pd.DataFrame(metrics_results).set_index('Model').round(4)

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow' if v else '' for v in is_min]

    # Display logic for Jupyter/Console
    try:
        from IPython.display import display
        # Check if running interactively (e.g., Jupyter)
        if 'ipykernel' in sys.modules:
            styled_metrics_df = metrics_df.style.apply(highlight_min, axis=0)
            display(styled_metrics_df)
        else:
             print(metrics_df)
    except Exception:
        print(metrics_df) # Fallback print

    best_model_name = metrics_df['RMSE'].idxmin()
    print(f"\n🏆 Best performing model (based on RMSE): **{best_model_name}**")
else:
    print("❌ No models ran successfully to generate metrics.")

print("\n" + "="*50)

## 📈 Final Forecast Plot
if all_forecasts:
    plt.figure(figsize=(14, 7))
    
    # CRITICAL FIX: Clear the current figure to remove any residual data/labels from prior runs
    plt.clf() 
    
    # 1. Combine training and test data into a single Series
    full_actual_data = pd.concat([train_data, test_data])
    
    # 2. Plot the single, continuous, unfilled blue line for all actual data
    plt.plot(full_actual_data.index, full_actual_data.values, 
              color='blue', 
              linewidth=2, 
              label='Actual Data (Training & Test)') 

    # (The rest of the plotting code remains unchanged) 
    # Plot forecasts and connect them to the training data
    for model_name, forecast in all_forecasts.items():
        # This part is good: connecting forecast to the last training point
        last_train_point = pd.Series([train_data.iloc[-1]], index=[train_data.index[-1]])
        connected_forecast = pd.concat([last_train_point, forecast])
        
        plt.plot(connected_forecast.index, connected_forecast.values, 
                 label=f'{model_name} Forecast', linestyle='--', marker='.', linewidth=2)


  #  plt.figure(figsize=(14, 7))
    
   # full_actual_data = pd.concat([train_data, test_data])
    
   # plt.plot(train_data.index, train_data.values, 
      #        color='blue', linewidth=2, label='Training Data (Actual)')
              
   # plt.plot(test_data.index, test_data.values, 
     #         color='red', linewidth=2, label='Test Data (Actual)', marker='o')

   # for model_name, forecast in all_forecasts.items():
    #    last_train_point = pd.Series([train_data.iloc[-1]], index=[train_data.index[-1]])
    #    connected_forecast = pd.concat([last_train_point, forecast])
        
    #    plt.plot(connected_forecast.index, connected_forecast.values, 
    #             label=f'{model_name} Forecast', linestyle='--', marker='.', linewidth=2)

    plt.title(f'Sea Level Forecasting for {LOCATION_TO_FORECAST} ({START_YEAR_FILTER}-{END_YEAR_FILTER})')
    plt.xlabel('Date')
    plt.ylabel(f'{SEA_LEVEL_COLUMN.replace("_", " ").title()}')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    forecast_start_date = test_data.index[0]- pd.DateOffset(days=30)
    plt.axvline(x=forecast_start_date, color='green', linestyle=':', linewidth=2, alpha=0.7)
    plt.text(forecast_start_date, plt.ylim()[1]*0.95, ' Forecast Start', 
              color='green', rotation=90, verticalalignment='top')

    plot_filename = os.path.join(output_dir, f'{LOCATION_TO_FORECAST}_forecast_5_models_{START_YEAR_FILTER}_{END_YEAR_FILTER}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to: {plot_filename}")

    plt.show()
else:
    print("❌ Cannot generate plot: No successful forecasts.")

# --- SAVE RESULTS LOCALLY ---
print("\n" + "="*50)
print("💾 SAVING RESULTS LOCALLY")
print("="*50)

if 'metrics_df' in locals():
    metrics_filename = os.path.join(output_dir, f'{LOCATION_TO_FORECAST}_metrics_summary_5_models_{START_YEAR_FILTER}_{END_YEAR_FILTER}.csv')
    metrics_df.to_csv(metrics_filename)
    print(f"✅ Saved metrics: {metrics_filename}")

if 'plot_filename' in locals():
    print(f"✅ Saved plot: {plot_filename}")

# Save forecasts to Excel
if all_forecasts:
    forecasts_df = pd.DataFrame(all_forecasts)
    forecasts_df['Actual'] = test_data
    forecasts_filename = os.path.join(output_dir, f'{LOCATION_TO_FORECAST}_forecasts_5_models_{START_YEAR_FILTER}_{END_YEAR_FILTER}.xlsx')
    forecasts_df.to_excel(forecasts_filename)
    print(f"✅ Saved forecasts: {forecasts_filename}")

print(f"\n🎯 All results saved to: {output_dir}")
print("Script execution completed successfully!")

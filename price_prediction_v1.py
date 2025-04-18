# -*- coding: utf-8 -*-
# price_prediction_v1.py
# Description: LSTM Predictor for Open/Close prices, modified for multi-stock handling.
#              Saves/Loads per-stock models and scalers. Returns scaler object.
#              Includes main block for initializing models for a stock list.
# Version: 1.1 (Multi-Stock Support + Return Scaler + Initializer Main Block)

import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
try:
    # Recommended for TF 2.x
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam # Use modern optimizer
except ImportError:
    try:
        # Fallback for standalone Keras or older TF
        from keras.models import Sequential, load_model
        from keras.layers import Dense, LSTM
        from keras.optimizers import Adam
    except ImportError:
        print("CRITICAL ERROR: Cannot import Keras or TensorFlow. Please ensure it is installed.")
        sys.exit(1) # Exit if essential libraries are missing
import warnings
import os
import datetime
from datetime import timedelta # Ensure timedelta is imported
import joblib # For saving/loading the scaler
import time
import sys
import json # Added for api_request error handling

# === Configuration ===
# --- Model/File Naming ---
MODEL_DIR = "lstm_models" # Directory to store models/scalers
# File path templates incorporating stock code
MODEL_FILENAME_TPL = os.path.join(MODEL_DIR, "lstm_predictor_{stock_code}.h5")
SCALER_FILENAME_TPL = os.path.join(MODEL_DIR, "scaler_{stock_code}.pkl")
TIMESTAMP_FILENAME_TPL = os.path.join(MODEL_DIR, "timestamp_{stock_code}.txt")

# --- Training Parameters ---
RETRAIN_INTERVAL_DAYS = 7   # How often to retrain each stock's model
TIMESTEPS = 7               # LSTM input sequence length
FEATURES = ['open', 'high', 'low', 'close'] # Features to use from OHLC data
TARGET_FEATURES = ['open', 'close']         # Features the LSTM predicts
USE_SAMPLE_WEIGHTING = True # Enable/disable time-based weighting
MIN_SAMPLE_WEIGHT = 0.5     # Weight for oldest sample
MAX_SAMPLE_WEIGHT = 1.5     # Weight for newest sample
EPOCHS = 50                 # Training epochs per retraining cycle
BATCH_SIZE = 32

# --- API Configuration ---
API_BASE_URL_STOCK = "http://140.116.86.242:8081" # API for stock data (:8081)
REQUEST_TIMEOUT = 25

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Export SCALER_FILENAME_TPL (Optional, stock_agent uses it to construct paths) ---
# This allows stock_agent to know the naming convention if needed elsewhere,
# although the primary way to get the scaler is now via the return value.
SCALER_FILENAME = SCALER_FILENAME_TPL

# === API Helper (Copied from stock_agent for consistency) ===
def api_request(base_url, method, endpoint, data=None, retry_attempts=2):
    """Generic function to handle API requests with retries."""
    url = f"{base_url}/{endpoint}"
    last_exception = None
    for attempt in range(retry_attempts + 1):
        try:
            if method.upper() == 'GET': response = requests.get(url, timeout=REQUEST_TIMEOUT)
            elif method.upper() == 'POST': response = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
            else: print(f"ERROR: Unsupported method {method}"); return None
            response.raise_for_status(); return response.json()
        except requests.exceptions.Timeout: msg = f"API Timeout (Attempt {attempt + 1})"; last_exception = TimeoutError(msg); print(f"WARN: {msg} URL: {url}")
        except requests.exceptions.HTTPError as http_err: msg = f"API HTTP Error: {http_err}"; last_exception = http_err; print(f"ERROR: {msg} URL: {url}"); break
        except requests.exceptions.RequestException as req_err: msg = f"API Request Failed: {req_err}"; last_exception = req_err; print(f"ERROR: {msg} URL: {url}")
        except json.JSONDecodeError as json_err: msg = f"Failed JSON decode: {json_err}"; last_exception = json_err; print(f"ERROR: {msg} URL: {url}"); break
        except Exception as e: msg = f"Unknown API Error: {e}"; last_exception = e; print(f"ERROR: {msg} URL: {url}")
        if attempt < retry_attempts: time.sleep(1 + attempt)
    print(f"ERROR: API request failed after {retry_attempts + 1} attempts. Last error: {last_exception} URL: {url}"); return None

def Get_Stock_Informations(stock_code, start_date_str, end_date_str):
    """Gets historical stock OHLC data for a specific stock code."""
    endpoint = f'stock/api/v1/api_get_stock_info_from_date_json/{stock_code}/{start_date_str}/{end_date_str}'
    result = api_request(API_BASE_URL_STOCK, 'GET', endpoint)
    if result and result.get('result') == 'success':
        return result.get('data', [])
    status = result.get('status', 'Unknown Error') if result else 'No Response/Error'
    # print(f"API Get_Stock_Informations failed for {stock_code}: {status}") # Reduce noise
    return []

# === LSTM Data Preparation and Model Building ===
def get_ohlc_data_for_stock(stock_code):
    """Fetches and preprocesses OHLC data for the specified stock."""
    print(f"LSTM ({stock_code}): Fetching historical OHLC data...")
    start_date = (datetime.datetime.now() - timedelta(days=730)).strftime('%Y%m%d') # Fetch ~2 years
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    stock_history_data_raw = Get_Stock_Informations(stock_code, start_date, end_date)

    if not stock_history_data_raw:
        print(f"ERROR ({stock_code}): Failed to fetch data or data is empty.")
        return None

    ohlc_data = []; dates = []
    for history in stock_history_data_raw:
         if all(key in history for key in FEATURES):
             try:
                 ohlc_data.append([float(history[key]) for key in FEATURES])
                 dates.append(history.get("Date"))
             except (ValueError, TypeError): continue # Skip invalid rows silently?
         # else: print(f"WARN ({stock_code}): Record missing features: {history.get('Date')}")

    if len(ohlc_data) < TIMESTEPS + 1:
         print(f"ERROR ({stock_code}): Insufficient valid OHLC data ({len(ohlc_data)} records). Need at least {TIMESTEPS + 1}.")
         return None

    ohlc_data = np.array(ohlc_data, dtype=np.float32)
    print(f"[OK] LSTM ({stock_code}): Processed {ohlc_data.shape[0]} days OHLC data. Last: {dates[-1] if dates else 'N/A'}")
    return ohlc_data

def prepare_lstm_data(scaled_data, target_indices):
    """Prepares sequences for LSTM training."""
    X_data, y_data = [], []
    num_features = scaled_data.shape[1]
    for i in range(TIMESTEPS, len(scaled_data)):
        X_data.append(scaled_data[i-TIMESTEPS:i, :]) # Input: previous TIMESTEPS OHLC
        y_data.append(scaled_data[i, target_indices]) # Target: next day Open & Close
    X_data, y_data = np.array(X_data), np.array(y_data)
    if X_data.shape[0] == 0: print("ERROR: Could not create training sequences."); return None, None
    return X_data, y_data

def build_lstm_model(input_shape, output_units):
    """Builds the LSTM model structure."""
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        LSTM(units=32),
        Dense(units=output_units) # Predict Open and Close
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    print("[OK] LSTM model structure built.")
    # model.summary()
    return model

# === Per-Stock Training and Saving ===
def train_and_save_model(stock_code, data):
    """Trains and saves LSTM model and scaler for a specific stock."""
    print(f"LSTM ({stock_code}): Starting training process...")
    model_path = MODEL_FILENAME_TPL.format(stock_code=stock_code)
    scaler_path = SCALER_FILENAME_TPL.format(stock_code=stock_code)
    timestamp_path = TIMESTAMP_FILENAME_TPL.format(stock_code=stock_code)

    # 1. Scale Data (Fit scaler ONLY on this stock's data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    try: scaled_data = scaler.fit_transform(data); print(f"   ({stock_code}): Data scaled.")
    except Exception as e: print(f"ERROR ({stock_code}): Failed scaling - {e}"); return None, None

    # 2. Prepare Sequences
    target_indices = [FEATURES.index(f) for f in TARGET_FEATURES]
    X_train, y_train = prepare_lstm_data(scaled_data, target_indices)
    if X_train is None: return None, None

    # 3. Calculate Sample Weights
    sample_weights = None
    if USE_SAMPLE_WEIGHTING:
        num_samples = X_train.shape[0]
        sample_weights = np.linspace(MIN_SAMPLE_WEIGHT, MAX_SAMPLE_WEIGHT, num_samples)

    # 4. Build Model
    num_features = data.shape[1]; num_target_features = len(TARGET_FEATURES)
    try: model = build_lstm_model(input_shape=(TIMESTEPS, num_features), output_units=num_target_features)
    except Exception as e: print(f"ERROR ({stock_code}): Failed building model - {e}"); return None, None

    # 5. Train Model
    print(f"   ({stock_code}): Training LSTM for {EPOCHS} epochs...")
    try:
        # Set verbose=1 or 2 to see training progress per epoch
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, sample_weight=sample_weights)
        print(f"   ({stock_code}): Training complete.")
    except Exception as e: print(f"ERROR ({stock_code}): Model training failed - {e}"); return None, None

    # 6. Save Model, Scaler, Timestamp
    try:
        model.save(model_path); joblib.dump(scaler, scaler_path)
        with open(timestamp_path, 'w') as f: f.write(datetime.datetime.now().isoformat())
        print(f"[OK] LSTM ({stock_code}): Model saved to {model_path}, Scaler to {scaler_path}")
        return model, scaler # Return trained model and scaler
    except Exception as e: print(f"ERROR ({stock_code}): Failed saving - {e}"); return None, None

# === Main Prediction Function (Modified for Multi-Stock & Returning Scaler) ===
def predict_next_open_close(stock_code, force_retrain=False):
    """
    Predicts next day's Open/Close for a specific stock_code.
    Handles per-stock loading/retraining and *returns the scaler used*.

    Args:
        stock_code (str): The 4-digit stock code.
        force_retrain (bool): If True, forces retraining.

    Returns:
        tuple: (predicted_open, predicted_close, scaler_object) or (None, None, None)
    """
    print(f"\n--- LSTM Prediction for Stock: {stock_code} ---")
    model = None; scaler = None; retrain_needed = False
    model_path = MODEL_FILENAME_TPL.format(stock_code=stock_code)
    scaler_path = SCALER_FILENAME_TPL.format(stock_code=stock_code)
    timestamp_path = TIMESTAMP_FILENAME_TPL.format(stock_code=stock_code)

    # 1. Check Retraining Need
    print(f"   ({stock_code}): Checking retraining need...")
    if force_retrain: print(f"   ({stock_code}): Force retraining enabled."); retrain_needed = True
    elif not all(os.path.exists(p) for p in [model_path, scaler_path, timestamp_path]): print(f"   ({stock_code}): Missing files. Training required."); retrain_needed = True
    else:
        try:
            with open(timestamp_path, 'r') as f: last_trained_time = datetime.datetime.fromisoformat(f.read())
            if (datetime.datetime.now() - last_trained_time).days >= RETRAIN_INTERVAL_DAYS: print(f"   ({stock_code}): Model outdated. Retraining required."); retrain_needed = True
            else: print(f"   ({stock_code}): Model recent. Loading existing.")
        except Exception as e: print(f"   WARN ({stock_code}): Error reading timestamp ({e}). Forcing retrain."); retrain_needed = True

    # 2. Train or Load
    if retrain_needed:
        print(f"   ({stock_code}): Initiating training...")
        ohlc_data = get_ohlc_data_for_stock(stock_code)
        if ohlc_data is None: return None, None, None
        model, scaler = train_and_save_model(stock_code, ohlc_data) # Train and get scaler
        if model is None: print(f"ERROR ({stock_code}): Training failed."); return None, None, None
    else:
        print(f"   ({stock_code}): Loading existing model and scaler...")
        try:
            model = load_model(model_path); scaler = joblib.load(scaler_path) # Load both
            print(f"[OK] LSTM ({stock_code}): Model and Scaler loaded.")
        except Exception as e:
            print(f"ERROR ({stock_code}): Failed loading - {e}. Trying fallback retrain...")
            ohlc_data = get_ohlc_data_for_stock(stock_code)
            if ohlc_data is None: return None, None, None
            model, scaler = train_and_save_model(stock_code, ohlc_data)
            if model is None: print(f"ERROR ({stock_code}): Fallback retraining failed."); return None, None, None

    if model is None or scaler is None: print(f"ERROR ({stock_code}): Model/Scaler unavailable."); return None, None, None

    # 3. Prepare Latest Data for Prediction
    print(f"   ({stock_code}): Preparing prediction input...")
    if not retrain_needed: # Fetch latest data only if not just trained
         ohlc_data = get_ohlc_data_for_stock(stock_code)
         # Check if data fetch failed *after* loading model/scaler successfully
         if ohlc_data is None: print(f"ERROR ({stock_code}): Failed getting latest data."); return None, None, scaler # Return loaded scaler
    # Check data length after potential refetch
    if len(ohlc_data) < TIMESTEPS: print(f"ERROR ({stock_code}): Not enough data for prediction."); return None, None, scaler

    last_sequence_raw = ohlc_data[-TIMESTEPS:]
    try: last_sequence_scaled = scaler.transform(last_sequence_raw)
    except Exception as e: print(f"ERROR ({stock_code}): Failed scaling input - {e}"); return None, None, scaler
    prediction_input = np.reshape(last_sequence_scaled, (1, TIMESTEPS, len(FEATURES)))

    # 4. Make Prediction
    print(f"   ({stock_code}): Predicting next Open/Close...")
    try: predicted_scaled_values = model.predict(prediction_input)[0]
    except Exception as e: print(f"ERROR ({stock_code}): LSTM prediction failed - {e}"); return None, None, scaler

    # 5. Inverse Transform
    try:
        num_features = len(FEATURES); dummy_prediction = np.zeros((1, num_features))
        target_indices = [FEATURES.index(f) for f in TARGET_FEATURES]
        for i, target_idx in enumerate(target_indices): dummy_prediction[0, target_idx] = predicted_scaled_values[i]
        predicted_values_unscaled = scaler.inverse_transform(dummy_prediction)[0]
        pred_map = {target: predicted_values_unscaled[FEATURES.index(target)] for target in TARGET_FEATURES}
        predicted_open = pred_map.get('Open'); predicted_close = pred_map.get('Close')
        if predicted_open is None or predicted_close is None: print(f"ERROR ({stock_code}): Failed extracting predictions."); return None, None, scaler
        print(f"[OK] LSTM ({stock_code}): Predicted Open={predicted_open:.2f}, Close={predicted_close:.2f}")
        # --- Return predictions AND the scaler used ---
        return predicted_open, predicted_close, scaler # <<< RETURN SCALER HERE >>>
    except Exception as e: print(f"ERROR ({stock_code}): Inverse transform failed - {e}"); return None, None, scaler

# === Main Execution Block (Modified for Initial Training of Stock List) ===
if __name__ == '__main__':
    print("="*40)
    print("   LSTM Predictor - Initial Training for Stock List")
    print("="*40)

    stock_list_file = "stock_list.txt" # Make sure this file exists and is correct
    target_stocks = []

    # --- Load Stock List ---
    print(f"Reading stock list from: {stock_list_file}")
    try:
        with open(stock_list_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.startswith('\ufeff'): content = content[1:] # Handle BOM
            target_stocks = [code.strip() for code in content.split(',') if code.strip().isdigit() and len(code.strip()) == 4]
        if not target_stocks:
            print(f"ERROR: No valid stock codes found in '{stock_list_file}'. Exiting.")
            sys.exit(1)
        print(f"[OK] Found {len(target_stocks)} stocks to process: {', '.join(target_stocks)}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Stock list file '{stock_list_file}' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed reading stock list '{stock_list_file}' - {e}. Exiting.")
        sys.exit(1)

    # --- Ensure Model Directory Exists ---
    if not os.path.exists(MODEL_DIR):
        print(f"Creating directory for models: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)

    # --- Iterate and Train/Predict for each stock ---
    print(f"\nStarting initial training/prediction loop for {len(target_stocks)} stocks...")
    results = {}
    start_initial_train_time = time.time()

    for i, stock_code in enumerate(target_stocks):
        print(f"\n--- Processing Stock {i+1}/{len(target_stocks)}: {stock_code} ---")
        stock_start_time = time.time()
        # --- Call predict function with force_retrain=True ---
        # This will trigger train_and_save_model if needed or forced
        pred_open, pred_close, used_scaler = predict_next_open_close(
            stock_code=stock_code,
            force_retrain=True # Force training/saving for initialization
        )
        # --- Store result ---
        results[stock_code] = {"open": pred_open, "close": pred_close, "scaler_ok": used_scaler is not None}
        stock_end_time = time.time()
        print(f"--- Finished processing {stock_code} in {stock_end_time - stock_start_time:.2f} seconds ---")
        # Optional small delay between stocks if API has limits
        time.sleep(1) # Wait 1 second

    end_initial_train_time = time.time()
    print("\n" + "="*40)
    print("   Initial Training/Prediction Summary")
    print("="*40)
    successful_count = 0
    failed_count = 0
    for code, res in results.items():
        if res["open"] is not None and res["scaler_ok"]:
            print(f"Stock {code}: [SUCCESS] Predicted Open={res['open']:.2f}, Close={res['close']:.2f}")
            successful_count += 1
        else:
            print(f"Stock {code}: [FAILED] Prediction or Scaler generation failed.")
            failed_count += 1
    print("-" * 40)
    print(f"Total Stocks Processed: {len(target_stocks)}")
    print(f"Successful Initializations: {successful_count}")
    print(f"Failed Initializations: {failed_count}")
    print(f"Total Initial Training Time: {end_initial_train_time - start_initial_train_time:.2f} seconds")
    print("="*40)
    print("\n--- LSTM Initial Training/Prediction Complete ---")
    if failed_count > 0:
        print("\nWARNING: Some stocks failed initialization. Check error messages above.")
        print("         The agent might skip these stocks during trading execution.")
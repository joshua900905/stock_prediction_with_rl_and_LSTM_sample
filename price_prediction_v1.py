import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
try:
    # TensorFlow 2.x approach
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
except ImportError:
    # Fallback for standalone Keras or older TF
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM
import warnings
import os
import datetime
import joblib # For saving/loading the scaler

# --- Configuration ---
MODEL_FILENAME = "lstm_ohlc_predictor_7day.h5"
SCALER_FILENAME = "ohlc_scaler_7day.pkl"
TIMESTAMP_FILENAME = "model_last_trained_7day.txt"
RETRAIN_INTERVAL_DAYS = 7
TIMESTEPS = 7
API_URL = "http://140.116.86.242:8081/api/stock/get_stock_history_data_for_ce_bot"
FEATURES = ['Open', 'High', 'Low', 'Close'] # Order matters!
TARGET_FEATURES = ['Open', 'Close'] # Features to predict

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Helper Functions ---

def get_stock_data_from_api(url):
    """Fetches and preprocesses stock data from the API."""
    print("正在從 API 獲取歷史數據...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        stock_history_data_raw = response.json().get("data", [])

        if not stock_history_data_raw:
            print("錯誤：未能從 API 獲取數據或數據為空。")
            return None

        ohlc_data = []
        for history in stock_history_data_raw:
             if all(key in history for key in FEATURES):
                 ohlc_data.append([float(history[key]) for key in FEATURES])
             else:
                 print(f"警告：記錄 {history.get('Date', '未知日期')} 缺少必要特徵，已跳過。")

        if len(ohlc_data) < TIMESTEPS + 1:
             print(f"錯誤：獲取的有效 OHLC 數據不足 {len(ohlc_data)} 筆，至少需要 {TIMESTEPS + 1} 筆。")
             return None

        ohlc_data = np.array(ohlc_data, dtype=np.float32)
        print(f"獲取並整理了 {ohlc_data.shape[0]} 天的 {len(FEATURES)} 特徵數據。")
        return ohlc_data

    except requests.exceptions.RequestException as e:
        print(f"錯誤：請求 API 失敗 - {e}")
        return None
    except Exception as e:
        print(f"錯誤：處理 API 數據時出錯 - {e}")
        return None

def prepare_lstm_data(scaled_data, target_indices):
    """Prepares data sequences for LSTM training."""
    X_data = []
    y_data = []
    num_features = scaled_data.shape[1]

    for i in range(TIMESTEPS, len(scaled_data)):
        X_data.append(scaled_data[i-TIMESTEPS:i, :]) # Input: previous TIMESTEPS days, all features
        y_data.append(scaled_data[i, target_indices]) # Target: next day's target features

    X_data, y_data = np.array(X_data), np.array(y_data)

    if X_data.shape[0] == 0:
        print("錯誤：無法創建訓練序列。")
        return None, None
    print(f"創建了 {X_data.shape[0]} 個訓練樣本。")
    print(f"X_data shape: {X_data.shape}") # (樣本數, TIMESTEPS, num_features)
    print(f"y_data shape: {y_data.shape}") # (樣本數, num_target_features)
    return X_data, y_data

def build_lstm_model(input_shape, output_units):
    """Builds the LSTM model structure."""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape)) # Increased units
    model.add(LSTM(units=32)) # Second LSTM layer
    model.add(Dense(units=output_units)) # Output layer predicts specified units
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM 模型結構建立完成。")
    model.summary() # Print model summary
    return model

def train_and_save_model(data, model_path, scaler_path, timestamp_path):
    """Trains the LSTM model, saves it, the scaler, and the timestamp."""
    print("開始模型訓練流程...")
    # 1. Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print("數據標準化完成。")

    # 2. Prepare Data for LSTM
    target_indices = [FEATURES.index(f) for f in TARGET_FEATURES] # Get indices of Open and Close
    X_train, y_train = prepare_lstm_data(scaled_data, target_indices)
    if X_train is None:
        return None, None # Training failed

    # 3. Build Model
    num_features = data.shape[1]
    num_target_features = len(TARGET_FEATURES)
    model = build_lstm_model(input_shape=(TIMESTEPS, num_features), output_units=num_target_features)

    # 4. Train Model
    print("正在訓練 LSTM 模型...")
    # Reduced epochs as it might be retrained periodically
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)
    print("模型訓練完成。")

    # 5. Save Model, Scaler, and Timestamp
    try:
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        with open(timestamp_path, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        print(f"模型已儲存至 {model_path}")
        print(f"Scaler 已儲存至 {scaler_path}")
        print(f"時間戳已儲存至 {timestamp_path}")
        return model, scaler
    except Exception as e:
        print(f"錯誤：儲存模型/Scaler/時間戳失敗 - {e}")
        return None, None

# --- Main Prediction Function ---

def predict_next_open_close(force_retrain=False):
    """
    Predicts next day's Open and Close prices.
    Loads existing model if recent, otherwise retrains every RETRAIN_INTERVAL_DAYS.
    """
    model = None
    scaler = None
    retrain_needed = False

    # 1. Check if model needs retraining
    if force_retrain:
        print("強制重新訓練模型。")
        retrain_needed = True
    elif not os.path.exists(MODEL_FILENAME) or not os.path.exists(SCALER_FILENAME) or not os.path.exists(TIMESTAMP_FILENAME):
        print("模型、Scaler 或時間戳文件不存在，需要訓練新模型。")
        retrain_needed = True
    else:
        try:
            with open(TIMESTAMP_FILENAME, 'r') as f:
                last_trained_time = datetime.datetime.fromisoformat(f.read())
            time_since_last_train = datetime.datetime.now() - last_trained_time
            if time_since_last_train.days >= RETRAIN_INTERVAL_DAYS:
                print(f"模型上次訓練時間為 {last_trained_time}，已超過 {RETRAIN_INTERVAL_DAYS} 天，需要重新訓練。")
                retrain_needed = True
            else:
                print(f"模型上次訓練時間為 {last_trained_time}，在 {RETRAIN_INTERVAL_DAYS} 天內，將載入現有模型。")
        except Exception as e:
            print(f"錯誤：讀取時間戳失敗，將重新訓練模型 - {e}")
            retrain_needed = True

    # 2. Train or Load Model and Scaler
    if retrain_needed:
        ohlc_data = get_stock_data_from_api(API_URL)
        if ohlc_data is None:
            return None, None # Exit if data fetching fails
        model, scaler = train_and_save_model(ohlc_data, MODEL_FILENAME, SCALER_FILENAME, TIMESTAMP_FILENAME)
        if model is None:
            print("模型訓練失敗。")
            return None, None
    else:
        try:
            print("正在載入現有模型和 Scaler...")
            model = load_model(MODEL_FILENAME)
            scaler = joblib.load(SCALER_FILENAME)
            print("模型和 Scaler 載入成功。")
        except Exception as e:
            print(f"錯誤：載入模型或 Scaler 失敗，將嘗試重新訓練 - {e}")
            # Fallback to retraining if loading fails
            ohlc_data = get_stock_data_from_api(API_URL)
            if ohlc_data is None:
                 return None, None
            model, scaler = train_and_save_model(ohlc_data, MODEL_FILENAME, SCALER_FILENAME, TIMESTAMP_FILENAME)
            if model is None:
                print("模型重新訓練失敗。")
                return None, None

    # 3. Prepare Data for Prediction
    # Fetch data again ONLY IF NOT just retrained (to get latest for prediction)
    if not retrain_needed:
         ohlc_data = get_stock_data_from_api(API_URL)
         if ohlc_data is None:
             print("無法獲取最新數據進行預測。")
             return None, None
    elif ohlc_data is None: # Should not happen if retrain was successful, but safety check
        print("數據不可用，無法預測。")
        return None, None

    # Take the last TIMESTEPS data points
    last_sequence = ohlc_data[-TIMESTEPS:]
    if last_sequence.shape[0] < TIMESTEPS:
        print(f"錯誤：最新數據序列長度不足 {TIMESTEPS}。")
        return None, None

    # Scale the input sequence using the loaded/trained scaler
    last_sequence_scaled = scaler.transform(last_sequence)

    # Reshape for LSTM input (1 sample, TIMESTEPS, num_features)
    prediction_input = np.reshape(last_sequence_scaled, (1, TIMESTEPS, ohlc_data.shape[1]))

    # 4. Make Prediction
    print("正在預測下一天開盤價和收盤價...")
    predicted_scaled_values = model.predict(prediction_input)[0] # Get the first (only) prediction -> [scaled_open, scaled_close]

    # 5. Inverse Transform
    # Create a dummy array with the same number of features as the scaler was fit on
    num_features = len(FEATURES)
    dummy_prediction = np.zeros((1, num_features))

    # Place the predicted scaled values into the correct columns (Open and Close)
    open_index = FEATURES.index('Open')
    close_index = FEATURES.index('Close')
    target_pred_indices = [FEATURES.index(f) for f in TARGET_FEATURES]

    # Handle the order based on TARGET_FEATURES
    pred_map = {target: pred for target, pred in zip(TARGET_FEATURES, predicted_scaled_values)}

    dummy_prediction[0, open_index] = pred_map.get('Open', 0) # Default to 0 if somehow not predicted
    dummy_prediction[0, close_index] = pred_map.get('Close', 0)

    # Inverse transform the dummy array
    predicted_values_unscaled = scaler.inverse_transform(dummy_prediction)[0]

    # Extract the final predicted Open and Close prices
    predicted_open = predicted_values_unscaled[open_index]
    predicted_close = predicted_values_unscaled[close_index]

    print(f"預測的下一天開盤價為: {predicted_open:.2f}")
    print(f"預測的下一天收盤價為: {predicted_close:.2f}")

    return predicted_open, predicted_close


# --- How to Use ---
if __name__ == '__main__':
    # Example usage:
    # To force retraining even if the model is recent:
    # pred_open, pred_close = predict_next_open_close(force_retrain=True)

    # Normal usage (retrains only if needed):
    pred_open, pred_close = predict_next_open_close()

    if pred_open is not None and pred_close is not None:
        print(f"\n函數返回的預測價格: Open={pred_open:.2f}, Close={pred_close:.2f}")
    else:
        print("\n未能成功獲取預測價格。")
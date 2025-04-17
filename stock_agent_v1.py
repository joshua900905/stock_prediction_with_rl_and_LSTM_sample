# stock_agent.py - RL Trading Agent using OHLC State and LSTM Predictions

from datetime import datetime
import os
import math
import numpy as np
import random
from collections import deque
import sys
import requests
import joblib # Needed to load the scaler

# Keras/TensorFlow imports
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential, load_model
    from keras.layers import Dense
    from keras.optimizers import Adam

# --- Import the improved LSTM predictor ---
try:
    # Assumes price_prediction_v1.py contains predict_next_open_close
    # and uses these filenames for its scaler
    from price_prediction_v1 import predict_next_open_close, SCALER_FILENAME as LSTM_SCALER_FILENAME
    print("成功導入 price_prediction_v1.py")
except ImportError:
    print("錯誤：無法找到或導入 'price_prediction_v1.py'。請確保該文件存在且路徑正確。")
    sys.exit(1)
except AttributeError:
    # Fallback if SCALER_FILENAME is not defined in the imported script
    print("警告: 無法從 price_prediction_v1.py 導入 SCALER_FILENAME, 使用預設名稱。")
    LSTM_SCALER_FILENAME = "ohlc_scaler_7day_weighted.pkl"


# === Configuration ===
NUM_FEATURES = 4 # OHLC

# === API Helper Functions (Get_Stock_Informations, Get_User_Stocks, Buy_Stock, Sell_Stock) ===
# ... (Keep these functions as they are - see previous response for the code) ...
# 取得股票資訊
def Get_Stock_Informations(stock_code, start_date, stop_date):
    """取得指定股票在日期範圍內的歷史資訊。"""
    information_url = (f'http://140.116.86.242:8081/stock/'
                       f'api/v1/api_get_stock_info_from_date_json/'
                       f'{stock_code}/{start_date}/{stop_date}')
    try:
        response = requests.get(information_url, timeout=10) # Added timeout
        response.raise_for_status()
        result = response.json()
        if result and result.get('result') == 'success':
            return result.get('data', [])
        else:
            print(f"API Get_Stock_Informations 失敗: {result.get('status', '未知錯誤')}")
            return []
    except requests.exceptions.Timeout:
        print("請求股票資訊 API 超時。")
        return []
    except requests.exceptions.RequestException as e:
        print(f"請求股票資訊 API 時出錯: {e}")
        return []
    except Exception as e:
        print(f"處理股票資訊時發生錯誤: {e}")
        return []

# 取得持有股票
def Get_User_Stocks(account, password):
    """取得指定用戶當前持有的股票列表。"""
    data = {'account': account, 'password': password}
    search_url = 'http://140.116.86.242:8081/stock/api/v1/get_user_stocks'
    try:
        response = requests.post(search_url, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result and result.get('result') == 'success':
            return result.get('data', [])
        else:
            print(f"API Get_User_Stocks 失敗: {result.get('status', '未知錯誤')}")
            return []
    except requests.exceptions.Timeout:
        print("請求用戶持股 API 超時。")
        return []
    except requests.exceptions.RequestException as e:
        print(f"請求用戶持股 API 時出錯: {e}")
        return []
    except Exception as e:
        print(f"處理用戶持股時發生錯誤: {e}")
        return []

# 預約購入股票
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    """向 API 發送預約購買股票的請求。"""
    print(f'嘗試購買股票 {stock_code} ({stock_shares} 張) @ {stock_price:.2f}...')
    data = {'account': account,
            'password': password,
            'stock_code': str(stock_code),
            'stock_shares': str(stock_shares),
            'stock_price': f"{stock_price:.2f}"}
    buy_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
    try:
        response = requests.post(buy_url, data=data, timeout=15) # Longer timeout for actions
        response.raise_for_status()
        result = response.json()
        print('購買請求結果: ' + result.get('result', 'N/A') + " | 狀態: " + result.get('status', 'N/A'))
        return result.get('result') == 'success'
    except requests.exceptions.Timeout:
        print("請求購買 API 超時。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"請求購買 API 時出錯: {e}")
        return False
    except Exception as e:
        print(f"處理購買請求時發生錯誤: {e}")
        return False

# 預約售出股票
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    """向 API 發送預約售出股票的請求。"""
    print(f'嘗試賣出股票 {stock_code} ({stock_shares} 張) @ {stock_price:.2f}...')
    data = {'account': account,
            'password': password,
            'stock_code': str(stock_code),
            'stock_shares': str(stock_shares),
            'stock_price': f"{stock_price:.2f}"}
    sell_url = 'http://140.116.86.242:8081/stock/api/v1/sell'
    try:
        response = requests.post(sell_url, data=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        print('售出請求結果: ' + result.get('result', 'N/A') + " | 狀態: " + result.get('status', 'N/A'))
        return result.get('result') == 'success'
    except requests.exceptions.Timeout:
        print("請求售出 API 超時。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"請求售出 API 時出錯: {e}")
        return False
    except Exception as e:
        print(f"處理售出請求時發生錯誤: {e}")
        return False
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# === RL Helper Functions (Refactored) ===

def formatPrice(n):
    """簡單格式化價格字串。"""
    try:
        return ("-" if n < 0 else "") + "NT${:,.2f}".format(abs(n))
    except TypeError:
        return "N/A"

def get_ohlc_data_for_rl():
    """
    獲取用於 RL 狀態的 OHLC 歷史數據。
    與 lstm_predictor 中的數據獲取保持一致。
    """
    print("RL Agent: 正在從 API 獲取 OHLC 歷史數據...")
    features_to_get = ['Open', 'High', 'Low', 'Close'] # Ensure consistency
    api_url = "http://140.116.86.242:8081/api/stock/get_stock_history_data_for_ce_bot"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        stock_history_data_raw = response.json().get("data", [])

        if not stock_history_data_raw:
            print("錯誤 (get_ohlc_data_for_rl)：無法從 API 獲取數據。")
            return None

        ohlc_data = []
        for history in stock_history_data_raw:
            if all(key in history for key in features_to_get):
                try:
                    ohlc_data.append([float(history[key]) for key in features_to_get])
                except (ValueError, TypeError):
                     print(f"警告 (get_ohlc_data_for_rl): 無法轉換記錄 {history.get('Date', '未知')} 的 OHLC 值。")
            else:
                 print(f"警告 (get_ohlc_data_for_rl): 記錄 {history.get('Date', '未知')} 缺少 OHLC 鍵。")

        if not ohlc_data: # Check if any valid data was processed
             print("錯誤 (get_ohlc_data_for_rl): 未能整理出有效的 OHLC 數據。")
             return None

        ohlc_data = np.array(ohlc_data, dtype=np.float32)
        print(f"get_ohlc_data_for_rl: 獲取並整理了 {ohlc_data.shape[0]} 天的 OHLC 數據。")
        return ohlc_data

    except requests.exceptions.Timeout:
        print("錯誤 (get_ohlc_data_for_rl): 請求 API 超時。")
        return None
    except requests.exceptions.RequestException as e:
        print(f"錯誤 (get_ohlc_data_for_rl): 請求 API 失敗 - {e}")
        return None
    except Exception as e:
        print(f"錯誤 (get_ohlc_data_for_rl): 處理 API 數據時出錯 - {e}")
        return None

def getState_OHLC(ohlc_data, t, window_size, scaler):
    """
    計算 RL 狀態，使用過去 window_size 天的正規化 OHLC 數據。
    `ohlc_data`: 完整的 OHLC NumPy 陣列 (days, features=4)。
    `t`: 當前時間點索引。
    `window_size`: 需要回看的天數。
    `scaler`: 已載入的、用於 LSTM 訓練的 MinMaxScaler 物件。
    Returns: shape (1, window_size * 4) 的 NumPy 陣列。
    """
    n_features = NUM_FEATURES # Should be 4
    state_dim = window_size * n_features

    if scaler is None:
        print("錯誤 (getState_OHLC): 未提供 Scaler 物件。")
        # Return a zero state of the expected shape
        return np.zeros((1, state_dim))
    if not hasattr(scaler, 'transform'):
         print("錯誤 (getState_OHLC): 提供的 scaler 物件無效。")
         return np.zeros((1, state_dim))

    if t < window_size - 1: # Need at least window_size days ending at index t
        # print(f"警告 (getState_OHLC): t={t} 時歷史數據不足 {window_size} 天。")
        # Return a zero state or handle differently?
        return np.zeros((1, state_dim)) # Pad with zeros

    start_index = t - window_size + 1
    end_index = t + 1 # Slice up to t (inclusive)

    if start_index < 0 or end_index > len(ohlc_data):
        print(f"錯誤 (getState_OHLC): 索引超出範圍 (start={start_index}, end={end_index}, len={len(ohlc_data)})。")
        return np.zeros((1, state_dim))

    # Extract the window of OHLC data
    window_data = ohlc_data[start_index : end_index, :] # Shape: (window_size, n_features)

    if window_data.shape != (window_size, n_features):
         print(f"錯誤 (getState_OHLC): 提取的數據窗口 shape 不正確 ({window_data.shape})，期望 ({window_size}, {n_features})。")
         return np.zeros((1, state_dim))

    # --- Normalize using the provided scaler ---
    try:
         scaled_window = scaler.transform(window_data)
    except ValueError as ve:
         # This might happen if the number of features doesn't match the scaler
         print(f"錯誤 (getState_OHLC): 使用 Scaler 進行 transform 時發生錯誤 - {ve}。檢查特徵數量是否匹配。")
         print(f"數據 Shape: {window_data.shape}, Scaler 預期特徵數: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else '未知'}")
         return np.zeros((1, state_dim))
    except Exception as e:
         print(f"錯誤 (getState_OHLC): 使用 Scaler 進行 transform 時發生未知錯誤 - {e}")
         return np.zeros((1, state_dim))


    # --- Flatten the scaled window into a single state vector ---
    state = scaled_window.flatten().reshape(1, -1) # Shape (1, window_size * n_features)

    if state.shape[1] != state_dim:
        print(f"錯誤 (getState_OHLC): 最終狀態 shape 不正確 ({state.shape})，期望 (1, {state_dim})。")
        return np.zeros((1, state_dim)) # Should not happen if flatten works

    return state

# === Reinforcement Learning Agent Class (Modified for new state size) ===
class Agent:
    """DQN Agent - Modified to accept state_size based on OHLC window."""
    def __init__(self, window_size, is_eval=False, model_path=""):
        # State size is now window_size * number_of_features (OHLC=4)
        self.state_size = window_size * NUM_FEATURES
        self.window_size = window_size # Keep original window size if needed elsewhere
        self.action_size = 3 # 0: Sit, 1: Buy, 2: Sell
        self.memory = deque(maxlen=1000)
        self.model_path = model_path
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 0.0 if is_eval else 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Load or build model
        if is_eval and model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                # --- Verification ---
                loaded_input_dim = self.model.layers[0].input_shape[1]
                if loaded_input_dim != self.state_size:
                     print(f"**** 嚴重警告 ****")
                     print(f"載入的 RL 模型 '{model_path}' 輸入維度 ({loaded_input_dim})")
                     print(f"與期望的 OHLC 狀態維度 ({self.state_size} = {window_size} * {NUM_FEATURES}) 不匹配！")
                     print(f"請使用以 OHLC 狀態重新訓練的模型！")
                     # Decide action: exit, fallback, or proceed with caution?
                     # For now, proceed but expect errors or bad performance.
                else:
                     print(f"RL 模型 '{model_path}' 成功載入，輸入維度 ({self.state_size}) 匹配。")
            except Exception as e:
                print(f"錯誤：載入 RL 模型 '{model_path}' 失敗 - {e}。將建立新模型。")
                self.model = self._build_model()
        else:
            if is_eval and model_path:
                print(f"警告：評估模式下找不到 RL 模型 '{model_path}'。將建立新模型。")
            self.model = self._build_model()

    def _build_model(self):
        """Builds the Q-network model with the correct input dimension."""
        model = Sequential()
        # --- Ensure input_dim matches the new state_size ---
        model.add(Dense(128, input_dim=self.state_size, activation='relu')) # Increased units slightly
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu')) # Optional extra layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        print(f"新的 RL Q-Network 模型已建立 (輸入維度: {self.state_size})。")
        return model

    # --- act, remember, replay, load, save methods ---
    # (These methods remain largely the same structurally, but rely on correct state shapes)
    def act(self, state):
        """Choose action based on state using epsilon-greedy policy."""
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore

        if state is None or not isinstance(state, np.ndarray) or state.shape != (1, self.state_size):
             print(f"錯誤 (act): 無效的狀態輸入 shape: {state.shape if state is not None else 'None'}，期望 (1, {self.state_size})。返回隨機動作。")
             return random.randrange(self.action_size)
        try:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0]) # Exploit
        except Exception as e:
            print(f"錯誤 (act): 模型預測失敗 - {e}。返回隨機動作。")
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
         """Store experience in replay memory (for training)."""
         if not self.is_eval:
             # Ensure states have the correct shape before appending
             if state.shape == (1, self.state_size) and next_state.shape == (1, self.state_size):
                 self.memory.append((state, action, reward, next_state, done))
             else:
                 print(f"警告 (remember): 狀態 shape 不正確，經驗未存儲。 State:{state.shape}, Next:{next_state.shape}")

    def replay(self, batch_size):
        """Train the model using experience replay (standard random sampling)."""
        if self.is_eval or len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare states and next_states from the batch
        states = np.vstack([t[0] for t in minibatch]) # Shape (batch_size, state_size)
        next_states = np.vstack([t[3] for t in minibatch]) # Shape (batch_size, state_size)

        # Verify shapes before prediction
        if states.shape[1] != self.state_size or next_states.shape[1] != self.state_size:
             print(f"錯誤 (replay): 批次中狀態維度不匹配。跳過此批次。")
             return

        try:
            # Predict Q-values in batches
            q_current_batch = self.model.predict(states)
            q_next_batch = self.model.predict(next_states)

            targets_batch = np.copy(q_current_batch) # Initialize targets

            # Calculate targets for Bellman equation
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    targets_batch[i, action] = reward
                else:
                    targets_batch[i, action] = reward + self.gamma * np.amax(q_next_batch[i])

            # Fit the model on the entire batch
            self.model.fit(states, targets_batch, epochs=1, verbose=0)

        except Exception as e:
            print(f"錯誤 (replay): 批次預測或擬合時發生錯誤 - {e}")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights."""
        if os.path.exists(name):
            try:
                # Load the whole model structure and weights
                self.model = load_model(name)
                # Verify input shape after loading
                loaded_input_dim = self.model.layers[0].input_shape[1]
                if loaded_input_dim != self.state_size:
                     print(f"**** 嚴重警告 (load) ****")
                     print(f"載入的 RL 模型 '{name}' 輸入維度 ({loaded_input_dim})")
                     print(f"與期望的 OHLC 狀態維度 ({self.state_size}) 不匹配！")
                else:
                     print(f"RL 模型從 '{name}' 完整載入成功。")
            except Exception as e:
                print(f"錯誤: 無法載入 RL 模型 '{name}' - {e}")
        else:
            print(f"警告: RL 模型文件 '{name}' 不存在。")

    def save(self, name):
        """Save the entire model."""
        try:
            self.model.save(name) # Save structure and weights
            print(f"RL 模型已儲存至 '{name}'。")
        except Exception as e:
            print(f"錯誤: 無法儲存 RL 模型至 '{name}' - {e}")


# === RL Training Function (Modified for OHLC State) ===
def training_rl_model():
    """Function to train the RL agent using the new OHLC state."""
    print("--- 開始 RL 模型訓練 (使用 OHLC 狀態) ---")
    # --- Config ---
    window_size = 10    # << MUST match Agent's window_size for state calculation
    episode_count = 50 # <<< SET TO A VALUE > 0 TO ENABLE TRAINING >>> (e.g., 50-100)
    batch_size = 32
    # <<< CHOOSE A *NEW* FILENAME for the model trained with OHLC state >>>
    model_save_prefix = "rl_agent_ohlc_state_"

    # --- Load Scaler (Crucial!) ---
    # Assumes the LSTM has been trained at least once and scaler file exists
    scaler_path = LSTM_SCALER_FILENAME
    if not os.path.exists(scaler_path):
        print(f"錯誤：找不到 LSTM Scaler 文件 '{scaler_path}'。請先運行 LSTM 預測器以生成 Scaler。訓練終止。")
        return
    try:
        scaler = joblib.load(scaler_path)
        print(f"LSTM Scaler '{scaler_path}' 載入成功。")
    except Exception as e:
        print(f"錯誤：載入 LSTM Scaler '{scaler_path}' 失敗 - {e}。訓練終止。")
        return

    # --- Initialize Agent ---
    # state_size is now window_size * NUM_FEATURES
    agent = Agent(window_size=window_size, is_eval=False)

    # --- Get OHLC Data for Training ---
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) <= window_size:
        print("訓練數據不足或獲取失敗，無法進行 RL 訓練。")
        return
    data_len = len(ohlc_data)

    print(f"RL 訓練數據: {ohlc_data.shape}")

    # --- Training Loop ---
    for e in range(episode_count):
        print(f"\n--- Episode {e + 1}/{episode_count} ---")
        # Initial state calculation needs index window_size-1
        current_t = window_size - 1
        state = getState_OHLC(ohlc_data, current_t, window_size, scaler)
        if state.shape[1] != agent.state_size: # Check if state generation failed
             print(f"錯誤：無法創建初始狀態 (t={current_t})。跳過此回合。")
             continue

        total_profit = 0
        episode_inventory = [] # Track simulated holdings

        # Loop through time steps available for state creation
        # Need data up to t+1 to calculate reward based on next day's close
        for t in range(window_size - 1, data_len - 1):
            action = agent.act(state)
            current_close_price = ohlc_data[t, 3] # Index 3 is 'Close'
            next_close_price = ohlc_data[t + 1, 3]

            # Get next state
            next_state = getState_OHLC(ohlc_data, t + 1, window_size, scaler)
            if next_state.shape[1] != agent.state_size:
                 print(f"錯誤：無法創建下一個狀態 (t={t+1})。結束此回合。")
                 done = True
                 reward = -10 # Penalty
            else:
                 done = (t == data_len - 2) # Episode ends when we reach the last possible next_state
                 reward = 0 # Default reward

                 # --- Simulate action and calculate reward ---
                 # Reward based on profit using *next day's* closing price for sell/buy cost
                 if action == 1: # Buy
                     episode_inventory.append(next_close_price) # Assume bought at next close
                     # print(f"t={t}: Buy (Cost ~ {next_close_price:.2f})")
                 elif action == 2 and episode_inventory: # Sell
                     bought_price = episode_inventory.pop(0) # FIFO
                     profit = next_close_price - bought_price # Realized at next close
                     reward = profit
                     total_profit += profit
                     # print(f"t={t}: Sell @ {next_close_price:.2f} (Bought Cost ~ {bought_price:.2f}), Profit: {profit:.2f}")
                 elif action == 2 and not episode_inventory: # Sell invalid
                     reward = -5 # Penalty

            # Store experience (only if states are valid)
            if state.shape[1] == agent.state_size and next_state.shape[1] == agent.state_size:
                 agent.remember(state, action, reward, next_state, done)
            else:
                print(f"警告：由於狀態無效，經驗 (t={t}) 未存儲。")

            state = next_state # Move to next state

            # Train the agent using experience replay
            # Optional: Only replay every few steps (e.g., if t % 4 == 0:)
            agent.replay(batch_size)

            if done:
                print(f"Episode {e + 1} 結束。模擬總利潤: {formatPrice(total_profit)}")
                print(f"最終 Epsilon: {agent.epsilon:.4f}")
                break # End of episode

        # Save model periodically
        if (e + 1) % 10 == 0 or e == episode_count - 1:
            save_filename = f"{model_save_prefix}ep{e+1}.h5"
            agent.save(save_filename)

    print("--- RL 模型訓練 (OHLC 狀態) 結束 ---")


# === Trading Execution Function (Modified for OHLC State) ===
def trading_execution():
    """Main function to run trading logic using OHLC state."""
    print("\n--- 開始執行交易決策 (使用 OHLC 狀態) ---")
    # --- Config ---
    stock_code_to_trade = "2330"
    shares_to_trade = 1
    # <<< POINT TO THE *NEW* RL MODEL TRAINED WITH OHLC STATE >>>
    rl_model_load_path = "rl_agent_ohlc_state_ep50.h5" # Example: Use model saved after 50 episodes
    window_size = 10   # <<< MUST MATCH the window_size used during RL training >>>
    account = "CIoT Bot-CE"
    password = "CIoT Bot-CE"

    # --- Load Scaler (Essential for getState_OHLC) ---
    scaler_path = LSTM_SCALER_FILENAME
    if not os.path.exists(scaler_path):
        print(f"錯誤：找不到 LSTM Scaler 文件 '{scaler_path}'。交易無法執行。")
        return
    try:
        scaler = joblib.load(scaler_path)
        print(f"LSTM Scaler '{scaler_path}' 載入成功 for RL state generation。")
    except Exception as e:
        print(f"錯誤：載入 LSTM Scaler '{scaler_path}' 失敗 - {e}。交易無法執行。")
        return

    # --- Load RL Model ---
    if not os.path.exists(rl_model_load_path):
        print(f"錯誤：找不到指定的 RL 模型 '{rl_model_load_path}'。交易無法執行。")
        return

    # Initialize Agent in Eval mode - it will load the model and verify dims
    agent = Agent(window_size=window_size, is_eval=True, model_path=rl_model_load_path)
    # Check if the loaded model's expected input dim matches our calculation
    if not hasattr(agent, 'model') or agent.model.layers[0].input_shape[1] != agent.state_size:
         print("RL 模型載入或維度驗證失敗。交易終止。")
         return

    # --- Get OHLC Data ---
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) < window_size:
        print("獲取 OHLC 數據失敗或數據不足。交易終止。")
        return

    last_index = len(ohlc_data) - 1 # Index for the latest available data

    # --- Create Current RL State using OHLC and Scaler ---
    current_state = getState_OHLC(ohlc_data, last_index, window_size, scaler)
    if current_state.shape[1] != agent.state_size:
        print(f"錯誤：無法創建當前 OHLC RL 狀態 (t={last_index})。交易終止。")
        return
    # print("當前 OHLC RL 狀態 (shape):", current_state.shape) # Optional

    # --- Get RL Action ---
    rl_action = agent.act(current_state)
    action_map = {0: "觀望 (Sit)", 1: "買入 (Buy)", 2: "賣出 (Sell)"}
    print(f"RL Agent 決策 (基於 OHLC 狀態): {rl_action} ({action_map.get(rl_action, '未知')})")

    # --- Get LSTM Price Prediction ---
    print("\n調用 LSTM 預測器 (price_prediction_v1)...")
    predicted_open, predicted_close = predict_next_open_close() # From imported file

    if predicted_open is None or predicted_close is None:
        print("LSTM 價格預測失敗。交易終止。")
        return

    # --- Decide Order Price ---
    order_price = predicted_open # Use predicted open price
    print(f"LSTM 預測: Open={predicted_open:.2f}, Close={predicted_close:.2f}")
    print(f"使用預測開盤價 {order_price:.2f} 執行交易。")

    # --- Execute Trade ---
    if rl_action == 1: # Buy
        Buy_Stock(account, password, stock_code_to_trade, shares_to_trade, order_price)
    elif rl_action == 2: # Sell
        user_stocks = Get_User_Stocks(account, password)
        holds_enough = False
        # ... (rest of the selling logic with stock holding check - same as before) ...
        for stock in user_stocks:
            if stock.get('stock_code') == stock_code_to_trade:
                 try:
                     if int(stock.get('stock_shares', 0)) >= shares_to_trade:
                         holds_enough = True
                         break
                 except (ValueError, TypeError):
                     print(f"警告：無法解析持有股票數量 for {stock_code_to_trade}")
                     continue

        if holds_enough:
            Sell_Stock(account, password, stock_code_to_trade, shares_to_trade, order_price)
        else:
            print(f"RL 建議賣出，但用戶未持有足夠 {shares_to_trade} 張 {stock_code_to_trade} 股票。不執行賣出。")
    else: # rl_action == 0 (Sit)
        print("RL 建議觀望，不執行交易。")

    print("\n--- 交易決策執行完畢 (OHLC 狀態) ---")


# === Main Execution Block ===
if __name__ == "__main__":
    # 1. Ensure lstm predictor runs first if scaler doesn't exist
    if not os.path.exists(LSTM_SCALER_FILENAME):
        print(f"找不到 LSTM Scaler '{LSTM_SCALER_FILENAME}'。需要先運行 LSTM 預測 (或訓練) 來生成它。")
        print("嘗試運行一次 LSTM 預測來生成 Scaler...")
        predict_next_open_close() # Run it once to potentially train/save scaler
        if not os.path.exists(LSTM_SCALER_FILENAME):
             print("仍然找不到 Scaler 文件。請檢查 price_prediction_v1.py。程式終止。")
             sys.exit(1)

    # 2. Optional: Train the RL model with the new state
    #    (Comment out if you have a pre-trained model)
    # print("檢查是否需要訓練 RL 模型 (使用 OHLC 狀態)...")
    # training_rl_model()

    # 3. Execute the trading logic
    trading_execution()
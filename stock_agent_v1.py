# stock_agent.py - RL Trading Agent using OHLC+Holding State and LSTM Predictions

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
    print("警告: 無法從 price_prediction_v1.py 導入 SCALER_FILENAME, 使用預設名稱。")
    LSTM_SCALER_FILENAME = "ohlc_scaler_7day_weighted.pkl"


# === Configuration ===
NUM_OHLC_FEATURES = 4 # Open, High, Low, Close
# State now includes OHLC window + 1 holding status feature
# STATE_SIZE will be calculated in Agent init based on window_size

# Stock to trade (used in multiple places)
STOCK_CODE_TO_TRADE = "2330" # Example: TSMC

# === API Helper Functions ===
# ... (Keep Get_Stock_Informations, Get_User_Stocks, Buy_Stock, Sell_Stock as before) ...
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

# === RL Helper Functions (Refactored for OHLC+Holding State) ===

def formatPrice(n):
    """簡單格式化價格字串。"""
    try:
        return ("-" if n < 0 else "") + "NT${:,.2f}".format(abs(n))
    except TypeError:
        return "N/A"

def get_ohlc_data_for_rl():
    """獲取用於 RL 狀態的 OHLC 歷史數據。"""
    # ... (This function remains the same as in the previous OHLC-only state version) ...
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

def getState_OHLC_Holding(ohlc_data, t, window_size, scaler, holding_status):
    """
    計算 RL 狀態，使用過去 window_size 天的正規化 OHLC 數據，
    並附加持有狀態。
    `holding_status`: 0.0 (未持有) or 1.0 (持有).
    Returns: shape (1, window_size * NUM_OHLC_FEATURES + 1) 的 NumPy 陣列。
    """
    n_ohlc_features = NUM_OHLC_FEATURES # Should be 4
    expected_state_dim = window_size * n_ohlc_features + 1 # New dimension

    # Create a fallback state with the correct dimension
    fallback_state = np.zeros((1, expected_state_dim))

    if scaler is None:
        print("錯誤 (getState_OHLC_Holding): 未提供 Scaler 物件。")
        return fallback_state
    if not hasattr(scaler, 'transform'):
        print("錯誤 (getState_OHLC_Holding): 提供的 scaler 物件無效。")
        return fallback_state
    if holding_status not in [0.0, 1.0]:
         print(f"警告 (getState_OHLC_Holding): 無效的 holding_status ({holding_status})，將使用 0.0。")
         holding_status = 0.0

    if t < window_size - 1:
        # print(f"警告 (getState_OHLC_Holding): t={t} 時歷史數據不足 {window_size} 天。返回零狀態。")
        # Append the current holding status even to the zero state
        fallback_state[0, -1] = holding_status
        return fallback_state

    start_index = t - window_size + 1
    end_index = t + 1 # Slice up to t (inclusive)

    if start_index < 0 or end_index > len(ohlc_data):
        print(f"錯誤 (getState_OHLC_Holding): 索引超出範圍 (start={start_index}, end={end_index}, len={len(ohlc_data)})。")
        fallback_state[0, -1] = holding_status
        return fallback_state

    window_data = ohlc_data[start_index : end_index, :]

    if window_data.shape != (window_size, n_ohlc_features):
        print(f"錯誤 (getState_OHLC_Holding): 提取的數據窗口 shape 不正確 ({window_data.shape})。")
        fallback_state[0, -1] = holding_status
        return fallback_state

    # Normalize OHLC data
    try:
        scaled_window = scaler.transform(window_data)
    except Exception as e:
        print(f"錯誤 (getState_OHLC_Holding): Scaler transform 失敗 - {e}")
        fallback_state[0, -1] = holding_status
        return fallback_state

    # Flatten the OHLC window
    ohlc_state_part = scaled_window.flatten() # Shape: (window_size * n_ohlc_features,)

    # --- Append the holding status ---
    # Convert holding_status to a NumPy array element if it isn't already
    holding_feature = np.array([holding_status], dtype=np.float32)

    # Concatenate OHLC part and holding status
    full_state_flat = np.concatenate((ohlc_state_part, holding_feature))

    # Reshape to (1, new_state_dimension)
    state = full_state_flat.reshape(1, -1)

    if state.shape[1] != expected_state_dim:
        print(f"錯誤 (getState_OHLC_Holding): 最終狀態 shape 不正確 ({state.shape})，期望 (1, {expected_state_dim})。")
        # Return fallback state which already includes holding status at the end
        fallback_state[0, -1] = holding_status
        return fallback_state

    return state


# === Reinforcement Learning Agent Class (Modified for Holding State) ===
class Agent:
    """DQN Agent - Modified for OHLC + Holding state."""
    def __init__(self, window_size, is_eval=False, model_path=""):
        # === NEW state_size calculation ===
        self.state_size = window_size * NUM_OHLC_FEATURES + 1 # OHLC window + 1 holding status
        # ==================================
        self.window_size = window_size # Store original window if needed
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.model_path = model_path
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 0.0 if is_eval else 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Load or build model, including verification of input dimension
        if is_eval and model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                loaded_input_dim = self.model.layers[0].input_shape[1]
                if loaded_input_dim != self.state_size:
                     print(f"**** 嚴重警告 ****")
                     print(f"載入的 RL 模型 '{model_path}' 輸入維度 ({loaded_input_dim})")
                     print(f"與期望的 OHLC+Holding 狀態維度 ({self.state_size}) 不匹配！")
                     print(f"請使用以 OHLC+Holding 狀態重新訓練的模型！程式可能出錯或表現異常。")
                     # Consider exiting: sys.exit(1)
                else:
                     print(f"RL 模型 '{model_path}' 成功載入，輸入維度 ({self.state_size}) 匹配。")
            except Exception as e:
                print(f"錯誤：載入 RL 模型 '{model_path}' 失敗 - {e}。將建立新模型。")
                self.model = self._build_model() # Fallback
        else:
            if is_eval and model_path:
                print(f"警告：評估模式下找不到 RL 模型 '{model_path}'。將建立新模型。")
            self.model = self._build_model() # Build new model for training

    def _build_model(self):
        """Builds the Q-network model with the updated state_size."""
        model = Sequential()
        # --- Ensure input_dim matches the new state_size ---
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        print(f"新的 RL Q-Network 模型已建立 (輸入維度: {self.state_size})。")
        return model

    # --- act, remember, replay, load, save methods ---
    # (Structurally same, but operate on states with the new dimension)
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        if state is None or not isinstance(state, np.ndarray) or state.shape != (1, self.state_size):
             print(f"錯誤 (act): 無效的狀態輸入 shape: {state.shape if state is not None else 'None'}，期望 (1, {self.state_size})。返回隨機動作。")
             return random.randrange(self.action_size)
        try:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        except Exception as e:
            print(f"錯誤 (act): 模型預測失敗 - {e}。返回隨機動作。")
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
         if not self.is_eval:
             # Ensure states have the correct shape before appending
             if state.shape == (1, self.state_size) and next_state.shape == (1, self.state_size):
                 self.memory.append((state, action, reward, next_state, done))
             else:
                 print(f"警告 (remember): 狀態 shape 不正確，經驗未存儲。 State:{state.shape}, Next:{next_state.shape}")

    def replay(self, batch_size):
        if self.is_eval or len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([t[0] for t in minibatch])
        next_states = np.vstack([t[3] for t in minibatch])

        if states.shape[1] != self.state_size or next_states.shape[1] != self.state_size:
             print(f"錯誤 (replay): 批次中狀態維度不匹配。跳過此批次。")
             return

        try:
            q_current_batch = self.model.predict(states)
            q_next_batch = self.model.predict(next_states)
            targets_batch = np.copy(q_current_batch)

            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    targets_batch[i, action] = reward
                else:
                    targets_batch[i, action] = reward + self.gamma * np.amax(q_next_batch[i])

            self.model.fit(states, targets_batch, epochs=1, verbose=0)

        except Exception as e:
            print(f"錯誤 (replay): 批次預測或擬合時發生錯誤 - {e}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        if os.path.exists(name):
            try:
                self.model = load_model(name)
                loaded_input_dim = self.model.layers[0].input_shape[1]
                if loaded_input_dim != self.state_size:
                     print(f"**** 嚴重警告 (load) ****")
                     print(f"載入的 RL 模型 '{name}' 輸入維度 ({loaded_input_dim})")
                     print(f"與期望的 OHLC+Holding 狀態維度 ({self.state_size}) 不匹配！")
                else:
                     print(f"RL 模型從 '{name}' 完整載入成功。")
            except Exception as e:
                print(f"錯誤: 無法載入 RL 模型 '{name}' - {e}")
        else:
            print(f"警告: RL 模型文件 '{name}' 不存在。")

    def save(self, name):
        try:
            self.model.save(name)
            print(f"RL 模型已儲存至 '{name}'。")
        except Exception as e:
            print(f"錯誤: 無法儲存 RL 模型至 '{name}' - {e}")

# === RL Training Function (Modified for OHLC+Holding State) ===
def training_rl_model():
    """Function to train the RL agent using the OHLC + Holding state."""
    print("--- 開始 RL 模型訓練 (使用 OHLC+Holding 狀態) ---")
    # --- Config ---
    window_size = 10    # << Time window for OHLC data
    episode_count = 50  # <<< SET TO A VALUE > 0 TO ENABLE TRAINING >>>
    batch_size = 64     # Increased batch size slightly
    # <<< CHOOSE A *NEW* FILENAME for the model trained with OHLC+Holding state >>>
    model_save_prefix = "rl_agent_ohlc_holding_state_"

    # --- Load Scaler ---
    scaler_path = LSTM_SCALER_FILENAME
    if not os.path.exists(scaler_path):
        print(f"錯誤：找不到 LSTM Scaler 文件 '{scaler_path}'。訓練終止。")
        return
    try:
        scaler = joblib.load(scaler_path)
        print(f"LSTM Scaler '{scaler_path}' 載入成功。")
    except Exception as e:
        print(f"錯誤：載入 LSTM Scaler '{scaler_path}' 失敗 - {e}。訓練終止。")
        return

    # --- Initialize Agent ---
    agent = Agent(window_size=window_size, is_eval=False) # is_eval=False for training

    # --- Get OHLC Data ---
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) <= window_size:
        print("訓練數據不足或獲取失敗，無法進行 RL 訓練。")
        return
    data_len = len(ohlc_data)
    print(f"RL 訓練數據: {ohlc_data.shape}")

    # --- Training Loop ---
    for e in range(episode_count):
        print(f"\n--- Episode {e + 1}/{episode_count} ---")
        # --- Initialize simulated holding status for the episode ---
        is_holding_simulated = False # Start episode not holding stock

        # Initial state calculation needs index window_size-1
        current_t_idx = window_size - 1
        # Get initial state with initial holding status (False -> 0.0)
        state = getState_OHLC_Holding(ohlc_data, current_t_idx, window_size, scaler, 0.0)
        if state.shape[1] != agent.state_size:
             print(f"錯誤：無法創建初始狀態 (t={current_t_idx})。跳過此回合。")
             continue

        total_profit = 0
        episode_inventory_cost = [] # Store buy costs for profit calculation

        # Loop through time steps available for state creation
        for t in range(window_size - 1, data_len - 1): # Up to second-to-last day
            # --- Determine holding status *for the current state* ---
            current_holding_float = 1.0 if is_holding_simulated else 0.0
            # We already calculated state for t=window_size-1, recalculate for subsequent t
            if t > window_size - 1:
                 state = getState_OHLC_Holding(ohlc_data, t, window_size, scaler, current_holding_float)
                 if state.shape[1] != agent.state_size:
                      print(f"錯誤：無法創建狀態 (t={t})。結束此回合。")
                      break # Exit inner loop for this episode

            # Agent chooses action based on current state (which includes holding info)
            action = agent.act(state)

            # Get prices for reward calculation
            # current_close_price = ohlc_data[t, 3]
            next_open_price = ohlc_data[t + 1, 0] # Use next day's open for cost/proceeds?
            next_close_price = ohlc_data[t + 1, 3]

            # --- Simulate action result and update holding status *for the next state* ---
            reward = 0 # Default reward
            action_taken_str = "Sit"

            if action == 1: # Try to Buy
                action_taken_str = "Buy"
                if not is_holding_simulated: # Can only buy if not already holding (simplified)
                    is_holding_simulated = True
                    episode_inventory_cost.append(next_open_price) # Assume bought at next open
                    # reward = -0.5 # Optional small penalty for transaction
                else:
                    reward = -1 # Penalty for trying to buy when already holding?

            elif action == 2: # Try to Sell
                 action_taken_str = "Sell"
                 if is_holding_simulated and episode_inventory_cost: # Can only sell if holding
                     bought_price = episode_inventory_cost.pop(0) # FIFO
                     profit = next_open_price - bought_price # Realized at next open
                     reward = profit # Reward is the profit
                     total_profit += profit
                     is_holding_simulated = False # No longer holding
                 else:
                     reward = -2 # Penalty for trying to sell when not holding

            # --- Determine holding status *for the next state* ---
            next_holding_float = 1.0 if is_holding_simulated else 0.0

            # Get next state (including the updated holding status)
            next_state = getState_OHLC_Holding(ohlc_data, t + 1, window_size, scaler, next_holding_float)
            if next_state.shape[1] != agent.state_size:
                 print(f"錯誤：無法創建下一個狀態 (t={t+1})。結束此回合。")
                 done = True
                 # Override reward?
            else:
                 done = (t == data_len - 2)

            # Store experience
            # Ensure state and next_state are valid before storing
            if state.shape[1] == agent.state_size and next_state.shape[1] == agent.state_size:
                 agent.remember(state, action, reward, next_state, done)
            else:
                 print(f"警告：經驗未存儲，因為狀態無效 (t={t})")


            state = next_state # Move to next state for the next iteration

            # Train the agent using experience replay
            agent.replay(batch_size)

            # Print progress occasionally (optional)
            # if t % 50 == 0:
            #     print(f"t={t}, Action={action_taken_str}, Reward={reward:.2f}, Holding={is_holding_simulated}")

            if done:
                # Handle any remaining inventory at the end? (e.g., liquidate at last price?)
                if is_holding_simulated and episode_inventory_cost:
                     # Optional: Calculate unrealized profit/loss
                     last_close = ohlc_data[-1, 3]
                     final_profit = last_close - episode_inventory_cost.pop(0)
                     total_profit += final_profit
                     print(f"回合結束，清算剩餘持股，額外利潤: {formatPrice(final_profit)}")

                print(f"Episode {e + 1} 結束。模擬總利潤: {formatPrice(total_profit)}")
                print(f"最終 Epsilon: {agent.epsilon:.4f}")
                break # End inner loop

        # Save model periodically
        if (e + 1) % 10 == 0 or e == episode_count - 1:
            save_filename = f"{model_save_prefix}ep{e+1}.h5"
            agent.save(save_filename) # Save the entire model

    print("--- RL 模型訓練 (OHLC+Holding 狀態) 結束 ---")


# === Trading Execution Function (Modified for OHLC+Holding State) ===
def trading_execution():
    """Main function to run trading logic using OHLC+Holding state."""
    print("\n--- 開始執行交易決策 (使用 OHLC+Holding 狀態) ---")
    # --- Config ---
    shares_to_trade = 1
    # <<< POINT TO THE *NEW* RL MODEL TRAINED WITH OHLC+Holding STATE >>>
    rl_model_load_path = "rl_agent_ohlc_holding_state_ep50.h5" # Example filename
    window_size = 10   # <<< MUST MATCH the window_size used during RL training >>>
    account = "CIoT Bot-CE"
    password = "CIoT Bot-CE"

    # --- Load Scaler ---
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

    # Initialize Agent in Eval mode - loads model and verifies dims
    agent = Agent(window_size=window_size, is_eval=True, model_path=rl_model_load_path)
    if not hasattr(agent, 'model') or agent.model.layers[0].input_shape[1] != agent.state_size:
         print("RL 模型載入或維度驗證失敗。交易終止。")
         return

    # --- Get OHLC Data ---
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) < window_size:
        print("獲取 OHLC 數據失敗或數據不足。交易終止。")
        return
    last_index = len(ohlc_data) - 1

    # --- Determine ACTUAL Holding Status ---
    print("正在查詢實際持股狀況...")
    user_stocks = Get_User_Stocks(account, password)
    actual_holding_status = 0.0 # Default to not holding
    for stock in user_stocks:
         if stock.get('stock_code') == STOCK_CODE_TO_TRADE:
             try:
                 if int(stock.get('stock_shares', 0)) > 0:
                     actual_holding_status = 1.0
                     print(f"檢測到實際持有股票 {STOCK_CODE_TO_TRADE}。")
                     break
             except (ValueError, TypeError):
                 continue # Ignore invalid stock share data
    if actual_holding_status == 0.0:
         print(f"檢測到未實際持有股票 {STOCK_CODE_TO_TRADE}。")

    # --- Create Current RL State (OHLC + Actual Holding Status) ---
    current_state = getState_OHLC_Holding(ohlc_data, last_index, window_size, scaler, actual_holding_status)
    if current_state.shape[1] != agent.state_size:
        print(f"錯誤：無法創建當前 OHLC+Holding RL 狀態 (t={last_index})。交易終止。")
        return

    # --- Get RL Action ---
    rl_action = agent.act(current_state)
    action_map = {0: "觀望 (Sit)", 1: "買入 (Buy)", 2: "賣出 (Sell)"}
    print(f"RL Agent 決策 (基於 OHLC+Holding 狀態): {rl_action} ({action_map.get(rl_action, '未知')})")

    # --- Get LSTM Price Prediction ---
    print("\n調用 LSTM 預測器 (price_prediction_v1)...")
    predicted_open, predicted_close = predict_next_open_close()
    if predicted_open is None or predicted_close is None:
        print("LSTM 價格預測失敗。交易終止。")
        return

    # --- Decide Order Price ---
    order_price = predicted_open
    print(f"LSTM 預測: Open={predicted_open:.2f}, Close={predicted_close:.2f}")
    print(f"使用預測開盤價 {order_price:.2f} 執行交易。")

    # --- Execute Trade ---
    if rl_action == 1: # Buy
        # Optional: Add check if already holding based on API?
        # if actual_holding_status == 1.0:
        #    print("RL 建議買入，但已持有股票。為避免重複買入，跳過操作。")
        # else:
        Buy_Stock(account, password, STOCK_CODE_TO_TRADE, shares_to_trade, order_price)
    elif rl_action == 2: # Sell
        # Use the already checked actual_holding_status
        if actual_holding_status == 1.0:
            Sell_Stock(account, password, STOCK_CODE_TO_TRADE, shares_to_trade, order_price)
        else:
            print(f"RL 建議賣出，但根據 API 查詢，用戶未持有股票 {STOCK_CODE_TO_TRADE}。不執行賣出。")
            # Agent should ideally not suggest this if trained properly, but safety check is good.
    else: # rl_action == 0 (Sit)
        print("RL 建議觀望，不執行交易。")

    print("\n--- 交易決策執行完畢 (OHLC+Holding 狀態) ---")


# === Main Execution Block ===
if __name__ == "__main__":
    # 1. Ensure LSTM scaler exists
    if not os.path.exists(LSTM_SCALER_FILENAME):
        print(f"找不到 LSTM Scaler '{LSTM_SCALER_FILENAME}'。嘗試運行一次 LSTM 預測來生成它...")
        predict_next_open_close()
        if not os.path.exists(LSTM_SCALER_FILENAME):
             print("仍然找不到 Scaler 文件。請檢查 price_prediction_v1.py。程式終止。")
             sys.exit(1)

    # 2. Optional: Train the RL model with the new state
    #    <<< UNCOMMENT THE LINE BELOW TO TRAIN A NEW MODEL >>>
    # training_rl_model()

    # 3. Execute the trading logic using the *newly trained* model
    trading_execution()
# -*- coding: utf-8 -*-
# stock_agent.py
# Description: RL Trading Agent simulating pre-market order/next-day confirmation.
#              Includes OHLC+Holding state, Stop-Loss, LSTM prediction,
#              and refactored training logic for delayed rewards.
# Version: 1.6 (Refactored Training Logic)

from datetime import datetime, timedelta
import os
import math
import numpy as np
import random
from collections import deque
import sys
import requests
import joblib # To load the scaler
import json   # To save/load trade info for stop-loss
import warnings
import time # For potential delays

# Suppress specific warnings if needed
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow info/warning

# --- Keras/TensorFlow Imports ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam # Use modern optimizer
    print("[Info] Using TensorFlow/Keras imports.")
except ImportError:
    try:
        from keras.models import Sequential, load_model
        from keras.layers import Dense
        from keras.optimizers import Adam # Use modern optimizer if using Keras standalone
        print("[Info] Using standalone Keras imports.")
    except ImportError:
        print("CRITICAL ERROR: Cannot import Keras or TensorFlow. Please ensure it is installed.")
        sys.exit(1)

# --- Import LSTM Predictor ---
try:
    from price_prediction_v1 import predict_next_open_close, SCALER_FILENAME as LSTM_SCALER_FILENAME
    print("[OK] Successfully imported 'price_prediction_v1.py'")
except ImportError:
    print("CRITICAL ERROR: Cannot find or import 'price_prediction_v1.py'.")
    sys.exit(1)
except AttributeError:
    print("WARNING: Cannot import SCALER_FILENAME from 'price_prediction_v1.py'. Using default.")
    LSTM_SCALER_FILENAME = "ohlc_scaler_7day_weighted.pkl"

# === Configuration ===
NUM_OHLC_FEATURES = 4
STOCK_CODE_TO_TRADE = "2330" # <<< SET YOUR TARGET STOCK CODE >>>
API_BASE_URL_STOCK = "http://140.116.86.242:8081"
API_BASE_URL_HISTORY = "http://140.116.86.241:8800"
REQUEST_TIMEOUT = 25
TRADE_INFO_FILE = f"trade_status_{STOCK_CODE_TO_TRADE}.json"
STOP_LOSS_PERCENTAGE = 5.0
# --- Account Info ---
USER_ACCOUNT = "CIoT Bot-CE"     # <<< REPLACE >>>
USER_PASSWORD = "CIoT Bot-CE"    # <<< REPLACE >>>
TEAM_ACCOUNT_ID = "team1"        # <<< REPLACE >>>
# --- Training Params --- (Used in training_rl_model)
TRAIN_INVALID_ACTION_PENALTY = -2.0 # Penalty for trying sell when not holding, etc.
TRAIN_HOLDING_REWARD_FACTOR = 0.0   # Optional: Reward for paper gains while holding (0 = disabled)

# === API Helper Functions ===
# ... (api_request, Get_Stock_Informations, Get_User_Stocks, Buy_Stock, Sell_Stock, Get_Transaction_History - Keep as in V1.5) ...
def api_request(base_url, method, endpoint, data=None, retry_attempts=2):
    url = f"{base_url}/{endpoint}"; last_exception = None
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
def Get_Stock_Informations(stock_code, start_date, stop_date):
    endpoint = f'stock/api/v1/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}'
    result = api_request(API_BASE_URL_STOCK, 'GET', endpoint)
    if result and result.get('result') == 'success': return result.get('data', [])
    status = result.get('status', 'Unknown') if result else 'NoResp'; print(f"API Get_Stock_Informations failed: {status}"); return []
def Get_User_Stocks(account, password):
    endpoint = 'stock/api/v1/get_user_stocks'; data = {'account': account, 'password': password}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result and result.get('result') == 'success': return result.get('data', [])
    status = result.get('status', 'Unknown') if result else 'NoResp'; print(f"API Get_User_Stocks failed: {status}. Returning None."); return None
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempting PRE-MARKET BUY: {stock_code} ({stock_shares} sh) @ {stock_price:.2f}')
    endpoint = 'stock/api/v1/buy'; data = {'account': account, 'password': password, 'stock_code': str(stock_code), 'stock_shares': str(stock_shares), 'stock_price': f"{stock_price:.2f}"}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result: print(f'BUY Request Result: {result.get("result", "N/A")} | Status: {result.get("status", "N/A")}') ; return result.get('result') == 'success'
    else: print('BUY Request Failed: No API response.'); return False
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempting PRE-MARKET SELL: {stock_code} ({stock_shares} sh) @ {stock_price:.2f}')
    endpoint = 'stock/api/v1/sell'; data = {'account': account, 'password': password, 'stock_code': str(stock_code), 'stock_shares': str(stock_shares), 'stock_price': f"{stock_price:.2f}"}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result: print(f'SELL Request Result: {result.get("result", "N/A")} | Status: {result.get("status", "N/A")}') ; return result.get('result') == 'success'
    else: print('SELL Request Failed: No API response.'); return False
def Get_Transaction_History(team_account_id, query_date_str):
    print(f"Querying Tx History for {team_account_id} on {query_date_str}...")
    endpoint = f'api/v1/transaction_history/{team_account_id}/{query_date_str}/{query_date_str}'
    result = api_request(API_BASE_URL_HISTORY, 'GET', endpoint)
    if result:
        if isinstance(result, list): print(f"[OK] Found {len(result)} raw tx(s) for {query_date_str}."); return result
        elif isinstance(result, dict) and result.get('result') == 'failed': print(f"WARN: Tx history query failed (API): {result.get('status', 'Unknown')}"); return []
        else: print(f"WARN: Unexpected tx history response format: {type(result)}"); return []
    else: print("ERROR: No response from tx history API."); return []

# === Stop-Loss & RL Helper Functions ===
# ... (load_trade_info, save_trade_info, formatPrice, get_ohlc_data_for_rl, getState_OHLC_Holding - Keep as in V1.5) ...
def load_trade_info(filename=TRADE_INFO_FILE):
    default_info = {STOCK_CODE_TO_TRADE: {"holding": False, "purchase_price": None, "intended_action": None}}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f: info = json.load(f)
            stock_data = info.get(STOCK_CODE_TO_TRADE);
            if not isinstance(stock_data, dict): print(f"WARN: Resetting invalid structure in '{filename}'."); return default_info
            stock_data.setdefault('holding', False); stock_data.setdefault('purchase_price', None); stock_data.setdefault('intended_action', None)
            if stock_data['purchase_price'] is not None:
                try: stock_data['purchase_price'] = float(stock_data['purchase_price'])
                except (ValueError, TypeError): stock_data['purchase_price'] = None
            stock_data['holding'] = bool(stock_data['holding']); info[STOCK_CODE_TO_TRADE] = stock_data; return info
        except (json.JSONDecodeError, Exception) as e: print(f"ERROR: Failed loading/decoding '{filename}' - {e}. Using default."); return default_info
    return default_info
def save_trade_info(trade_info, filename=TRADE_INFO_FILE):
    try:
        stock_data = trade_info.get(STOCK_CODE_TO_TRADE)
        if stock_data and stock_data.get('purchase_price') is not None:
            try: stock_data['purchase_price'] = float(stock_data['purchase_price'])
            except (ValueError, TypeError): stock_data['purchase_price'] = None
        with open(filename, 'w', encoding='utf-8') as f: json.dump(trade_info, f, indent=4, ensure_ascii=False)
    except Exception as e: print(f"ERROR: Failed saving trade info to '{filename}' - {e}")
def formatPrice(n):
    try: return ("-" if n < 0 else "") + "NT${:,.2f}".format(abs(n))
    except (TypeError, ValueError): return "N/A"
def get_ohlc_data_for_rl():
    print("RL Agent: Fetching historical OHLC data..."); features_to_get = ['Open', 'High', 'Low', 'Close']
    start_date_history = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d'); end_date_history = datetime.now().strftime('%Y%m%d')
    stock_history_data_raw = Get_Stock_Informations(STOCK_CODE_TO_TRADE, start_date_history, end_date_history)
    if not stock_history_data_raw: print("ERROR (get_ohlc_data): Failed fetch."); return None
    ohlc_data = []; dates = []
    for i, history in enumerate(stock_history_data_raw):
        if all(key in history for key in features_to_get):
            try: ohlc_data.append([float(history[key]) for key in features_to_get]); dates.append(history.get('Date'))
            except (ValueError, TypeError): print(f"WARN (get_ohlc_data): Skip invalid numeric data idx {i}.")
    if not ohlc_data: print("ERROR (get_ohlc_data): No valid OHLC records."); return None
    ohlc_data = np.array(ohlc_data, dtype=np.float32); print(f"[OK] Processed {ohlc_data.shape[0]} OHLC days. Last: {dates[-1] if dates else 'N/A'}"); return ohlc_data
def getState_OHLC_Holding(ohlc_data, t_index, window_size, scaler, holding_status):
    n_ohlc_features = NUM_OHLC_FEATURES; expected_state_dim = window_size * n_ohlc_features + 1
    fallback_state = np.zeros((1, expected_state_dim))
    if scaler is None or not hasattr(scaler, 'transform'): print("ERROR (getState): Invalid Scaler."); return fallback_state
    holding_status = float(holding_status) if holding_status in [0.0, 1.0] else 0.0; fallback_state[0, -1] = holding_status
    start_index = t_index - window_size + 1; end_index = t_index + 1
    if start_index < 0: return fallback_state # Not enough history
    if end_index > len(ohlc_data): print(f"ERROR (getState): Index out of bounds."); return fallback_state
    window_data = ohlc_data[start_index : end_index, :]
    if window_data.shape != (window_size, n_ohlc_features): print(f"ERROR (getState): Window shape mismatch."); return fallback_state
    try: scaled_window = scaler.transform(window_data)
    except Exception as e: print(f"ERROR (getState): Scaler transform failed - {e}"); return fallback_state
    ohlc_state_part = scaled_window.flatten(); holding_feature = np.array([holding_status], dtype=np.float32)
    full_state_flat = np.concatenate((ohlc_state_part, holding_feature)); state = full_state_flat.reshape(1, -1)
    if state.shape[1] != expected_state_dim: print(f"ERROR (getState): Final state shape mismatch."); return fallback_state
    return state

# === Reinforcement Learning Agent Class ===
class Agent:
    """DQN Agent using OHLC+Holding state."""
    def __init__(self, window_size, is_eval=False, model_path=""):
        self.state_size = window_size * NUM_OHLC_FEATURES + 1; self.window_size = window_size
        self.action_size = 3; self.memory = deque(maxlen=5000) # Larger memory?
        self.model_path = model_path; self.is_eval = is_eval; self.gamma = 0.95
        self.epsilon = 0.0 if is_eval else 1.0; self.epsilon_min = 0.01; self.epsilon_decay = 0.995
        self.model = self._load_or_build_model(model_path)
    def _load_or_build_model(self, model_path):
        if self.is_eval and model_path and os.path.exists(model_path):
            print(f"Attempting to load RL model: {model_path}")
            try:
                model = load_model(model_path, compile=True); loaded_input_dim = model.layers[0].input_shape[-1]
                if loaded_input_dim != self.state_size: print("*"*20+" CRITICAL WARNING "+"*"*20+f"\nLoaded RL model input dim ({loaded_input_dim}) != expected ({self.state_size})!\n"+"*"*65); sys.exit("Incompatible RL model.")
                else: print(f"[OK] RL model '{model_path}' loaded successfully."); return model
            except Exception as e: print(f"ERROR: Failed loading RL model '{model_path}' - {e}."); sys.exit("Failed loading model in eval mode.")
        elif self.is_eval and model_path: print(f"ERROR: Specified RL model '{model_path}' not found. Exiting."); sys.exit(1)
        return self._build_model() # Build new if training
    def _build_model(self):
        model = Sequential([Dense(128, input_dim=self.state_size, activation='relu'), Dense(64, activation='relu'), Dense(self.action_size, activation='linear')])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001)); print(f"[OK] New RL Q-Network built (Input Dim: {self.state_size})."); return model
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        if state is None or state.shape != (1, self.state_size): print(f"ERROR (act): Invalid state shape. Random action."); return random.randrange(self.action_size)
        try:
            with np.errstate(all='ignore'): act_values = self.model.predict_on_batch(state)
            if np.isnan(act_values).any() or np.isinf(act_values).any(): print("WARN (act): Model prediction NaN/Inf. Random action."); return random.randrange(self.action_size)
            return np.argmax(act_values[0])
        except Exception as e: print(f"ERROR (act): Model prediction failed - {e}. Random action."); return random.randrange(self.action_size)
    def remember(self, state, action, reward, next_state, done):
         if not self.is_eval:
             if state is not None and next_state is not None and state.shape == (1, self.state_size) and next_state.shape == (1, self.state_size):
                 self.memory.append((state, action, reward, next_state, done))
    def replay(self, batch_size):
         if self.is_eval or len(self.memory) < batch_size: return
         minibatch = random.sample(self.memory, batch_size); states = np.vstack([t[0] for t in minibatch]); next_states = np.vstack([t[3] for t in minibatch])
         if states.shape[1] != self.state_size or next_states.shape[1] != self.state_size: print(f"ERROR (replay): Batch state dimension mismatch."); return
         try:
             q_current_batch = self.model.predict_on_batch(states); q_next_batch = self.model.predict_on_batch(next_states)
             targets_batch = np.copy(q_current_batch)
             for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                 if done or np.isnan(q_next_batch[i]).any() or np.isinf(q_next_batch[i]).any(): targets_batch[i, action] = reward
                 else: targets_batch[i, action] = reward + self.gamma * np.amax(q_next_batch[i])
             valid_targets = ~np.isnan(targets_batch).any(axis=1) & ~np.isinf(targets_batch).any(axis=1)
             if not valid_targets.all(): states = states[valid_targets]; targets_batch = targets_batch[valid_targets]
             if states.shape[0] > 0: self.model.train_on_batch(states, targets_batch)
         except Exception as e: print(f"ERROR (replay): Batch processing/training failed - {e}")
         if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save(self, name):
        if hasattr(self, 'model') and self.model is not None:
            try: self.model.save(name); print(f"RL model saved to '{name}'.")
            except Exception as e: print(f"ERROR: Failed saving RL model to '{name}' - {e}")
        else: print("ERROR (save): No model instance to save.")

# === RL Training Function (Refactored for Delayed Confirmation Simulation) ===
def training_rl_model():
    """
    Trains the RL agent simulating the pre-market order / next-day confirmation constraint.
    Rewards are calculated based on the outcome observed *after* the simulated day's market.
    """
    print("\n" + "#"*60)
    print("#" + " "*10 + "RL 模型訓練 (模擬盤前下單/隔日確認延遲)" + " "*10 + "#")
    print("#"*60)

    # --- Configuration ---
    window_size = 10        # State window (must match Agent init)
    episode_count = 100     # <<< SET TO NUMBER OF TRAINING EPISODES >>> (e.g., 50, 100, 500)
    batch_size = 64
    # <<< CHOOSE A *NEW* FILENAME for the correctly trained model >>>
    model_save_prefix = f"rl_agent_{STOCK_CODE_TO_TRADE}_ohlc_hold_delay_"

    # --- Load Scaler ---
    print("[Train] Loading LSTM Scaler...")
    scaler_path = LSTM_SCALER_FILENAME
    if not os.path.exists(scaler_path): print(f"ERROR: Scaler file '{scaler_path}' not found."); return
    try: scaler = joblib.load(scaler_path); print(f"[OK] Scaler '{scaler_path}' loaded.")
    except Exception as e: print(f"ERROR: Failed loading Scaler '{scaler_path}' - {e}."); return

    # --- Initialize Agent for Training ---
    print("[Train] Initializing RL Agent for training...")
    agent = Agent(window_size=window_size, is_eval=False) # is_eval=False

    # --- Get OHLC Data ---
    print("[Train] Fetching OHLC data...")
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) <= window_size + 1: print("ERROR: Insufficient OHLC data."); return
    data_len = len(ohlc_data)
    print(f"[OK] Using {data_len} days of OHLC data for training.")

    # --- Training Loop ---
    print(f"[Train] Starting training for {episode_count} episodes...")
    start_time_train = time.time()
    max_total_profit = -float('inf') # Track best episode profit

    for e in range(episode_count):
        episode_start_time = time.time()
        print(f"\n--- Episode {e + 1}/{episode_count} ---")

        # --- Episode State ---
        # holding status *at the end* of day t (after market) -> determines state s_t
        is_holding_episode = False
        purchase_price_episode = None # Stores simulated buy price
        total_profit = 0.0
        trade_count = 0

        # Initial state s_0 uses data up to t=window_size-1, holding=False
        initial_t_idx = window_size - 1
        state = getState_OHLC_Holding(ohlc_data, initial_t_idx, window_size, scaler, 0.0)
        if state.shape[1] != agent.state_size: print(f"ERROR: Cannot create initial state. Skipping episode."); continue

        # --- Simulate daily steps ---
        # Loop from the first day we can make a decision (t=window_size-1)
        # up to the second-to-last day (t=data_len-2), because we need day t+1's data
        for t in range(initial_t_idx, data_len - 1):
            # 1. Decide Action (a_t) for *next* day (t+1) based on current state (s_t)
            action = agent.act(state)

            # 2. Simulate Market Day t+1 & Determine Outcome
            open_t1 = ohlc_data[t + 1, 0]
            high_t1 = ohlc_data[t + 1, 1]
            low_t1  = ohlc_data[t + 1, 2]
            close_t1= ohlc_data[t + 1, 3]

            # --- Simulate Fill & Calculate Reward (r_t) & Next Holding (h_{t+1}) ---
            reward = 0.0
            next_day_holding = is_holding_episode # Assume holding status carries over unless changed
            simulated_fill_price = None

            if action == 1: # Intend to Buy on day t+1
                if not is_holding_episode:
                    # Simulate buy fill: Assume order at open_t1, check if possible
                    simulated_buy_price = open_t1 # Simple: assume fills at open
                    if low_t1 <= simulated_buy_price <= high_t1: # Check if open was in range
                        next_day_holding = True
                        purchase_price_episode = simulated_buy_price # Record cost basis
                        reward = 0.0 # Or optional paper gain: TRAIN_HOLDING_REWARD_FACTOR * (close_t1 - simulated_buy_price)
                        simulated_fill_price = simulated_buy_price
                        trade_count += 1
                    # else: print(f"Debug ep{e+1},t={t}: Buy order @{simulated_buy_price} failed (Out of range {low_t1}-{high_t1})") # Optional debug
                else: # Invalid action: Buy when holding
                    reward = TRAIN_INVALID_ACTION_PENALTY
            elif action == 2: # Intend to Sell on day t+1
                if is_holding_episode and purchase_price_episode is not None:
                    # Simulate sell fill: Assume order at open_t1, check if possible
                    simulated_sell_price = open_t1 # Simple: assume fills at open
                    if low_t1 <= simulated_sell_price <= high_t1:
                        profit = simulated_sell_price - purchase_price_episode
                        reward = profit # Reward is realized P/L
                        total_profit += profit
                        next_day_holding = False # Sold
                        purchase_price_episode = None # Reset cost basis
                        simulated_fill_price = simulated_sell_price
                        trade_count += 1
                    # else: print(f"Debug ep{e+1},t={t}: Sell order @{simulated_sell_price} failed (Out of range {low_t1}-{high_t1})") # Optional debug
                else: # Invalid action: Sell when not holding
                    reward = TRAIN_INVALID_ACTION_PENALTY
            else: # action == 0 (Sit/Hold)
                 if is_holding_episode and purchase_price_episode is not None:
                      # Optional reward for holding based on price movement
                      paper_change = close_t1 - ohlc_data[t, 3] # Change from prev close to current close
                      reward = paper_change * TRAIN_HOLDING_REWARD_FACTOR
                 # else: reward = 0 # Sitting while not holding = 0 reward

            # --- Update holding status based on simulation outcome ---
            is_holding_episode = next_day_holding

            # 3. Generate Next State (s_{t+1}) using info up to t+1
            next_state = getState_OHLC_Holding(ohlc_data, t + 1, window_size, scaler, float(is_holding_episode))
            if next_state.shape[1] != agent.state_size:
                 print(f"ERROR: Cannot create next state for t={t+1}. Ending episode.")
                 done = True; reward += -10 # Add penalty if state fails
            else:
                 done = (t == data_len - 2) # Episode ends if we are at the last possible step

            # 4. Store Experience (s_t, a_t, r_t, s_{t+1}, done)
            agent.remember(state, action, reward, next_state, done)

            # 5. Update State for Next Iteration
            state = next_state

            # 6. Perform Experience Replay (less frequently?)
            if t % 5 == 0 and len(agent.memory) > batch_size: # Replay every 5 steps
                 agent.replay(batch_size)

            if done: break # End inner loop if episode finished

        # --- End of Episode ---
        episode_duration = time.time() - episode_start_time
        print(f"Episode {e + 1} finished. Duration: {episode_duration:.2f}s | "
              f"Profit: {formatPrice(total_profit)} | Trades: {trade_count} | Epsilon: {agent.epsilon:.4f}")
        if total_profit > max_total_profit: max_total_profit = total_profit

        # Save model periodically and potentially based on performance improvement
        if (e + 1) % 20 == 0 or e == episode_count - 1 or total_profit >= max_total_profit * 0.95 : # Save every 20 eps, last ep, or near best
            save_filename = f"{model_save_prefix}ep{e+1}.h5"
            agent.save(save_filename)

    # --- End of Training ---
    total_training_time = time.time() - start_time_train
    print("\n" + "="*60); print(f"RL 模型訓練完成。總耗時: {total_training_time:.2f} 秒"); print("="*60 + "\n")

# === Trading Execution Function (Integrates History, Stop-Loss, SimDayTrade) ===
# --- (trading_execution function remains the same as in V1.5 - uses the correctly trained model) ---
def trading_execution():
    """Executes pre-market decisions using history, stop-loss, and RL."""
    print("\n" + "="*60); print("=" + " "*10 + "開始執行【盤前】交易決策 (整合歷史/止損)" + " "*10 + "="); print("="*60)
    # --- Config ---
    shares_to_trade = 1
    # <<< CRITICAL: Path to RL model TRAINED with DELAYED REWARD logic >>>
    rl_model_load_path = "rl_agent_ohlc_holding_delayed_ep100.h5" # <<< UPDATE THIS FILENAME (Example)
    window_size = 10
    user_account_stock = USER_ACCOUNT; user_password_stock = USER_PASSWORD
    team_account_history = TEAM_ACCOUNT_ID
    day_trade_sell_price_method = DAY_TRADE_SELL_METHOD
    day_trade_profit_target = DAY_TRADE_PROFIT_TARGET_PERCENT

    # --- 1. Load Dependencies ---
    print("\n[Step 1/11] 載入必要文件...")
    try: scaler = joblib.load(LSTM_SCALER_FILENAME); print(f"[OK] Scaler '{LSTM_SCALER_FILENAME}' loaded.")
    except Exception as e: print(f"ERROR: Failed loading Scaler - {e}."); return
    if not os.path.exists(rl_model_load_path): print(f"ERROR: RL model missing '{rl_model_load_path}'."); return
    agent = Agent(window_size=window_size, is_eval=True, model_path=rl_model_load_path) # Loads and validates model
    if not hasattr(agent, 'model') or agent.model.layers[0].input_shape[-1] != agent.state_size: print("ERROR: RL model validation failed."); return
    print("[OK] RL Agent initialized.")
    trade_info = load_trade_info(); last_run_info = trade_info.get(STOCK_CODE_TO_TRADE, {"holding": False, "purchase_price": None, "intended_action": None})
    last_run_intended_action = last_run_info.get("intended_action"); last_run_purchase_price = last_run_info.get("purchase_price")
    print(f"[OK] Loaded Prev Trade Info: Intended={last_run_intended_action}, Cost={last_run_purchase_price}")

    # --- 2. Get Yesterday's Date ---
    print("\n[Step 2/11] 確定昨日日期...")
    yesterday = datetime.now() - timedelta(days=1); yesterday_str = yesterday.strftime('%Y%m%d') # TODO: Handle weekends/holidays
    print(f"[OK] Yesterday's date for history query: {yesterday_str}")

    # --- 3. Query Yesterday's Transaction History ---
    print("\n[Step 3/11] 查詢昨日交易歷史...")
    transactions_yesterday = Get_Transaction_History(team_account_history, yesterday_str)
    relevant_transactions = [ t for t in transactions_yesterday if t.get("stock_code") == STOCK_CODE_TO_TRADE and t.get("state") == "交易成功" ]
    print(f"[OK] Found {len(relevant_transactions)} relevant successful transaction(s) yesterday.")

    # --- 4. Reconcile Trade Info using History ---
    print("\n[Step 4/11] 核對交易記錄文件與昨日歷史...")
    current_purchase_price = None; reconciled_holding = last_run_info.get("holding", False)
    # ... (Reconciliation Logic using last_run_intended_action and relevant_transactions - same as V1.5) ...
    if last_run_intended_action == "buy":
        successful_buy = next((t for t in relevant_transactions if t.get("type") == "買進"), None)
        if successful_buy:
            try: actual_buy_price = float(successful_buy["price"]); print(f"   [CONFIRMED] Yesterday's BUY successful @ {actual_buy_price:.2f}"); reconciled_holding = True; current_purchase_price = actual_buy_price
            except (ValueError, KeyError, TypeError): print(f"   ERROR: Could not parse price from buy tx: {successful_buy}"); reconciled_holding = True; current_purchase_price = None
        else: print("   [FAILED] Yesterday's BUY intention failed."); reconciled_holding = False; current_purchase_price = None
    elif last_run_intended_action == "sell":
        successful_sell = next((t for t in relevant_transactions if t.get("type") == "賣出"), None)
        if successful_sell: print(f"   [CONFIRMED] Yesterday's SELL successful."); reconciled_holding = False; current_purchase_price = None
        else: print("   [FAILED] Yesterday's SELL intention failed."); reconciled_holding = True; current_purchase_price = last_run_purchase_price
    else: reconciled_holding = last_run_info.get("holding", False); current_purchase_price = last_run_purchase_price; print("   No buy/sell intention yesterday.")
    trade_info[STOCK_CODE_TO_TRADE] = {"holding": reconciled_holding, "purchase_price": current_purchase_price, "intended_action": None}
    save_trade_info(trade_info); print(f"   Reconciled Status Saved: Holding={reconciled_holding}, CostBasis={current_purchase_price}")


    # --- 5. Get ACTUAL Current Holding Status (Final Check) ---
    print("\n[Step 5/11] 查詢【當前】實際持股狀況 (最終確認)...")
    user_stocks = Get_User_Stocks(user_account_stock, user_password_stock)
    actual_holding_status_float = 0.0; actual_shares_held = 0; final_confirmed_holding = False
    # ... (Logic to check user_stocks and set final_confirmed_holding - same as V1.5) ...
    if user_stocks is not None:
        for stock in user_stocks:
             if stock.get('stock_code') == STOCK_CODE_TO_TRADE:
                 try: actual_shares_held = int(stock.get('stock_shares', 0));
                 except (ValueError, TypeError): continue
                 if actual_shares_held > 0 : actual_holding_status_float = 1.0; final_confirmed_holding = True; break
    else: print("WARN: Failed querying user stocks for final check.")
    # Final sync check
    if final_confirmed_holding != reconciled_holding:
        print("*"*20 + " SYNC WARNING " + "*"*20); print(f"API holding ({final_confirmed_holding}) != reconciled file ({reconciled_holding})! API overrides."); print("*"*60)
        trade_info[STOCK_CODE_TO_TRADE]['holding'] = final_confirmed_holding
        if not final_confirmed_holding: trade_info[STOCK_CODE_TO_TRADE]['purchase_price'] = None
        elif final_confirmed_holding and not reconciled_holding: trade_info[STOCK_CODE_TO_TRADE]['purchase_price'] = None # Cost basis lost
        current_purchase_price = trade_info[STOCK_CODE_TO_TRADE]['purchase_price']
        save_trade_info(trade_info)
    print(f"[OK] Final Confirmed Status: Holding={final_confirmed_holding}, Shares={actual_shares_held}, Cost Basis={current_purchase_price}")

    # --- 6. Get Historical OHLC Data ---
    print("\n[Step 6/11] 獲取 OHLC 數據 (用於狀態生成)...")
    ohlc_data = get_ohlc_data_for_rl()
    if ohlc_data is None or len(ohlc_data) < window_size: print("ERROR: Failed getting OHLC data."); return
    last_available_data_index = len(ohlc_data) - 1
    last_close_price = ohlc_data[last_available_data_index, 3]

    # --- 7. Pre-Market Stop-Loss Check ---
    print("\n[Step 7/11] 執行盤前止損檢查...")
    stop_loss_triggered = False
    # ... (Stop-loss check logic - same as V1.5) ...
    if final_confirmed_holding and current_purchase_price is not None:
        stop_loss_price = current_purchase_price * (1 - STOP_LOSS_PERCENTAGE / 100.0)
        print(f"   Holding, Confirmed Cost: {current_purchase_price:.2f}, Stop-Loss Trigger: {stop_loss_price:.2f}")
        if last_close_price < stop_loss_price: print(f"   !!! STOP-LOSS TRIGGERED !!!"); stop_loss_triggered = True
        else: print(f"   Stop-loss not triggered.")
    else: print("   Not holding or cost basis unknown. Skipping stop-loss check.")

    # --- 8. Generate RL State ---
    print("\n[Step 8/11] 生成當前 RL 狀態...")
    current_state = getState_OHLC_Holding(ohlc_data, last_available_data_index, window_size, scaler, float(final_confirmed_holding))
    if current_state.shape[1] != agent.state_size: print(f"ERROR: Failed creating RL state."); return

    # --- 9. Get RL Action & Determine Final Action ---
    print("\n[Step 9/11] 獲取 RL 決策並確定最終動作...")
    rl_action = agent.act(current_state)
    action_map = {0: "觀望 (Sit)", 1: "買入 (Buy)", 2: "賣出 (Sell)"}
    print(f"   RL Agent Decision: {rl_action} ({action_map.get(rl_action, 'Unknown')})")
    final_action = rl_action; action_reason = f"RL Agent ({action_map.get(rl_action, 'Unknown')})"
    if stop_loss_triggered:
        if final_action != 2: print(f"   ACTION OVERRIDE: Stop-Loss triggered. Forcing SELL."); final_action = 2
        action_reason = "Stop-Loss Triggered" + (" & RL Agent" if rl_action == 2 else "")
    print(f"   Final Action Determined: {final_action} ({action_reason})")

    # --- 10. Get LSTM Prediction & Set Order Prices ---
    print("\n[Step 10/11] 獲取 LSTM 預測並設定下單價格...")
    predicted_open, predicted_close = predict_next_open_close()
    if predicted_open is None or predicted_close is None: print("ERROR: LSTM prediction failed."); return
    buy_order_price = predicted_open
    sell_order_price = predicted_open # For SL/RL sell
    if DAY_TRADE_SELL_METHOD == 'profit_target' and DAY_TRADE_PROFIT_TARGET_PERCENT is not None: day_trade_sell_price = buy_order_price * (1 + DAY_TRADE_PROFIT_TARGET_PERCENT / 100.0)
    else: day_trade_sell_price = predicted_close
    print(f"[OK] LSTM Predicted Today: Open={predicted_open:.2f}, Close={predicted_close:.2f}")
    print(f"    Order Prices -> Buy: {buy_order_price:.2f} | Sell (SL/RL): {sell_order_price:.2f} | Sell (DayTrade): {day_trade_sell_price:.2f}")

    # --- 11. Execute Pre-Market Orders & Save Intention ---
    print(f"\n[Step 11/11] 根據最終動作 ({final_action} - {action_reason}) 執行預約單操作...")
    intended_action_today = "sit"; buy_order_sent = False; sell_order_sent = False
    # ... (Order placement logic and intention saving - same as V1.5) ...
    if final_action == 1: # Buy
        if final_confirmed_holding: print("INFO: Final Action is BUY, but already holding. No action.")
        else:
            print(f"ACTION: Placing PRE-MARKET BUY order @ {buy_order_price:.2f}")
            buy_success = Buy_Stock(user_account_stock, user_password_stock, STOCK_CODE_TO_TRADE, shares_to_trade, buy_order_price)
            if buy_success:
                buy_order_sent = True; intended_action_today = "buy"
                print(f"ACTION: Attempting Day Trade - Placing PRE-MARKET SELL order @ {day_trade_sell_price:.2f}")
                sell_success_daytrade = Sell_Stock(user_account_stock, user_password_stock, STOCK_CODE_TO_TRADE, shares_to_trade, day_trade_sell_price)
                if sell_success_daytrade: sell_order_sent = True
                else: print(f"   WARN: Failed placing day trade sell order request.")
            else: print("   WARN: Failed placing buy order request.")
    elif final_action == 2: # Sell
        if final_confirmed_holding:
             if actual_shares_held >= shares_to_trade:
                 print(f"ACTION: Placing PRE-MARKET SELL order @ {sell_order_price:.2f} (Reason: {action_reason})")
                 sell_success = Sell_Stock(user_account_stock, user_password_stock, STOCK_CODE_TO_TRADE, shares_to_trade, sell_order_price)
                 if sell_success: sell_order_sent = True; intended_action_today = "sell"
                 else: print("   WARN: Failed placing sell order request.")
             else: print(f"INFO: Action is SELL, but holding ({actual_shares_held}) < required ({shares_to_trade}). No action.")
        else: print(f"INFO: Action is SELL, but not holding. No action taken.")
    else: # Sit
        print("INFO: Final action is SIT. No orders placed.")
        intended_action_today = "sit"
    # Save intention
    final_holding_intended = (intended_action_today == "buy") or (final_confirmed_holding and intended_action_today != "sell")
    final_purchase_price_intended = buy_order_price if intended_action_today == "buy" else (current_purchase_price if final_holding_intended else None)
    trade_info[STOCK_CODE_TO_TRADE] = {"holding": final_holding_intended, "purchase_price": final_purchase_price_intended, "intended_action": intended_action_today}
    print(f"\nSaving Trade Intention: Holding={final_holding_intended}, PurchasePrice={final_purchase_price_intended}, Action={intended_action_today}")
    save_trade_info(trade_info)

    print("\n" + "="*60); print("=" + " "*14 + "【盤前】決策與預約單處理完成" + " "*14 + "="); print("=" + " "*9 + "實際交易結果需在明日開市後或收盤後查詢確認" + " "*9 + "="); print("="*60 + "\n")

# === Main Execution Block ===
if __name__ == "__main__":
    print("="*40); print("   股票交易機器人 (v1.6 - 延遲訓練/歷史確認)"); print("="*40)
    start_run_time = datetime.now()
    print(f"啟動時間: {start_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易目標: {STOCK_CODE_TO_TRADE}"); print(f"止損比例: {STOP_LOSS_PERCENTAGE}%")

    # 1. Ensure Scaler file exists
    print("\n檢查依賴文件 (Scaler)...")
    if not os.path.exists(LSTM_SCALER_FILENAME): print(f"\nCRITICAL ERROR: LSTM Scaler file '{LSTM_SCALER_FILENAME}' not found."); sys.exit(1)
    else: print(f"[OK] Found Scaler file: '{LSTM_SCALER_FILENAME}'")

    # 2. Optional: Execute Training (Needs correct implementation)
    # print("\n檢查是否執行 RL 訓練...")
    # training_rl_model() # <<< RUN THIS ONLY AFTER REFACTORING IT CORRECTLY!

    # 3. Execute Pre-Market Trading Logic
    print("\n執行盤前交易決策流程...")
    trading_execution()

    end_run_time = datetime.now()
    print(f"\n程式執行完畢: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總耗時: {end_run_time - start_run_time}")
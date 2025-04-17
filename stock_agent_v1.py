# -*- coding: utf-8 -*-
# stock_agent.py
# Description: Multi-Stock RL Agent for pre-market orders (Fixed List),
#              with History Check, Stop-Loss, OHLC+Holding State, LSTM Prediction.
# Version: 1.9 (Fixed Stock List)

from datetime import datetime, timedelta
import os
import math
import numpy as np
import random
from collections import deque
import sys
import requests
import joblib
import json
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Keras/TensorFlow Imports ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    print("[Info] Using TensorFlow/Keras imports.")
except ImportError:
    try:
        from keras.models import Sequential, load_model
        from keras.layers import Dense
        from keras.optimizers import Adam
        print("[Info] Using standalone Keras imports.")
    except ImportError:
        print("CRITICAL ERROR: Cannot import Keras or TensorFlow."); sys.exit(1)

# --- Import LSTM Predictor ---
try:
    # Assumes price_prediction_v1.py returns (open, close, scaler)
    from price_prediction_v1 import predict_next_open_close
    print("[OK] Successfully imported 'price_prediction_v1.py'")
    # We no longer import SCALER_FILENAME here, get scaler from return value
except ImportError:
    print("CRITICAL ERROR: Cannot find or import 'price_prediction_v1.py'."); sys.exit(1)

# === Configuration ===
NUM_OHLC_FEATURES = 4
STOCK_LIST_FILE = "stock_list.txt" # <<< USE THIS FILE FOR FIXED LIST >>>
API_BASE_URL_STOCK = "http://140.116.86.242:8081"
API_BASE_URL_HISTORY = "http://140.116.86.241:8800"
REQUEST_TIMEOUT = 25
TRADE_INFO_FILE = f"trade_status_multi_fixed.json" # Status file for fixed list
STOP_LOSS_PERCENTAGE = 5.0
# --- Account Info ---
USER_ACCOUNT = "CIoT Bot-CE"     # <<< REPLACE >>>
USER_PASSWORD = "CIoT Bot-CE"    # <<< REPLACE >>>
TEAM_ACCOUNT_ID = "team1"        # <<< REPLACE >>>
# --- RL Model (Generic) ---
RL_MODEL_LOAD_PATH = "rl_agent_generic_ohlc_holding_delayed.h5" # <<< UPDATE >>>
WINDOW_SIZE = 10
# --- Trading Params ---
SHARES_TO_TRADE = 1
DAY_TRADE_SELL_METHOD = 'predicted_close'
DAY_TRADE_PROFIT_TARGET_PERCENT = 2.0

# === Function to Load Fixed Stock List ===
def load_fixed_stock_list(filename=STOCK_LIST_FILE):
    """Reads stock codes (comma-separated) from the specified file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.startswith('\ufeff'): content = content[1:] # Handle BOM
            stock_codes = [code.strip() for code in content.split(',') if code.strip().isdigit() and len(code.strip()) == 4]
            if not stock_codes: print(f"ERROR: No valid 4-digit stock codes found in '{filename}'."); return None
            print(f"[OK] Loaded {len(stock_codes)} fixed stock codes from '{filename}'.")
            return stock_codes
    except FileNotFoundError: print(f"CRITICAL ERROR: Fixed stock list file '{filename}' not found."); return None
    except Exception as e: print(f"CRITICAL ERROR: Failed reading fixed stock list '{filename}' - {e}"); return None

# === API Helper Functions ===
# ... (api_request, Get_Stock_Informations, Get_User_Stocks, Buy_Stock, Sell_Stock, Get_Transaction_History - Keep as before) ...
# --- (Copy these functions from the previous complete code block here) ---
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
    endpoint = f'stock/api/v1/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}'; result = api_request(API_BASE_URL_STOCK, 'GET', endpoint)
    if result and result.get('result') == 'success': return result.get('data', [])
    status = result.get('status', 'Unknown') if result else 'NoResp'; # print(f"API Get_Stock_Informations failed for {stock_code}: {status}"); # Reduce noise
    return []
def Get_User_Stocks(account, password):
    endpoint = 'stock/api/v1/get_user_stocks'; data = {'account': account, 'password': password}; result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result and result.get('result') == 'success': return result.get('data', [])
    status = result.get('status', 'Unknown') if result else 'NoResp'; print(f"API Get_User_Stocks failed: {status}. Returning None."); return None
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempting PRE-MARKET BUY: {stock_code} ({stock_shares} sh) @ {stock_price:.2f}'); endpoint = 'stock/api/v1/buy'; data = {'account': account, 'password': password, 'stock_code': str(stock_code), 'stock_shares': str(stock_shares), 'stock_price': f"{stock_price:.2f}"}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result: print(f'BUY Request Result: {result.get("result", "N/A")} | Status: {result.get("status", "N/A")}') ; return result.get('result') == 'success'
    else: print('BUY Request Failed: No API response.'); return False
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempting PRE-MARKET SELL: {stock_code} ({stock_shares} sh) @ {stock_price:.2f}'); endpoint = 'stock/api/v1/sell'; data = {'account': account, 'password': password, 'stock_code': str(stock_code), 'stock_shares': str(stock_shares), 'stock_price': f"{stock_price:.2f}"}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data)
    if result: print(f'SELL Request Result: {result.get("result", "N/A")} | Status: {result.get("status", "N/A")}') ; return result.get('result') == 'success'
    else: print('SELL Request Failed: No API response.'); return False
def Get_Transaction_History(team_account_id, query_date_str):
    print(f"Querying Tx History for {team_account_id} on {query_date_str}..."); endpoint = f'api/v1/transaction_history/{team_account_id}/{query_date_str}/{query_date_str}'
    result = api_request(API_BASE_URL_HISTORY, 'GET', endpoint)
    if result:
        if isinstance(result, list): print(f"[OK] Found {len(result)} raw tx(s) for {query_date_str}."); return result
        elif isinstance(result, dict) and result.get('result') == 'failed': print(f"WARN: Tx history query failed (API): {result.get('status', 'Unknown')}"); return []
        else: print(f"WARN: Unexpected tx history response format: {type(result)}"); return []
    else: print("ERROR: No response from tx history API."); return []


# === Stop-Loss & Trade Info Helpers ===
# ... (load_trade_info, save_trade_info - Keep as before, handle multi-stock dict) ...
def load_trade_info(filename=TRADE_INFO_FILE):
    default_info = {} # Default is empty dict for multi-stock
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f: info = json.load(f)
            if not isinstance(info, dict): print(f"WARN: Invalid format in '{filename}'. Resetting."); return {}
            validated_info = {}
            for stock_code, stock_data in info.items():
                 if isinstance(stock_data, dict):
                     stock_data.setdefault('holding', False); stock_data.setdefault('purchase_price', None); stock_data.setdefault('intended_action', None)
                     if stock_data['purchase_price'] is not None:
                         try: stock_data['purchase_price'] = float(stock_data['purchase_price'])
                         except (ValueError, TypeError): stock_data['purchase_price'] = None
                     stock_data['holding'] = bool(stock_data['holding']); validated_info[stock_code] = stock_data
            return validated_info
        except (json.JSONDecodeError, Exception) as e: print(f"ERROR: Failed loading/decoding '{filename}' - {e}. Returning empty."); return {}
    return default_info
def save_trade_info(trade_info, filename=TRADE_INFO_FILE):
    try:
        for stock_code, stock_data in trade_info.items():
             if stock_data and stock_data.get('purchase_price') is not None:
                 try: stock_data['purchase_price'] = float(stock_data['purchase_price'])
                 except (ValueError, TypeError): stock_data['purchase_price'] = None
        with open(filename, 'w', encoding='utf-8') as f: json.dump(trade_info, f, indent=4, ensure_ascii=False)
    except Exception as e: print(f"ERROR: Failed saving trade info to '{filename}' - {e}")


# === RL Helper Functions ===
# ... (formatPrice - Keep as before) ...
# ... (get_ohlc_data_for_stock - Keep as before) ...
# ... (getState_OHLC_Holding - Keep as before, requires scaler) ...
def formatPrice(n): try: return ("-" if n < 0 else "") + "NT${:,.2f}".format(abs(n)); except: return "N/A"
def get_ohlc_data_for_stock(stock_code):
    print(f"RL Agent: Fetching OHLC data for {stock_code}..."); features_to_get=['Open','High','Low','Close']
    start_date=(datetime.now()-timedelta(days=730)).strftime('%Y%m%d'); end_date=datetime.now().strftime('%Y%m%d')
    stock_history_data_raw = Get_Stock_Informations(stock_code, start_date, end_date)
    if not stock_history_data_raw: print(f"ERROR (get_ohlc {stock_code}): Failed fetch."); return None
    ohlc_data = []; dates = []
    for i,h in enumerate(stock_history_data_raw):
        if all(k in h for k in features_to_get):
            try: ohlc_data.append([float(h[k]) for k in features_to_get]); dates.append(h.get('Date'))
            except: pass # Skip invalid
    if not ohlc_data: print(f"ERROR (get_ohlc {stock_code}): No valid OHLC records."); return None
    ohlc_data = np.array(ohlc_data, dtype=np.float32); # print(f"[OK] Processed {ohlc_data.shape[0]} OHLC for {stock_code}.")
    return ohlc_data
def getState_OHLC_Holding(ohlc_data, t_index, window_size, scaler, holding_status):
    n_feat=NUM_OHLC_FEATURES; state_dim=window_size*n_feat+1; fallback=np.zeros((1,state_dim))
    if scaler is None or not hasattr(scaler, 'transform'): print("ERROR (getState): Invalid Scaler."); return fallback
    holding_status=float(holding_status) if holding_status in [0.0,1.0] else 0.0; fallback[0,-1]=holding_status
    start=t_index-window_size+1; end=t_index+1
    if start < 0: return fallback
    if end > len(ohlc_data): print(f"ERROR (getState): Index out of bounds."); return fallback
    window_data=ohlc_data[start:end,:];
    if window_data.shape != (window_size,n_feat): print(f"ERROR (getState): Window shape mismatch."); return fallback
    try: scaled_window = scaler.transform(window_data)
    except Exception as e: print(f"ERROR (getState): Scaler transform failed - {e}"); return fallback
    ohlc_part=scaled_window.flatten(); hold_feat=np.array([holding_status],dtype=np.float32)
    state=np.concatenate((ohlc_part, hold_feat)).reshape(1,-1)
    if state.shape[1] != state_dim: print(f"ERROR (getState): Final state shape mismatch."); return fallback
    return state

# === Reinforcement Learning Agent Class ===
# ... (Keep Agent class definition as before) ...
class Agent:
    def __init__(self, window_size, is_eval=False, model_path=""): # ... (same as V1.6/V1.7) ...
        self.state_size = window_size * NUM_OHLC_FEATURES + 1; self.window_size = window_size
        self.action_size = 3; self.memory = deque(maxlen=5000); self.model_path = model_path
        self.is_eval = is_eval; self.gamma = 0.95; self.epsilon = 0.0 if is_eval else 1.0
        self.epsilon_min = 0.01; self.epsilon_decay = 0.995
        self.model = self._load_or_build_model(model_path)
    def _load_or_build_model(self, model_path): # ... (same as V1.6/V1.7, includes validation) ...
        if self.is_eval and model_path and os.path.exists(model_path):
            print(f"Attempting to load RL model: {model_path}")
            try:
                model = load_model(model_path, compile=True); loaded_input_dim = model.layers[0].input_shape[-1]
                if loaded_input_dim != self.state_size: print("*"*20+" CRITICAL WARNING "+"*"*20+f"\nLoaded RL model input dim ({loaded_input_dim}) != expected ({self.state_size})!\n"+"*"*65); sys.exit("Incompatible RL model.")
                else: print(f"[OK] RL model '{model_path}' loaded successfully."); return model
            except Exception as e: print(f"ERROR: Failed loading RL model '{model_path}' - {e}."); sys.exit("Failed loading model in eval mode.")
        elif self.is_eval and model_path: print(f"ERROR: Specified RL model '{model_path}' not found. Exiting."); sys.exit(1)
        return self._build_model()
    def _build_model(self): # ... (same as V1.6/V1.7) ...
        model = Sequential([Dense(128, input_dim=self.state_size, activation='relu'), Dense(64, activation='relu'), Dense(self.action_size, activation='linear')])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001)); print(f"[OK] New RL Q-Network built (Input Dim: {self.state_size})."); return model
    def act(self, state): # ... (same as V1.6/V1.7) ...
        if not self.is_eval and np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        if state is None or state.shape != (1, self.state_size): print(f"ERROR (act): Invalid state shape. Random action."); return random.randrange(self.action_size)
        try:
            with np.errstate(all='ignore'): act_values = self.model.predict_on_batch(state)
            if np.isnan(act_values).any() or np.isinf(act_values).any(): print("WARN (act): Model prediction NaN/Inf. Random action."); return random.randrange(self.action_size)
            return np.argmax(act_values[0])
        except Exception as e: print(f"ERROR (act): Model prediction failed - {e}. Random action."); return random.randrange(self.action_size)
    def remember(self, state, action, reward, next_state, done): # ... (same as V1.6/V1.7) ...
         if not self.is_eval:
             if state is not None and next_state is not None and state.shape == (1, self.state_size) and next_state.shape == (1, self.state_size): self.memory.append((state, action, reward, next_state, done))
    def replay(self, batch_size): # ... (same as V1.6/V1.7) ...
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
    def save(self, name): # ... (same as V1.6/V1.7) ...
        if hasattr(self, 'model') and self.model is not None:
            try: self.model.save(name); print(f"RL model saved to '{name}'.")
            except Exception as e: print(f"ERROR: Failed saving RL model to '{name}' - {e}")
        else: print("ERROR (save): No model instance to save.")

# === RL Training Function (Placeholder - Needs Refactoring for Multi-Stock & Delay) ===
def training_rl_model():
    """Placeholder: Needs MAJOR refactoring for multi-stock, delayed rewards."""
    print("\n" + "#"*60 + "\n#" + " "*13 + "RL 模型訓練 (佔位符 - 待重構)" + " "*13 + "#\n" + "#"*60)
    print("錯誤: 此訓練函數需要重構以支持多股票和延遲獎勵。跳過訓練。")
    print("#"*60 + "\n")
    return

# === Trading Execution Function (Fixed Stock List) ===
def trading_execution():
    """Executes pre-market decisions for a fixed list of stocks."""
    print("\n" + "="*60); print("=" + " "*15 + "開始執行【盤前】固定列表股票交易決策" + " "*15 + "="); print("="*60)

    # --- Configuration & Initial Setup ---
    shares_to_trade = SHARES_TO_TRADE
    rl_model_load_path = RL_MODEL_LOAD_PATH
    window_size = WINDOW_SIZE
    user_account_stock = USER_ACCOUNT; user_password_stock = USER_PASSWORD
    team_account_history = TEAM_ACCOUNT_ID
    day_trade_sell_price_method = DAY_TRADE_SELL_METHOD
    day_trade_profit_target = DAY_TRADE_PROFIT_TARGET_PERCENT

    # --- 1. Load FIXED Stock List ---
    print("\n[Step 1/N] 載入固定股票列表...")
    target_stock_codes = load_fixed_stock_list() # Use new function
    if not target_stock_codes: print("ERROR: 無法載入股票列表，交易終止。"); return

    # --- 2. Load Dependencies (Model, Trade Info) ---
    print("\n[Step 2/N] 載入共用模型與狀態文件...")
    # Load RL Model (Generic)
    if not os.path.exists(rl_model_load_path): print(f"ERROR: RL model missing '{rl_model_load_path}'."); return
    try: agent = Agent(window_size=window_size, is_eval=True, model_path=rl_model_load_path)
    except SystemExit: return # Exit if agent validation fails
    if not hasattr(agent, 'model') or agent.model.layers[0].input_shape[-1] != agent.state_size: print("ERROR: RL model validation failed."); return
    print("[OK] 通用 RL Agent 初始化成功。")
    # Load Trade Info File
    trade_info = load_trade_info(); print(f"[OK] 已載入 {len(trade_info)} 筆股票交易狀態。")
    # Scaler will be loaded dynamically per stock

    # --- 3. Get Yesterday's Date & Transaction History ---
    print("\n[Step 3/N] 查詢昨日交易歷史...")
    yesterday = datetime.now() - timedelta(days=1); yesterday_str = yesterday.strftime('%Y%m%d') # TODO: Handle non-trading days
    transactions_yesterday = Get_Transaction_History(team_account_history, yesterday_str)
    successful_transactions_yesterday = { t['stock_code']: t for t in transactions_yesterday if t.get("state") == "交易成功" and t.get("stock_code") in target_stock_codes }
    print(f"[OK] 獲取 {len(transactions_yesterday)} 筆昨日記錄，過濾後 {len(successful_transactions_yesterday)} 筆相關成功記錄。")

    # --- 4. Get Current Holdings ---
    print("\n[Step 4/N] 查詢當前實際持股...")
    user_stocks_list = Get_User_Stocks(user_account_stock, user_password_stock)
    if user_stocks_list is None: print("ERROR: 無法查詢當前持股！"); return
    current_holdings = { stock.get('stock_code_id', stock.get('stock_code')): int(stock.get('shares', stock.get('stock_shares', 0))) for stock in user_stocks_list if stock.get('stock_code_id') or stock.get('stock_code') }
    print(f"[OK] 當前持有 {len(current_holdings)} 種股票。")

    # --- 5. Iterate Through FIXED Stock List ---
    print("\n[Step 5/N] 開始遍歷【固定列表】股票進行決策...")
    actions_summary = []

    for stock_code in target_stock_codes:
        print(f"\n--- Processing Stock: {stock_code} ---")
        stock_trade_status = trade_info.get(stock_code, {"holding": False, "purchase_price": None, "intended_action": None})
        last_run_intended_action = stock_trade_status.get("intended_action")
        last_run_purchase_price = stock_trade_status.get("purchase_price") # Already float/None

        # --- 5a. Reconcile Status ---
        print(f"   [5a] 核對 {stock_code} 昨日交易歷史...")
        current_purchase_price = None; reconciled_holding = stock_trade_status.get("holding", False)
        relevant_tx = successful_transactions_yesterday.get(stock_code)
        if last_run_intended_action == "buy":
            successful_buy = next((t for t in [relevant_tx] if t and t.get("type") == "買進"), None)
            if successful_buy:
                try: actual_buy_price = float(successful_buy["price"]); print(f"      [CONFIRMED] BUY successful @ {actual_buy_price:.2f}"); reconciled_holding = True; current_purchase_price = actual_buy_price
                except: print(f"      ERROR: Parse price failed for buy tx: {successful_buy}"); reconciled_holding = True; current_purchase_price = None
            else: print("      [FAILED] BUY intention failed."); reconciled_holding = False; current_purchase_price = None
        elif last_run_intended_action == "sell":
            successful_sell = next((t for t in [relevant_tx] if t and t.get("type") == "賣出"), None)
            if successful_sell: print(f"      [CONFIRMED] SELL successful."); reconciled_holding = False; current_purchase_price = None
            else: print("      [FAILED] SELL intention failed."); reconciled_holding = True; current_purchase_price = last_run_purchase_price
        else: reconciled_holding = stock_trade_status.get("holding", False); current_purchase_price = last_run_purchase_price;
        # Update trade_info with reconciled data before final check
        trade_info[stock_code] = {"holding": reconciled_holding, "purchase_price": current_purchase_price, "intended_action": None}
        # Do not save yet, wait for final sync


        # --- 5b. Final Sync ---
        print(f"   [5b] 與當前 API 持股最終同步...")
        actual_shares_held = current_holdings.get(stock_code, 0)
        final_confirmed_holding = (actual_shares_held > 0)
        if final_confirmed_holding != reconciled_holding:
            print(f"      SYNC WARNING: API holding ({final_confirmed_holding}) != Reconciled file ({reconciled_holding})! API overrides.")
            trade_info[stock_code]['holding'] = final_confirmed_holding
            if not final_confirmed_holding: trade_info[stock_code]['purchase_price'] = None
            elif final_confirmed_holding and not reconciled_holding: trade_info[stock_code]['purchase_price'] = None # Lost cost basis
            current_purchase_price = trade_info[stock_code].get('purchase_price')
        print(f"      Final Status: Holding={final_confirmed_holding}, Shares={actual_shares_held}, Cost={current_purchase_price}")

        # --- 5c. Get OHLC Data ---
        print(f"   [5c] 獲取 {stock_code} 的 OHLC 數據...")
        ohlc_data_stock = get_ohlc_data_for_stock(stock_code)
        if ohlc_data_stock is None or len(ohlc_data_stock) < window_size: print(f"      ERROR: Failed getting OHLC for {stock_code}. Skipping."); continue
        last_idx_stock = len(ohlc_data_stock) - 1
        last_close_stock = ohlc_data_stock[last_idx_stock, 3]

        # --- 5d. Stop-Loss Check ---
        print(f"   [5d] 執行 {stock_code} 止損檢查...")
        stop_loss_triggered = False
        if final_confirmed_holding and current_purchase_price is not None:
            stop_loss_price = current_purchase_price * (1 - STOP_LOSS_PERCENTAGE / 100.0)
            if last_close_stock < stop_loss_price: print(f"      !!! STOP-LOSS TRIGGERED (PrevClose {last_close_stock:.2f} < Stop {stop_loss_price:.2f}) !!!"); stop_loss_triggered = True

        # --- 5e. Get LSTM Prediction & Scaler ---
        print(f"   [5e] 獲取 {stock_code} 的 LSTM 預測與 Scaler...")
        # Call predictor, now expecting (open, close, scaler_object)
        predicted_open, predicted_close, scaler_for_stock = predict_next_open_close(stock_code=stock_code)
        if predicted_open is None or scaler_for_stock is None:
             print(f"      ERROR: LSTM prediction or Scaler failed for {stock_code}. Skipping RL decision.");
             # Save current reconciled status before skipping
             trade_info[stock_code] = {"holding": final_confirmed_holding, "purchase_price": current_purchase_price, "intended_action": "sit"} # Sit if cannot predict
             actions_summary.append({'stock': stock_code, 'action': 'sit', 'buy_sent': False, 'sell_sent': False})
             continue # Skip to next stock

        # --- 5f. Generate RL State ---
        print(f"   [5f] 生成 {stock_code} 的 RL 狀態...")
        current_state = getState_OHLC_Holding(ohlc_data_stock, last_idx_stock, window_size, scaler_for_stock, float(final_confirmed_holding))
        if current_state.shape[1] != agent.state_size:
            print(f"      ERROR: Failed creating RL state for {stock_code}. Sitting.");
            final_action_stock = 0; action_reason = "State Error"
        else:
            # --- 5g. Get RL Action & Final Action ---
            print(f"   [5g] 獲取 {stock_code} 的 RL 決策...")
            rl_action = agent.act(current_state)
            action_map = {0: "Sit", 1: "Buy", 2: "Sell"}
            final_action_stock = rl_action; action_reason = f"RL Agent ({action_map.get(rl_action, '?')})"
            if stop_loss_triggered:
                if final_action_stock != 2: print(f"      ACTION OVERRIDE: Stop-Loss. Forcing SELL."); final_action_stock = 2
                action_reason = "Stop-Loss" + (" & RL" if rl_action == 2 else "")
            print(f"      Final Action for {stock_code}: {final_action_stock} ({action_reason})")

        # --- 5h. Set Order Prices ---
        buy_order_price = predicted_open; sell_order_price = predicted_open
        if DAY_TRADE_SELL_METHOD == 'profit_target': day_trade_sell_price = buy_order_price * (1 + DAY_TRADE_PROFIT_TARGET_PERCENT / 100.0)
        else: day_trade_sell_price = predicted_close
        print(f"      Pred: Open={predicted_open:.2f}, Close={predicted_close:.2f}. Orders-> Buy:{buy_order_price:.2f}, Sell:{sell_order_price:.2f}, DayT:{day_trade_sell_price:.2f}")

        # --- 5i. Execute Orders & Set Intention ---
        print(f"   [5i] 執行 {stock_code} 的預約單...")
        intended_action_stock = "sit"; buy_sent = False; sell_sent = False
        # ... (Order placement logic based on final_action_stock - same as V1.7) ...
        if final_action_stock == 1: # Buy
            if not final_confirmed_holding:
                print(f"      ACTION: Placing BUY @ {buy_order_price:.2f}")
                if Buy_Stock(user_account_stock, user_password_stock, stock_code, shares_to_trade, buy_order_price):
                     buy_sent=True; intended_action_stock="buy"
                     print(f"      ACTION: Placing DayTrade SELL @ {day_trade_sell_price:.2f}")
                     if Sell_Stock(user_account_stock, user_password_stock, stock_code, shares_to_trade, day_trade_sell_price): sell_sent=True
                     else: print("      WARN: DayTrade SELL order failed.")
                else: print("      WARN: BUY order failed.")
            else: print("      INFO: BUY action, but already holding. No action.")
        elif final_action_stock == 2: # Sell
            if final_confirmed_holding:
                 if actual_shares_held >= shares_to_trade:
                     print(f"      ACTION: Placing SELL @ {sell_order_price:.2f} (Reason: {action_reason})")
                     if Sell_Stock(user_account_stock, user_password_stock, stock_code, shares_to_trade, sell_order_price): sell_sent=True; intended_action_stock="sell"
                     else: print("      WARN: SELL order failed.")
                 else: print(f"      INFO: SELL action, but have {actual_shares_held} < needed {shares_to_trade}. No action.")
            else: print("      INFO: SELL action, but not holding. No action.")
        else: # Sit
            print("      INFO: SIT action. No orders placed.")
            intended_action_stock = "sit"

        # Update trade_info dict for this stock (intention for next day)
        holding_intended = (intended_action_stock == "buy") or (final_confirmed_holding and intended_action_stock != "sell")
        price_intended = buy_order_price if intended_action_stock == "buy" else (current_purchase_price if holding_intended else None)
        trade_info[stock_code] = {"holding": holding_intended, "purchase_price": price_intended, "intended_action": intended_action_stock}
        actions_summary.append({'stock': stock_code, 'action': intended_action_stock, 'buy_sent': buy_sent, 'sell_sent': sell_sent})
        # print(f"      Intention Set: Holding={holding_intended}, Price={price_intended}, Action={intended_action_stock}") # Reduce noise

    # --- 6. Save All Trade Info ---
    print("\n[Step 6/N] 保存所有股票的交易意圖...")
    save_trade_info(trade_info) # Save the entire dictionary
    print("[OK] Trade intentions saved.")

    # --- Final Summary ---
    print("\n" + "="*60); print("=" + " "*14 + "【盤前】固定列表股票決策與預約單完成" + " "*14 + "=")
    print(f"處理股票數: {len(target_stock_codes)}")
    for summary in actions_summary: print(f"  - {summary['stock']}: Intended={summary['action']}, BuySent={summary['buy_sent']}, SellSent={summary['sell_sent']}")
    print("=" + " "*9 + "實際交易結果需在明日開市後或收盤後查詢確認" + " "*9 + "="); print("="*60 + "\n")


# === Main Execution Block ===
if __name__ == "__main__":
    print("="*40); print("   股票交易機器人 (v1.9 - 固定列表)"); print("="*40)
    start_run_time = datetime.now(); print(f"啟動時間: {start_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用固定股票列表: {STOCK_LIST_FILE}"); print(f"止損比例: {STOP_LOSS_PERCENTAGE}%")

    # 1. Execute Pre-Market Trading Logic
    #    Training function needs significant work for multi-stock and delay.
    # training_rl_model() # <<< DO NOT RUN unless refactored

    trading_execution()

    end_run_time = datetime.now(); print(f"\n程式執行完畢: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}"); print(f"總耗時: {end_run_time - start_run_time}")
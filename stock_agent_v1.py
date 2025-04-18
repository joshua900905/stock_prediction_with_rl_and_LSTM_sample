# -*- coding: utf-8 -*-
# stock_agent.py
# Description: Multi-Stock RL Agent (Fixed List) aiming for ~1M TWD trade value.
#              Includes History Check, Stop-Loss, OHLC+Holding State, LSTM Prediction,
#              and Refactored Training Logic for Delayed Confirmation.
# Version: 1.9.3 (Fixed List + Target Value + Corrected Syntax + Refactored Training Logic)

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

# --- Keras/TensorFlow Imports (Simplified) ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    print("[Info] Using TensorFlow Keras imports.")
except ImportError:
    print("CRITICAL ERROR: Cannot import TensorFlow/Keras. Please install tensorflow.")
    sys.exit(1)

# --- Import LSTM Predictor ---
try:
    from price_prediction_v1 import predict_next_open_close
    from price_prediction_v1 import SCALER_FILENAME as SCALER_FILENAME_TPL
    print("[OK] Successfully imported 'price_prediction_v1.py'")
except ImportError: print("CRITICAL ERROR: Cannot import 'price_prediction_v1.py'."); sys.exit(1)
except AttributeError: print("WARNING: Cannot import SCALER_FILENAME_TPL. Using default."); SCALER_FILENAME_TPL="lstm_models/scaler_{stock_code}.pkl"


# === Configuration ===
NUM_OHLC_FEATURES = 4
STOCK_LIST_FILE = "stock_list.txt" # File with fixed stock codes
API_BASE_URL_STOCK = "http://140.116.86.242:8081"
API_BASE_URL_HISTORY = "http://140.116.86.241:8800"
REQUEST_TIMEOUT = 25
TRADE_INFO_FILE = f"trade_status_multi_fixed.json"
STOP_LOSS_PERCENTAGE = 5.0
# --- Account Info ---
USER_ACCOUNT = "CIoT Bot-CE"     # <<< REPLACE >>>
USER_PASSWORD = "CIoT Bot-CE"    # <<< REPLACE >>>
TEAM_ACCOUNT_ID = "team1"        # <<< REPLACE >>>
# --- RL Model (Generic) ---
RL_MODEL_LOAD_PATH = "rl_agent_multi_ohlc_hold_delay_ep100.h5" # <<< UPDATE to your trained model file >>>
WINDOW_SIZE = 10 # <<< MUST match model training >>>
# --- Training Params ---
TRAIN_INVALID_ACTION_PENALTY = -2.0
TRAIN_HOLDING_REWARD_FACTOR = 0.0 # Optional: 0 = disabled
# --- Trading Params ---
TARGET_TRADE_VALUE = 1000000.0 # Target TWD value per trade
MIN_SHARES_PER_TRADE = 1      # Minimum shares (in lots)
DAY_TRADE_SELL_METHOD = 'predicted_close'
DAY_TRADE_PROFIT_TARGET_PERCENT = 2.0

# === Function to Load Fixed Stock List ===
# ... (load_fixed_stock_list - Keep as before) ...
def load_fixed_stock_list(filename=STOCK_LIST_FILE):
    try:
        with open(filename, 'r', encoding='utf-8') as f: content = f.read();
        if content.startswith('\ufeff'): content = content[1:]
        stock_codes = [c.strip() for c in content.split(',') if c.strip().isdigit() and len(c.strip())==4]
        if not stock_codes: print(f"ERROR: No valid stock codes in '{filename}'."); return None
        print(f"[OK] Loaded {len(stock_codes)} fixed stock codes."); return stock_codes
    except FileNotFoundError: print(f"CRITICAL ERROR: File '{filename}' not found."); return None
    except Exception as e: print(f"CRITICAL ERROR: Failed reading '{filename}' - {e}"); return None

# === API Helper Functions ===
# ... (api_request, Get_Stock_Informations, Get_User_Stocks, Buy_Stock, Sell_Stock, Get_Transaction_History - Keep as before) ...
def api_request(base_url, method, endpoint, data=None, retry_attempts=2):
    url = f"{base_url}/{endpoint}"; last_exception = None
    for attempt in range(retry_attempts + 1):
        try:
            if method.upper() == 'GET': response = requests.get(url, timeout=REQUEST_TIMEOUT)
            elif method.upper() == 'POST': response = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
            else: print(f"ERROR: Unsupported method {method}"); return None
            response.raise_for_status(); return response.json()
        except requests.exceptions.Timeout: msg=f"API Timeout (Attempt {attempt + 1})"; last_exception=TimeoutError(msg); print(f"WARN: {msg} URL: {url}")
        except requests.exceptions.HTTPError as http_err: msg=f"API HTTP Error: {http_err}"; last_exception=http_err; print(f"ERROR: {msg} URL: {url}"); break
        except requests.exceptions.RequestException as req_err: msg=f"API Request Failed: {req_err}"; last_exception=req_err; print(f"ERROR: {msg} URL: {url}")
        except json.JSONDecodeError as json_err: msg=f"Failed JSON decode: {json_err}"; last_exception=json_err; print(f"ERROR: {msg} URL: {url}"); break
        except Exception as e: msg=f"Unknown API Error: {e}"; last_exception=e; print(f"ERROR: {msg} URL: {url}")
        if attempt < retry_attempts: time.sleep(1 + attempt)
    print(f"ERROR: API request failed after {retry_attempts + 1} attempts. Last error: {last_exception} URL: {url}"); return None
def Get_Stock_Informations(stock_code, start_date, stop_date):
    endpoint=f'stock/api/v1/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}'; result=api_request(API_BASE_URL_STOCK,'GET',endpoint)
    if result and result.get('result')=='success': return result.get('data',[])
    status = result.get('status','Unk') if result else 'NoResp'; return []
def Get_User_Stocks(account, password):
    endpoint='stock/api/v1/get_user_stocks'; data={'account': account,'password': password}; result=api_request(API_BASE_URL_STOCK,'POST',endpoint,data=data)
    if result and result.get('result')=='success': return result.get('data',[])
    status = result.get('status','Unk') if result else 'NoResp'; print(f"API Get_User_Stocks failed: {status}. None."); return None
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempt PRE-MARKET BUY: {stock_code}({stock_shares} sh)@{stock_price:.2f}')
    endpoint='stock/api/v1/buy'; data={'account':account,'password':password,'stock_code':str(stock_code),'stock_shares':str(stock_shares),'stock_price':f"{stock_price:.2f}"}
    result=api_request(API_BASE_URL_STOCK,'POST',endpoint,data=data)
    if result: print(f' BUY Result:{result.get("result","N/A")}|Status:{result.get("status","N/A")}'); return result.get('result')=='success'
    else: print(' BUY Request Fail: No API response.'); return False
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print(f'Attempt PRE-MARKET SELL: {stock_code}({stock_shares} sh)@{stock_price:.2f}')
    endpoint='stock/api/v1/sell'; data={'account':account,'password':password,'stock_code':str(stock_code),'stock_shares':str(stock_shares),'stock_price':f"{stock_price:.2f}"}
    result=api_request(API_BASE_URL_STOCK,'POST',endpoint,data=data)
    if result: print(f' SELL Result:{result.get("result","N/A")}|Status:{result.get("status","N/A")}'); return result.get('result')=='success'
    else: print(' SELL Request Fail: No API response.'); return False
def Get_Transaction_History(team_account_id, query_date_str):
    print(f"Query Tx History for {team_account_id} on {query_date_str}...")
    endpoint=f'api/v1/transaction_history/{team_account_id}/{query_date_str}/{query_date_str}'; result=api_request(API_BASE_URL_HISTORY,'GET',endpoint)
    if result:
        if isinstance(result, list): print(f"[OK] Found {len(result)} raw tx(s)."); return result
        elif isinstance(result, dict) and result.get('result')=='failed': print(f"WARN: Tx history query failed(API):{result.get('status','Unk')}"); return []
        else: print(f"WARN: Unexpected tx history format:{type(result)}"); return []
    else: print("ERROR: No response from tx history API."); return []


# === Stop-Loss & Trade Info Helpers ===
# ... (load_trade_info, save_trade_info - Keep as before) ...
def load_trade_info(filename=TRADE_INFO_FILE):
    default_info = {};
    if os.path.exists(filename):
        try:
            with open(filename,'r',encoding='utf-8') as f: info = json.load(f)
            if not isinstance(info, dict): print(f"WARN: Invalid format in '{filename}'. Resetting."); return {}
            validated_info = {}
            for sc,sd in info.items():
                 if isinstance(sd, dict):
                     sd.setdefault('holding',False); sd.setdefault('purchase_price',None); sd.setdefault('intended_action',None)
                     if sd['purchase_price'] is not None:
                         try: sd['purchase_price'] = float(sd['purchase_price'])
                         except: sd['purchase_price'] = None
                     sd['holding']=bool(sd['holding']); validated_info[sc]=sd
            return validated_info
        except Exception as e: print(f"ERROR: Failed loading '{filename}': {e}. Using empty."); return {}
    return default_info
def save_trade_info(trade_info, filename=TRADE_INFO_FILE):
    try:
        for sc,sd in trade_info.items():
             if sd and sd.get('purchase_price') is not None:
                 try: sd['purchase_price'] = float(sd['purchase_price'])
                 except: sd['purchase_price'] = None
        with open(filename,'w',encoding='utf-8') as f: json.dump(trade_info,f,indent=4,ensure_ascii=False)
    except Exception as e: print(f"ERROR: Failed saving to '{filename}': {e}")


# === RL Helper Functions ===
# ... (formatPrice, get_ohlc_data_for_stock, getState_OHLC_Holding - Keep as before) ...
def formatPrice(n): # <<< CORRECTED VERSION >>>
    """簡單格式化價格字串。"""
    try: price_value = float(n); return ("-" if price_value < 0 else "") + "NT${:,.2f}".format(abs(price_value))
    except (TypeError, ValueError): return "N/A"
    except Exception as e: print(f"ERROR (formatPrice): Unexpected error - {e}"); return "N/A"
def get_ohlc_data_for_stock(stock_code):
    print(f"RL Agent: Fetching OHLC data for {stock_code}..."); features_to_get=['open','high','low','close'] # Use correct case based on API
    start_date=(datetime.now()-timedelta(days=730)).strftime('%Y%m%d'); end_date=datetime.now().strftime('%Y%m%d')
    stock_history=Get_Stock_Informations(stock_code,start_date,end_date)
    if not stock_history: print(f"ERROR (get_ohlc {stock_code}): Failed fetch."); return None
    ohlc_data=[]; dates=[]
    for h in stock_history:
        # --- Adjust Key Names Here if API returns lowercase ---
        keys_in_history = {k.lower() for k in h.keys()} # Convert API keys to lowercase for comparison
        expected_keys_lower = {f.lower() for f in features_to_get}
        # if all(k_lower in keys_in_history for k_lower in expected_keys_lower): # Check lowercase
        if all(k in h for k in features_to_get): # Keep original check if API uses TitleCase
            try:
                # Use original case FEATURES for appending order
                ohlc_data.append([float(h[k]) for k in features_to_get]);
                dates.append(h.get('Date'))
            except (ValueError, TypeError): pass # Skip rows with non-numeric data
    if not ohlc_data: print(f"ERROR (get_ohlc {stock_code}): No valid records extracted. Check FEATURES keys vs API response."); return None
    ohlc_data=np.array(ohlc_data,dtype=np.float32);
    return ohlc_data
def getState_OHLC_Holding(ohlc_data,t_index,window_size,scaler,holding_status):
    n_feat=NUM_OHLC_FEATURES; state_dim=window_size*n_feat+1; fallback=np.zeros((1,state_dim))
    if scaler is None or not hasattr(scaler,'transform'): print("ERROR (getState): Invalid Scaler."); return fallback
    holding_status=float(holding_status) if holding_status in [0.0,1.0] else 0.0; fallback[0,-1]=holding_status
    start=t_index-window_size+1; end=t_index+1
    if start<0: return fallback
    if end>len(ohlc_data): print(f"ERROR (getState): Index out of bounds {end}>{len(ohlc_data)} for t={t_index}."); return fallback
    window_data=ohlc_data[start:end,:];
    if window_data.shape!=(window_size,n_feat): print(f"ERROR (getState): Window shape mismatch {window_data.shape} != {(window_size, n_feat)}."); return fallback
    try: scaled_window = scaler.transform(window_data)
    except Exception as e: print(f"ERROR (getState): Scaler transform failed: {e}"); return fallback
    ohlc_part=scaled_window.flatten(); hold_feat=np.array([holding_status],dtype=np.float32)
    state=np.concatenate((ohlc_part,hold_feat)).reshape(1,-1)
    if state.shape[1]!=state_dim: print("ERROR (getState): Final state shape mismatch."); return fallback
    return state

# === Reinforcement Learning Agent Class ===
class Agent: # <<< Using V1.9.3 Corrected Version >>>
    """DQN Agent using OHLC+Holding state."""
    def __init__(self, window_size, is_eval=False, model_path=""):
        self.state_size = window_size * NUM_OHLC_FEATURES + 1; self.window_size = window_size
        self.action_size = 3; self.memory = deque(maxlen=5000); self.model_path = model_path
        self.is_eval = is_eval; self.gamma = 0.95; self.epsilon = 0.0 if is_eval else 1.0
        self.epsilon_min = 0.01; self.epsilon_decay = 0.995
        self.model = self._load_or_build_model(model_path)
    def _load_or_build_model(self, model_path):
        if self.is_eval and model_path and os.path.exists(model_path):
            print(f"Attempting load RL model: {model_path}");
            try:
                model=load_model(model_path,compile=True); loaded_dim=model.layers[0].input_shape[-1]
                if loaded_dim != self.state_size: print("*"*20+" CRITICAL WARNING "+"*"*20+f"\nLoaded RL model dim ({loaded_dim}) != expected ({self.state_size})!\n"+"*"*65); sys.exit("Incompatible RL model.")
                else: print(f"[OK] RL model '{model_path}' loaded."); return model
            except Exception as e: print(f"ERROR loading RL model: {e}."); sys.exit("Failed loading model.")
        elif self.is_eval and model_path: print(f"ERROR: RL model '{model_path}' not found. Exiting."); sys.exit(1)
        return self._build_model()
    def _build_model(self):
        model = Sequential([Dense(128, input_dim=self.state_size, activation='relu'), Dense(64, activation='relu'), Dense(self.action_size, activation='linear')])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001)); print(f"[OK] New RL Q-Network built (Input Dim: {self.state_size})."); return model
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        if state is None or state.shape != (1, self.state_size): print(f"ERROR (act): Invalid state shape. Random."); return random.randrange(self.action_size)
        try:
            with np.errstate(all='ignore'): act_values = self.model.predict_on_batch(state)
            if np.isnan(act_values).any() or np.isinf(act_values).any(): print("WARN (act): Prediction NaN/Inf. Random."); return random.randrange(self.action_size)
            return np.argmax(act_values[0])
        except Exception as e: print(f"ERROR (act): Prediction failed - {e}. Random."); return random.randrange(self.action_size)
    def remember(self, state, action, reward, next_state, done):
         if not self.is_eval:
             if state is not None and next_state is not None and state.shape == (1, self.state_size) and next_state.shape == (1, self.state_size): self.memory.append((state, action, reward, next_state, done))
    def replay(self, batch_size):
         if self.is_eval or len(self.memory) < batch_size: return
         minibatch = random.sample(self.memory, batch_size); states = np.vstack([t[0] for t in minibatch]); next_states = np.vstack([t[3] for t in minibatch])
         if states.shape[1] != self.state_size or next_states.shape[1] != self.state_size: print(f"ERROR (replay): Batch state dimension mismatch."); return
         try:
             q_curr=self.model.predict_on_batch(states); q_next=self.model.predict_on_batch(next_states); targets=np.copy(q_curr)
             for i,(s,a,r,ns,d) in enumerate(minibatch):
                 if d or np.isnan(q_next[i]).any() or np.isinf(q_next[i]).any(): targets[i,a] = r
                 else: targets[i,a] = r + self.gamma * np.amax(q_next[i])
             valid = ~np.isnan(targets).any(axis=1) & ~np.isinf(targets).any(axis=1)
             if not valid.all(): states=states[valid]; targets=targets[valid]
             if states.shape[0]>0: self.model.train_on_batch(states, targets)
         except Exception as e: print(f"ERROR (replay): Batch train failed - {e}")
         if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save(self, name): # <<< CORRECTED VERSION >>>
        """Saves the entire model."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.save(name)
                print(f"RL model saved to '{name}'.")
            except Exception as e:
                print(f"ERROR: Failed saving RL model to '{name}' - {e}")
        else:
            print("ERROR (save): No model instance exists to save.")

# === RL Training Function (Refactored for Multi-Stock & Delayed Confirmation) ===
def training_rl_model(): # <<< REFACTORED VERSION from V1.9.1 >>>
    """
    Trains the GENERIC RL agent using multiple stocks from the list,
    simulating the pre-market order / next-day confirmation constraint.
    """
    print("\n" + "#"*60); print("#" + " "*10 + "RL 模型訓練 (多股票 - 模擬延遲確認)" + " "*10 + "#"); print("#"*60)
    # --- Config ---
    window_size=WINDOW_SIZE; episode_count=100 # <<< SET EPISODES >>>
    batch_size=64; model_save_prefix=f"rl_agent_multi_ohlc_hold_delay_"
    # --- Load Stock List ---
    print("[Train] Loading target stock list..."); target_stock_codes=load_fixed_stock_list()
    if not target_stock_codes: print("ERROR: Cannot load stock list."); return
    # --- Load Scalers ---
    print("[Train] Loading Scalers..."); scalers={}; all_scalers_loaded=True
    for code in target_stock_codes:
        scaler_path=SCALER_FILENAME_TPL.format(stock_code=code)
        if not os.path.exists(scaler_path): print(f"ERROR: Scaler missing for {code}."); all_scalers_loaded=False; break
        try: scalers[code] = joblib.load(scaler_path)
        except Exception as e: print(f"ERROR: Failed loading scaler for {code}: {e}."); all_scalers_loaded=False; break
    if not all_scalers_loaded: return; print(f"[OK] Loaded {len(scalers)} scalers.")
    # --- Initialize Agent ---
    print("[Train] Initializing RL Agent..."); agent=Agent(window_size=window_size,is_eval=False)
    # --- Get OHLC Data ---
    print("[Train] Fetching OHLC data..."); all_ohlc_data={}; min_data_len=float('inf')
    for code in target_stock_codes:
        data=get_ohlc_data_for_stock(code)
        if data is not None and len(data)>window_size+1: all_ohlc_data[code]=data; min_data_len=min(min_data_len,len(data))
        else: print(f"WARN: Insufficient data for {code}, skipped.")
    if not all_ohlc_data or min_data_len<=window_size+1: print("ERROR: No stocks with sufficient data."); return
    usable_data_len=min_data_len; print(f"[OK] Using data up to length {usable_data_len}.")
    trainable_stock_codes=list(all_ohlc_data.keys());
    if not trainable_stock_codes: print("ERROR: No trainable stocks."); return
    # --- Training Loop ---
    print(f"[Train] Starting {episode_count} episodes across {len(trainable_stock_codes)} stocks...")
    start_time_train=time.time(); max_avg_profit=-float('inf')
    for e in range(episode_count):
        ep_start=time.time(); print(f"\n--- Episode {e+1}/{episode_count} ---")
        ep_profit=0.0; ep_trades=0; random.shuffle(trainable_stock_codes)
        for stock_code in trainable_stock_codes:
            holding=False; cost=None; stock_profit=0.0; trades=0
            ohlc_data=all_ohlc_data[stock_code]; scaler=scalers[stock_code]
            t_init=window_size-1; state=getState_OHLC_Holding(ohlc_data,t_init,window_size,scaler,0.0)
            if state.shape[1]!=agent.state_size: continue
            for t in range(t_init, usable_data_len-1):
                action=agent.act(state); O1=ohlc_data[t+1,0];H1=ohlc_data[t+1,1];L1=ohlc_data[t+1,2];C1=ohlc_data[t+1,3]
                reward=0.0; next_hold=holding; fill_p=None
                if action==1: # Buy
                    if not holding:
                        buy_p=O1;
                        if L1<=buy_p<=H1: next_hold=True; cost=buy_p; reward=(C1-buy_p)*TRAIN_HOLDING_REWARD_FACTOR; trades+=1
                    else: reward=TRAIN_INVALID_ACTION_PENALTY
                elif action==2: # Sell
                    if holding and cost is not None:
                        sell_p=O1;
                        if L1<=sell_p<=H1: profit=sell_p-cost; reward=profit; stock_profit+=profit; next_hold=False; cost=None; trades+=1
                    else: reward=TRAIN_INVALID_ACTION_PENALTY
                else: # Sit
                     if holding: reward=(C1-ohlc_data[t,3])*TRAIN_HOLDING_REWARD_FACTOR
                holding=next_hold
                next_state=getState_OHLC_Holding(ohlc_data,t+1,window_size,scaler,float(holding))
                done=(t==usable_data_len-2) or (next_state.shape[1]!=agent.state_size)
                agent.remember(state,action,reward,next_state,done)
                state=next_state
                if len(agent.memory)>batch_size*5 and t%10==0: agent.replay(batch_size) # Replay less often?
                if done: break
            ep_profit+=stock_profit; ep_trades+=trades
        ep_dur=time.time()-ep_start; avg_prof=ep_profit/len(trainable_stock_codes) if trainable_stock_codes else 0
        print(f"Ep {e+1} done. Dur:{ep_dur:.1f}s|AvgProfit:{formatPrice(avg_prof)}|Trades:{ep_trades}|Eps:{agent.epsilon:.4f}")
        if avg_prof>max_avg_profit: max_avg_profit=avg_prof
        if (e+1)%20==0 or e==episode_count-1 or avg_prof>=max_avg_profit*0.9: agent.save(f"{model_save_prefix}ep{e+1}.h5")
    train_dur=time.time()-start_time_train; print("\n"+"="*60+f"\nRL Training Complete. Total Time: {train_dur:.2f}s\n"+"="*60+"\n")


# === Trading Execution Function (Fixed Stock List, Target Value) ===
def trading_execution(): # <<< Using V1.9.2 Logic >>>
    """Executes pre-market decisions for fixed list, aiming for ~1M TWD value."""
    print("\n" + "="*60); print("=" + " "*15 + "開始執行【盤前】固定列表股票交易決策 (目標金額)" + " "*15 + "="); print("="*60)
    # --- Config & Setup ---
    rl_model_load_path=RL_MODEL_LOAD_PATH; window_size=WINDOW_SIZE;
    user_account_stock=USER_ACCOUNT; user_password_stock=USER_PASSWORD; team_account_history=TEAM_ACCOUNT_ID
    day_trade_sell_method=DAY_TRADE_SELL_METHOD; day_trade_profit_target=DAY_TRADE_PROFIT_TARGET_PERCENT

    # --- 1. Load FIXED Stock List ---
    print("\n[Step 1/N] 載入固定股票列表..."); target_stock_codes = load_fixed_stock_list()
    if not target_stock_codes: print("ERROR: 無法載入股票列表."); return

    # --- 2. Load Dependencies ---
    print("\n[Step 2/N] 載入共用模型與狀態文件...");
    if not os.path.exists(rl_model_load_path): print(f"ERROR: RL model missing '{rl_model_load_path}'."); return
    try: agent = Agent(window_size=window_size, is_eval=True, model_path=rl_model_load_path)
    except SystemExit: return
    if not hasattr(agent,'model') or agent.model.layers[0].input_shape[-1] != agent.state_size: print("ERROR: RL model validation failed."); return
    print("[OK] 通用 RL Agent 初始化成功。")
    trade_info = load_trade_info(); print(f"[OK] 已載入 {len(trade_info)} 筆股票交易狀態。")

    # --- 3. Get Yesterday's Date & Tx History ---
    print("\n[Step 3/N] 查詢昨日交易歷史..."); yesterday=datetime.now()-timedelta(days=1); yesterday_str=yesterday.strftime('%Y%m%d') # TODO: Handle non-trading days
    transactions_yesterday=Get_Transaction_History(team_account_history, yesterday_str)
    successful_tx_yesterday={ t['stock_code']:t for t in transactions_yesterday if t.get("state")=="交易成功" and t.get("stock_code") in target_stock_codes }
    print(f"[OK] 获取 {len(transactions_yesterday)} 筆昨日記錄, 過濾後 {len(successful_tx_yesterday)} 筆相關成功記錄。")

    # --- 4. Get Current Holdings ---
    print("\n[Step 4/N] 查詢當前實際持股..."); user_stocks_list=Get_User_Stocks(user_account_stock, user_password_stock)
    if user_stocks_list is None: print("ERROR: 無法查詢當前持股！"); return
    current_holdings={ stock.get('stock_code_id', stock.get('stock_code')): int(stock.get('shares', stock.get('stock_shares', 0))) for stock in user_stocks_list if stock.get('stock_code_id') or stock.get('stock_code') }
    print(f"[OK] 當前持有 {len(current_holdings)} 種股票。")

    # --- 5. Iterate Through FIXED Stock List ---
    print("\n[Step 5/N] 開始遍歷【固定列表】股票進行決策...")
    actions_summary=[]

    for stock_code in target_stock_codes:
        print(f"\n--- Processing Stock: {stock_code} ---")
        stock_trade_status = trade_info.get(stock_code, {"holding": False, "purchase_price": None, "intended_action": None})
        last_run_intended_action=stock_trade_status.get("intended_action"); last_run_purchase_price=stock_trade_status.get("purchase_price")

        # --- 5a. Reconcile Status ---
        print(f"   [5a] 核對 {stock_code} 昨日交易歷史...")
        current_purchase_price=None; reconciled_holding=stock_trade_status.get("holding", False)
        relevant_tx=successful_tx_yesterday.get(stock_code)
        # ... (Reconciliation logic - same as V1.9.1/1.9.2) ...
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
        trade_info[stock_code] = {"holding": reconciled_holding, "purchase_price": current_purchase_price, "intended_action": None} # Update dict in memory

        # --- 5b. Final Sync ---
        print(f"   [5b] 與當前 API 持股最終同步...")
        actual_shares_held=current_holdings.get(stock_code, 0); final_confirmed_holding=(actual_shares_held > 0)
        if final_confirmed_holding != reconciled_holding:
            print(f"      SYNC WARNING: API holding ({final_confirmed_holding}) != Reconciled file ({reconciled_holding})! API overrides.")
            trade_info.setdefault(stock_code, {})['holding'] = final_confirmed_holding
            if not final_confirmed_holding: trade_info[stock_code]['purchase_price'] = None
            elif final_confirmed_holding and not reconciled_holding: trade_info[stock_code]['purchase_price'] = None
            current_purchase_price = trade_info[stock_code].get('purchase_price')
        print(f"      Final Status: Holding={final_confirmed_holding}, Shares={actual_shares_held}, Cost={current_purchase_price}")

        # --- 5c. Get OHLC Data ---
        print(f"   [5c] 獲取 {stock_code} 的 OHLC 數據...")
        ohlc_data_stock = get_ohlc_data_for_stock(stock_code)
        if ohlc_data_stock is None or len(ohlc_data_stock) < window_size: print(f"      ERROR: Failed getting OHLC for {stock_code}. Skipping."); continue
        last_idx_stock=len(ohlc_data_stock)-1; last_close_stock=ohlc_data_stock[last_idx_stock, 3]

        # --- 5d. Stop-Loss Check ---
        print(f"   [5d] 執行 {stock_code} 止損檢查...")
        stop_loss_triggered = False
        if final_confirmed_holding and current_purchase_price is not None:
            stop_loss_price = current_purchase_price * (1 - STOP_LOSS_PERCENTAGE / 100.0)
            if last_close_stock < stop_loss_price: print(f"      !!! STOP-LOSS TRIGGERED (PrevClose {last_close_stock:.2f} < Stop {stop_loss_price:.2f}) !!!"); stop_loss_triggered = True

        # --- 5e. Get LSTM Prediction & Scaler ---
        print(f"   [5e] 獲取 {stock_code} 的 LSTM 預測與 Scaler...")
        predicted_open, predicted_close, scaler_for_stock = predict_next_open_close(stock_code=stock_code)
        if predicted_open is None or scaler_for_stock is None: print(f"      ERROR: LSTM prediction or Scaler failed for {stock_code}. Skipping."); continue

        # --- 5f. Generate RL State ---
        print(f"   [5f] 生成 {stock_code} 的 RL 狀態...")
        current_state = getState_OHLC_Holding(ohlc_data_stock, last_idx_stock, window_size, scaler_for_stock, float(final_confirmed_holding))
        final_action_stock = 0; action_reason = "State Error (Default Sit)"
        if current_state.shape[1] == agent.state_size:
            # --- 5g. Get RL Action & Final Action ---
            print(f"   [5g] 獲取 {stock_code} 的 RL 決策...")
            rl_action = agent.act(current_state); action_map = {0: "Sit", 1: "Buy", 2: "Sell"}
            final_action_stock = rl_action; action_reason = f"RL Agent ({action_map.get(rl_action, '?')})"
            if stop_loss_triggered:
                if final_action_stock != 2: print(f"      ACTION OVERRIDE: Stop-Loss. Forcing SELL."); final_action_stock = 2
                action_reason = "Stop-Loss" + (" & RL" if rl_action == 2 else "")
            print(f"      Final Action for {stock_code}: {final_action_stock} ({action_reason})")
        else: print(f"      ERROR: Failed creating RL state for {stock_code}. Sitting.")


        # --- 5h. Set Order Prices & Calculate Shares ---
        print(f"   [5h] 設定 {stock_code} 的下單價格與數量...")
        buy_order_price=predicted_open; sell_order_price=predicted_open
        if DAY_TRADE_SELL_METHOD=='profit_target': day_trade_sell_price=buy_order_price*(1+DAY_TRADE_PROFIT_TARGET_PERCENT/100.0)
        else: day_trade_sell_price=predicted_close if predicted_close is not None else buy_order_price
        print(f"      Pred: Open={predicted_open:.2f}, Close={predicted_close:.2f}. BuyP:{buy_order_price:.2f}, SellP:{sell_order_price:.2f}, DayTP:{day_trade_sell_price:.2f}")
        # Calculate shares
        shares_to_trade=MIN_SHARES_PER_TRADE; shares_to_sell_calculated=MIN_SHARES_PER_TRADE
        if predicted_open > 0:
             calc_shares = int(TARGET_TRADE_VALUE / (predicted_open * 1000.0))
             shares_to_trade = max(MIN_SHARES_PER_TRADE, calc_shares)
             shares_to_sell_calculated = min(actual_shares_held, shares_to_trade) if final_confirmed_holding else 0
             print(f"      Calc Shares: TargetVal {TARGET_TRADE_VALUE:,.0f} / ({predicted_open:.2f}*1k) => {calc_shares} lots. Trade: {shares_to_trade} lots. Max Sell: {shares_to_sell_calculated} lots.")
        else: print(f"      WARN: Pred Open <= 0. Using min shares: {shares_to_trade} lot(s).")

        # --- 5i. Execute Orders & Set Intention ---
        print(f"   [5i] 執行 {stock_code} 的預約單...")
        intended_action_stock = "sit"; buy_sent = False; sell_sent = False
        # ... (Order placement logic - same as V1.9.2) ...
        if final_action_stock == 1:
            if not final_confirmed_holding:
                print(f"      ACTION: Placing BUY {shares_to_trade} sh @ {buy_order_price:.2f}")
                if Buy_Stock(user_account_stock, user_password_stock, stock_code, shares_to_trade, buy_order_price):
                     buy_sent=True; intended_action_stock="buy"
                     print(f"      ACTION: Placing DayTrade SELL {shares_to_trade} sh @ {day_trade_sell_price:.2f}")
                     if Sell_Stock(user_account_stock, user_password_stock, stock_code, shares_to_trade, day_trade_sell_price): sell_sent=True
                     else: print("      WARN: DayTrade SELL order failed.")
                else: print("      WARN: BUY order failed.")
            else: print("      INFO: BUY action, but already holding.")
        elif final_action_stock == 2:
            if final_confirmed_holding:
                 if shares_to_sell_calculated >= MIN_SHARES_PER_TRADE:
                     print(f"      ACTION: Placing SELL {shares_to_sell_calculated} sh @ {sell_order_price:.2f} (Reason: {action_reason})")
                     if Sell_Stock(user_account_stock, user_password_stock, stock_code, shares_to_sell_calculated, sell_order_price): sell_sent=True; intended_action_stock="sell"
                     else: print("      WARN: SELL order failed.")
                 else: print(f"      INFO: SELL action, but sellable shares ({shares_to_sell_calculated}) < min ({MIN_SHARES_PER_TRADE}).")
            else: print("      INFO: SELL action, but not holding.")
        else: print("      INFO: SIT action."); intended_action_stock = "sit"

        # Update trade_info dict
        holding_intended = (intended_action_stock=="buy") or (final_confirmed_holding and intended_action_stock!="sell")
        price_intended = buy_order_price if intended_action_stock=="buy" else (current_purchase_price if holding_intended else None)
        trade_info[stock_code] = {"holding": holding_intended, "purchase_price": price_intended, "intended_action": intended_action_stock}
        actions_summary.append({'stock': stock_code, 'action': intended_action_stock, 'shares': shares_to_trade if intended_action_stock=='buy' else (shares_to_sell_calculated if intended_action_stock=='sell' else 0), 'buy_sent': buy_sent, 'sell_sent': sell_sent})

    # --- 6. Save All Trade Info ---
    print("\n[Step 6/N] 保存所有股票的交易意圖...")
    save_trade_info(trade_info); print("[OK] Trade intentions saved.")

    # --- Final Summary ---
    print("\n" + "="*60); print("=" + " "*14 + "【盤前】固定列表股票決策與預約單完成" + " "*14 + "=")
    print(f"處理股票數: {len(target_stock_codes)}")
    for summary in actions_summary: print(f"  - {summary['stock']}: Intended={summary['action']}, Shares={summary['shares']}, BuySent={summary['buy_sent']}, SellSent={summary['sell_sent']}")
    print("=" + " "*9 + "實際交易結果需在明日開市後或收盤後查詢確認" + " "*9 + "="); print("="*60 + "\n")


# === Main Execution Block ===
if __name__ == "__main__":
    print("="*40); print("   股票交易機器人 (v1.9.3 - 固定列表/目標金額/延遲訓練)"); print("="*40)
    start_run_time = datetime.now(); print(f"啟動時間: {start_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用固定股票列表: {STOCK_LIST_FILE}"); print(f"止損比例: {STOP_LOSS_PERCENTAGE}%")
    print(f"目標交易金額: {TARGET_TRADE_VALUE:,.0f} TWD")

    # 1. Optional: Execute Training (Run this manually when needed)
    #    Ensure Scaler files exist before running training!
    # print("\n檢查是否執行 RL 訓練...")
    # training_rl_model() # <<< RUN THIS SEPARATELY TO TRAIN THE MODEL!

    # 2. Execute Pre-Market Trading Logic
    print("\n執行盤前交易決策流程...")
    trading_execution()

    end_run_time = datetime.now(); print(f"\n程式執行完畢: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}"); print(f"總耗時: {end_run_time - start_run_time}")
from datetime import datetime
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
from collections import deque
import sys
import requests
import price_prediction

#考慮開收高低 ， 考慮前7天後預測 ， 7天訓練一次 ， 輸出開收盤價



# 取得股票資訊
# Input:
#   stock_code: 股票ID
#   start_date: 開始日期，YYYYMMDD
#   stop_date: 結束日期，YYYYMMDD
# Output: 持有股票陣列
def Get_Stock_Informations(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])



# 取得持有股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
# Output: 持有股票陣列
def Get_User_Stocks(account, password):
    data = {'account': account,
            'password': password
            }
    search_url = 'http://140.116.86.242:8081/stock/api/v1/get_user_stocks'
    result = requests.post(search_url, data=data).json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])



# 預約購入股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 購入張數
#   stock_price: 購入價格
# Output: 是否成功預約購入(True/False)
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Buying stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    buy_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
    result = requests.post(buy_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'



# 預約售出股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 售出張數
#   stock_price: 售出價格
# Output: 是否成功預約售出(True/False)
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Selling stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    sell_url = 'http://140.116.86.242:8081/stock/api/v1/sell'
    result = requests.post(sell_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'





#-----------------------------------------------------------------------------------------------------------------------------------------------------------



#格式化字串
def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n))



#拿取股票系統資料庫的歷史股價資訊。
#預設拿取2330(台積電)。
#建議大家把code改成從網路上抓取自己想要訓練的股價csv檔下來，然後再正規化股價資訊。
def getStockDataVec():
    vec_close = []
    vec_high = []
    vec_low = []
    stock_history_data = requests.get("http://140.116.86.242:8081/api/stock/get_stock_history_data_for_ce_bot").json()["data"]
    for history in stock_history_data:
        vec_close.append(history["Close"])
        vec_high.append(history["High"])
        vec_low.append(history["Low"])
    vec_close = vec_close[-100:]
    vec_high = vec_high[-100:]
    vec_low = vec_low[-100:]
    return vec_close, vec_high, vec_low



# Sigmoid function for getting RL state.
def sigmoid(x):
    return 1/(1+math.exp(-x))



# Getting state for RL training.
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])





# Reinforcement Learning Agent
class Agent:

    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model(model_name) if is_eval else self._model()


    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model


    def act(self, state):
        if not self.is_eval and random.random()<= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state) # Model predicting the action.
        return np.argmax(options[0]) # Returning the index of maximum value in the array.

    
    #If the agent memory gets full, expReplay will reset the memory.
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





#
# Training stock data
#
def training_model():
    stock_name = "TSMC"
    window_size = 10
    episode_count = 0
    agent = Agent(window_size)
    data, vec_high, vec_low = getStockDataVec()
    l = len(data) - 1
    batch_size = 32


    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state) # Agent predicts the action.
            print(action)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            print("next_state")
            print(next_state)

            reward = 0
            price_reward = data[t + 1] - data[t]
            if action == 1: # buy
                agent.inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))
            
            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = window_size_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0) # Getting reward when selling stocks.
                total_profit += data[t] - bought_price
                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done)) # Agent memory import current state.
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

            #If the agent memory gets full, expReplay will reset the memory.
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
        
        # Saving the model when finishing training each episode.
        if e % 10 == 0:
            agent.model.save("model_" + str(e))

    print(agent.memory)





#-----------------------------------------------------------------------------------------------------------------------------------------------------------





#
# Actual trading after training RL model
#
def trading():
    stock_name = "TSMC"
    model_name = "model_0"
    model = load_model(model_name)
    window_size = model.layers[0].input.shape.as_list()[1]
    agent = Agent(window_size, True, model_name)
    data, vec_high, vec_low = getStockDataVec()
    l = len(data) - 2
    batch_size = 32

    reward = 0
    if len(agent.memory) != 0:
        state_array = agent.memory.pop()
        agent.memory.append(state_array)
        state = state_array[3]
        pre_state = getState(data, l, window_size + 1)

        # Reward counting, you can design your own reward mechanism.
        if state[0][-1] >= pre_state[0][-1]:
            reward += 1

        # If reward is greater than zero, then using the current state to predict the action.
        # Else then using the pre state to predict the action.
        if reward > 0:
            action = agent.act(state) # Agent predicts the action.
        else:
            action = agent.act(pre_state) # Agent predicts the action.
            state = pre_state
    else:
        state = getState(data, l, window_size + 1)
        action = agent.act(state) # Agent predicts the action.
    print(state)
    print(action)


    # Predicting future price by using LSTM in price_prediction.py
    predicted_stock_price = price_prediction.predict_stock_price()
    print("predicted_stock_price")
    print(predicted_stock_price)



    # 本 sample code 使用股票系統的 CIoT Bot-CE 帳號進行下單，並且預設下單2330(台積電)，張數為1張。
    # 若要自行下單，請將code更改成自己在股票系統中的帳號密碼。
    # 若欲下單其他股票，請自行更改股票代碼以及下單張數。
    today = datetime.today().strftime('%Y%m%d')
    user_stocks = Get_User_Stocks("CIoT Bot-CE", "CIoT Bot-CE")  # 取得使用者持有股票
    today_stock_information = Get_Stock_Informations("2330", '20200101', today)  # 取得選定股票最新資訊


    if(len(today_stock_information) == 0):  # 若選定股票沒有任何資訊
        print('未曾開市')
    else:
        if action == 1: # buy
            agent.inventory.append(predicted_stock_price) # Agent inventory import current state.
            Buy_Stock("CIoT Bot-CE", "CIoT Bot-CE", 2330, 1, predicted_stock_price) 
        elif action == 2 and len(agent.inventory) > 0 and len(user_stocks) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = max(predicted_stock_price - bought_price, 0) # Getting reward when selling stocks.
            Sell_Stock("CIoT Bot-CE", "CIoT Bot-CE", 2330, 1, predicted_stock_price)
        elif action != 0: # Action is not 'sit'.
            agent.inventory.append(predicted_stock_price) # Agent inventory import current state.
            Buy_Stock("CIoT Bot-CE", "CIoT Bot-CE", 2330, 1, predicted_stock_price) 

        next_state = getState(data, l + 1, window_size + 1)
        agent.memory.append((state, action, reward, next_state, True)) # Agent memory import current state.

        #If the agent memory gets full, expReplay will reset the memory.
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)



# training_model() 
trading()



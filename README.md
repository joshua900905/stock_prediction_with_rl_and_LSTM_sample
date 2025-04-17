# 程式下單範例程式碼(RL、LSTM)

-   從股票系統資料庫抓取歷史股票資訊來做股價預測以及判斷股票買賣
1. 股價預測運用LSTM:
    (1) Normalize the training set
    (2) Create the LSTM model
    (3) Normalize the testing set
    (4) LSTM model predicts future stock price
2. 股票買賣判斷運用Reinforcement Learning:
    (1) Create RL agent
    (2) Get current agent state
    (3) Agent predicts next action through state
    (4) Getting reward from environment
    (5) Getting next state and looping from step 3 until last state


# Installation

-   Install keras for RL by using pip

```bash=
pip3 install keras
```


# How to run?

-   Execute the python file to run the RL and LSTM sample code

```bash=
python3 stock_agent.py
```

-   model_0 is the model that finishing training by reinforcement learning.


# Reminding

-   本 sample code 使用股票系統的 CIoT Bot-CE 帳號進行下單，並且預設下單2330(台積電)，張數為1張。
-   若要自行下單，請將code更改成自己在股票系統中的帳號密碼。
-   若欲下單其他股票，請自行更改股票代碼以及下單張數。


# Result

-   RL訓練結果會輸出0(不買不賣)、1(買入)、2(賣出)三種行為。
-   若有成功買入or賣出，可至股票系統的股票交易分頁查看自行下的單。


# Reference

-   Predicting Stock Prices using Reinforcement Learning (with Python Code!)
-   https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/


#   s t o c k _ p r e d i c t i o n _ w i t h _ r l _ a n d _ L S T M _ s a m p l e  
 
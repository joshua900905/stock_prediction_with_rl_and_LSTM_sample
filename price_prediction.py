import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import requests

def predict_stock_price():

    #拿取股票系統資料庫的歷史股價資訊。
    #預設拿取2330(台積電)。
    #建議大家把code改成從網路上抓取自己想要訓練的股價csv檔下來，然後再正規化股價資訊。
    training_set = []
    stock_history_data = requests.get("http://140.116.86.242:8081/api/stock/get_stock_history_data_for_ce_bot").json()["data"]
    for history in stock_history_data:
        training_set.append([history["Close"]])


    sc= MinMaxScaler()
    training_set=sc.fit_transform(training_set)

    # Normalize the training set
    train_size = len(training_set)
    X_train= training_set[0:train_size - 1]
    y_train= training_set[1:train_size]
    X_train=np.reshape(X_train, (train_size - 1 , 1 , 1))


    # Create the LSTM model
    regressor = Sequential()
    regressor.add(LSTM(units=4, activation= 'sigmoid', input_shape= (None,1)))
    regressor.add(Dense( units=1 ))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=32, epochs=200)


    #拿取股票系統資料庫的歷史股價資訊。
    #預設拿取2330(台積電)。
    #建議大家把code改成從網路上抓取自己想要訓練的股價csv檔下來，然後再正規化股價資訊。
    real_stock_price = []
    test_set = requests.get("http://140.116.86.242:8081/api/stock/get_stock_history_data_for_ce_bot").json()["data"]
    for history in test_set:
        real_stock_price.append([history["Close"]])


    # Normalize the testing set
    inputs = real_stock_price
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(real_stock_price) , 1, 1))
    # LSTM model predicts future stock price
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(predicted_stock_price)

    return predicted_stock_price[-1][0]

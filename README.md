執行 stock_agent.py (版本 1.9.1，固定列表版) 來進行每日盤前交易決策，需要按順序完成一系列準備和執行步驟。

【重要】執行前的準備工作（只需要做一次，或在模型/配置更新時做）：

步驟 1：設置 Python 環境與安裝依賴庫

確保已安裝 Python (建議 3.7+)。

創建並激活 Python 虛擬環境 (推薦)。

將 requirements.txt 文件保存在項目目錄。

在激活的虛擬環境中運行：pip install -r requirements.txt

步驟 2：準備必要的文件

將 stock_agent.py (v1.9.1) 和 price_prediction_v1.py (v1.1，支持多股票並返回 Scaler) 放在同一個項目目錄下。

創建 stock_list.txt 文件，填入你的 20 支目標股票代碼，用逗號分隔。

步驟 3：【初始化】生成 LSTM 模型和 Scaler 文件

目的: 為 stock_list.txt 中的每一支股票創建初始的 LSTM 模型 (.h5) 和對應的數據縮放器 (.pkl)。這是 stock_agent.py 運行的基礎。

操作:

修改 price_prediction_v1.py: 找到文件末尾的 if __name__ == '__main__': 區塊，用我們之前提供的、用於遍歷 stock_list.txt 並強制訓練的版本替換它。

運行初始化: 在終端（激活虛擬環境）中執行：

python price_prediction_v1.py


等待完成: 這個過程會為列表中的每支股票訓練模型，可能需要較長時間。觀察輸出，確保沒有關鍵錯誤。成功後，lstm_models 目錄下應包含每支股票的 .h5, .pkl, .txt 文件。

還原(可選): 初始化完成後，你可以選擇將 price_prediction_v1.py 的 if __name__ == '__main__': 改回簡單的測試代碼或註釋掉，但保留初始化版本也沒問題。

步驟 4：【訓練】生成強化學習 (RL) 模型

目的: 訓練一個能夠理解 OHLC+持股狀態並在延遲確認環境下做決策的 RL Agent 模型。

操作:

打開 stock_agent.py 文件。

找到 training_rl_model() 函數。 這個函數的邏輯必須與作業的延遲確認要求相匹配（v1.9.1 中提供的版本已包含此模擬邏輯，但你需要驗證其效果或進一步調整）。

在 training_rl_model() 函數內部，找到 episode_count 變量，將其設置為一個大於 0 的數值（例如 100, 500, 甚至更多，取決於模型收斂速度）。

設置 model_save_prefix 來定義保存的模型文件名前綴。

在終端（激活虛擬環境）中執行訓練:

python stock_agent.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(注意：這次運行主要是為了執行 training_rl_model 內部設置的訓練迴圈)

等待訓練完成: 這會非常耗時（幾小時到幾天都有可能）。觀察訓練過程中的平均利潤等指標，判斷模型是否收斂。

結果: 訓練完成後，你會在項目目錄下找到根據 model_save_prefix 生成的 RL 模型 .h5 文件（例如 rl_agent_multi_ohlc_hold_delay_ep100.h5）。你需要根據訓練過程中的表現選擇一個效果最好的模型文件。

步驟 5：最終配置 stock_agent.py

再次打開 stock_agent.py。

【關鍵】更新 RL 模型路徑: 找到 trading_execution() 函數內的 rl_model_load_path 變量，將其值修改為你在步驟 4 中訓練得到的【最佳】 RL 模型的文件名。

填寫帳號信息: 仔細檢查並替換 USER_ACCOUNT, USER_PASSWORD, TEAM_ACCOUNT_ID 為你的實際憑證。

【關鍵】禁用訓練模式: 找到文件末尾 if __name__ == '__main__': 區塊，確保對 training_rl_model() 的調用被註釋掉或 episode_count 在函數內部被設回 0。否則每次執行交易都會重新觸發漫長的訓練！

if __name__ == "__main__":
    # ... (startup messages) ...

    # --- Ensure training is DISABLED for daily execution ---
    # training_rl_model() # <<< COMMENT THIS OUT or set episode_count=0 inside it

    # --- Execute Trading Logic ---
    trading_execution()

    # ... (end messages) ...
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

檢查其他參數: 如 STOP_LOSS_PERCENTAGE, SHARES_TO_TRADE 等是否符合你的策略。

【每日】執行交易機器人 (盤前操作):

步驟 6：每日運行

時間: 在每個交易日的開盤前，作業規定的時間窗口內。

環境: 打開終端，進入項目目錄，激活 Python 虛擬環境。

執行: 運行命令：

python stock_agent.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

觀察輸出: 程式會依次執行 trading_execution 中的步驟：載入文件 -> 查詢歷史 -> 核對狀態 -> 查詢持股 -> 同步 -> 獲取數據 -> 止損檢查 -> 生成狀態 -> RL決策 -> LSTM預測 -> 下預約單 -> 保存意圖。注意觀察是否有錯誤信息。

步驟 7：隔日確認與監控 (人工)

第二天盤前 (或平台更新後): 登錄模擬交易平台。

核對成交: 查看前一天下的預約單（買單和可能的賣單）是否成功成交？成交價格是多少？

核對持股: 查看平台顯示的當前持股。

運行腳本: 再次運行 python stock_agent.py。觀察步驟 4 和 5 的輸出，看程式通過查詢歷史和當前持股进行的狀態同步是否與你在平台上看到的結果一致。特別注意成本價 (Cost Basis) 是否被正確更新（如果買入成功）。

檢查狀態文件: 打開 trade_status_multi_fixed.json，確認裡面記錄的 holding 和 purchase_price 是否與實際情況相符。

步驟 8：模型維護 (可選)

定期（例如數周或數月）使用最新的市場數據重新運行步驟 4 (RL 模型訓練)，以獲得可能更適應近期市場狀況的 RL 模型，並更新 RL_MODEL_LOAD_PATH。

LSTM 模型會根據 RETRAIN_INTERVAL_DAYS 自動重新訓練，通常無需手動干預，除非你想強制更新。

簡而言之，核心流程是：環境設置 -> 文件準備 -> LSTM 初始化 -> RL 訓練 -> 配置 Agent -> 每日運行 Agent -> 隔日監控確認。 其中 LSTM 初始化和 RL 訓練是前期最關鍵且耗時的步驟。
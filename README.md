# IEEE-CIS 信用卡詐騙偵測

本專案旨在建立一個機器學習模型，用於預測 IEEE-CIS 提供的信用卡交易數據集中的詐騙行為。專案流程包含離線模型訓練，以及模擬線上學習情境，透過概念漂移檢測來觸發模型更新。

## 資料集說明

數據集包含兩個主要的 CSV 檔案，通過 `TransactionID` 進行關聯：
*   `train_transaction.csv`: 包含交易資訊，如交易金額 (`TransactionAmt`)、產品代碼 (`ProductCD`)、卡片資訊 (`card1-6`) 等。
*   `train_identity.csv`: 包含用戶身份與設備資訊，如設備類型 (`DeviceType`) 等。

目標變數為 `isFraud`，一個二元標籤，`1` 代表詐騙，`0` 代表正常交易。

## 專案設置

1.  **下載資料集**:
    - 前往 Kaggle 競賽頁面: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
    - 下載所需的 CSV 檔案 (例如 `train_transaction.csv`, `train_identity.csv` 等)。

2.  **建立資料夾**:
    - 在專案根目錄下建立一個名為 `datasets` 的資料夾。
    - 將所有從 Kaggle 下載的 `.csv` 檔案放入 `datasets` 資料夾中。

## 環境需求

- **Python 版本**: 3.8+
- **主要依賴套件**:
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `lightgbm`
    - `river`

建議使用 `pip` 安裝所有依賴：
```bash
pip install -r requirements.txt
```

## 如何運行

本專案提供兩種實驗運行腳本：

1.  **`run_exp.py`**: 執行基線模型和不同特徵工程策略的比較實驗。

    ```bash
    python run_exp.py
    ```

2.  **`run_exp_incre.py`**: 模擬線上增量學習情境，並在偵測到概念漂移時更新模型。

    ```bash
    python run_exp_incre.py
    ```

## 專案模組說明

`run_exp.py` 和 `run_exp_incre.py` 是主執行腳本，它們會依序調用以下核心模組來完成整個實驗流程。

### `src/data/datahandler.py`
- **功能**: 負責所有數據的準備工作。
- `DataHandler` 類:
    - `load()`: 載入 `transaction` 和 `identity`數據並合併。
    - `add_time_features()`: 新增基於時間的特徵，如 `TransactionDay`。
    - `handle_missing_values()`: 統一填補缺失值。
    - `encode_categorical()`: 將類別特徵轉換為數值編碼。
    - `split_train_test()`: 將數據切分為用於離線訓練的訓練集和用於線上模擬的測試集。

### `src/features/engineer_with_cache.py`
- **功能**: 創建基於時間窗口的滾動聚合特徵。
- `FeatureEngineerWithCache` 類:
    - `create_feature()`: 針對 `card1`, `addr1` 等關鍵欄位，計算在不同時間窗口（如1天、7天）內的交易次數、平均金額等統計特徵。
    - 支持特徵快取，避免重複計算。

### `src/models/baseline.py`
- **功能**: 訓練初始的離線模型。
- `BaselineModel` 類:
    - `train()`: 使用 `DataHandler` 切分出的訓練集來訓練一個 LightGBM 基線模型。這個模型將作為線上學習的起點。
    - `predict()`: 提供預測功能。

### `src/models/incremental_learner.py`
- **功能**: 模擬線上學習環境，並在偵測到概念漂移時更新模型。
- `IncrementalLearner` 類:
    - `predict_batch()`: 對批次數據進行預測。
    - `update()`: 使用 `DriftDetector` 檢查預測錯誤，當偵測到數據分佈發生顯著變化（漂移）時，觸發 `_retrain` 流程，使用新的數據重新訓練模型。

### `src/models/drift_detector.py`
- **功能**: 概念漂移檢測器。
- `DriftDetector` 類:
    - 基於 `ADWIN` (Adaptive Windowing) 演算法。
    - `update()`: 接收模型的預測錯誤（0或1），並判斷當前的錯誤率是否發生了統計上的顯著變化，以判斷是否發生概念漂移。

## 實驗結果

實驗結果會被記錄在 `outputs/` 文件夾中。例如 `results_xgb.csv` 記錄了不同實驗設定下的模型表現指標：

```csv
exp_id,windows_size,precision,recall,AUC
baseline,None,0.202,0.6099,0.8408
e1,1,0.2069,0.6053,0.8424
e6,1+3+7,0.2124,0.6099,0.8497
...
```

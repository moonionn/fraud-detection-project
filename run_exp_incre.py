"""
Incremental Learning Experiment Runner
Usage: python run_exp_incre.py
"""
import os
import pandas as pd
from src.data.datahandler import DataHandler
from src.features.engineer_with_cache import FeatureEngineerWithCache
from src.models.model import train_model
from src.utils.evaluator import ModelEvaluator
from src.models.incremental_learner import IncrementalLearner

# ==================== Configs ====================
# 選擇要使用的特徵工程實驗 ID ('baseline', 'e1', 'e6', etc.)
# 'baseline' 表示不使用額外的窗口特徵
EXP_ID = 'e6'
USE_CACHE = True  # 是否使用特徵快取

MODEL_TYPE = 'lgb'  # 選擇模型類型 ('lgb', 'xgb', 'rf')
SAMPLING_TYPE = 'smote'  # Options: 'none', 'undersample', 'oversample', 'smote', 'smoteenn'
USE_DRIFT_DETECTION = True
# ================================================

# 從 run_exp.py 複製實驗設定
EXPERIMENT_CONFIGS = {
    'baseline': {'windows': [], 'name': 'Baseline (No Window Features)'},
    'e1': {'windows': [1], 'name': '1-day Window'},
    'e1a': {'windows': [2], 'name': '2-day Window'},
    'e2': {'windows': [3], 'name': '3-day Window'},
    'e2a': {'windows': [5], 'name': '5-day Window'},
    'e3': {'windows': [7], 'name': '7-day Window'},
    'e4': {'windows': [14], 'name': '14-day Window'},
    'e5': {'windows': [30], 'name': '30-day Window'},
    'e6': {'windows': [1, 3, 7], 'name': '1+3+7-day Windows'},
    'e6b': {'windows': [1, 14], 'name': '1+14-day Windows'},
    'e7': {'windows': [1, 2], 'name': '1+2-day Windows'},
    'e8': {'windows': [1, 2, 3], 'name': '1+2+3-day Windows'},
}


def run_incremental_learning_simulation(
        exp_id,
        model_type,
        sampling_type,
        use_drift_detection=True,
        use_cache=True,
        **model_params
):
    """執行增量學習模擬"""
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    config = EXPERIMENT_CONFIGS[exp_id]
    windows = config['windows']
    exp_name = config['name']

    print('=' * 60)
    print('Incremental Learning Simulation')
    print(f'Feature Set: {exp_id.upper()} ({exp_name})')
    print(f'Model: {model_type.upper()} | Drift Detection: {use_drift_detection}')
    print('=' * 60 + '\n')

    # --- 資料準備 ---
    print('[Step 1] Data Preparation\n')
    processed_path = 'datasets/processed/processed_data.csv'
    data = pd.read_csv(processed_path)
    handler = DataHandler(data_dir='datasets', offline_weeks=8)

    # --- 特徵工程 (在分割資料前對整個數據集進行) ---
    if windows:
        print(f'[Step 2] Create {exp_name} Features for the entire dataset\n')
        # 注意：這裡我們需要一個方法來為整個數據集生成特徵
        # 我們可以暫時將整個數據集視為 "train" 來生成特徵
        engineer = FeatureEngineerWithCache(windows=windows, use_cache=use_cache)
        data = engineer.create_feature(data, data_type='full_dataset_for_incremental')
        print(f'  Total features: {len(data.columns)}\n')

        # 處理新增特徵產生的 NaN
        if data.isnull().any().any():
            print('  Handling missing values in new features...')
            data = handler.handle_missing_values(data)
        else:
            print('  No missing values in new features.')
    else:
        print('[Step 2] Skip Feature Engineering (Baseline)\n')

    # 將資料分為初始訓練集和後續的流式數據集
    initial_train_data, stream_data = handler.split_train_test(data)
    print(f'  Initial training data: {len(initial_train_data)} samples')
    print(f'  Stream data for simulation: {len(stream_data)} samples\n')

    # --- 初始模型訓練 ---
    print(f'[Step 3] Train Initial {model_type.upper()} Model\n')
    base_model, feature_cols = train_model(
        initial_train_data,
        model_type=model_type,
        use_sampling=True,
        sampling_type=sampling_type,
        **model_params
    )

    # --- 初始化增量學習器 ---
    print('[Step 4] Initialize Incremental Learner\n')
    learner = IncrementalLearner(
        base_model=base_model,
        feature_cols=feature_cols,
        model_type=model_type,
        sampling_type=sampling_type,  # 用於重新訓練
        use_drift_detection=use_drift_detection
    )

    # --- 模擬流式預測與更新 ---
    print('[Step 5] Simulate Stream Processing (by day)\n')
    all_y_true = []
    all_y_pred_proba = []
    all_y_pred = []

    # 按天分組作為批次
    stream_batches = stream_data.groupby('TransactionDay')
    total_batches = len(stream_batches)

    for i, (day, batch_data) in enumerate(stream_batches):
        print(f'Processing Batch {i + 1}/{total_batches} (Day {day}, {len(batch_data)} samples)...')
        y_pred_proba, y_pred = learner.predict_batch(batch_data)
        all_y_true.extend(batch_data['isFraud'].values)
        all_y_pred_proba.extend(y_pred_proba)
        all_y_pred.extend(y_pred)
        learner.update(batch_data)

    print('\nStream processing completed!\n')

    # --- 評估整體性能 ---
    print('[Step 6] Final Evaluation\n')
    metrics = ModelEvaluator.evaluate(
        pd.Series(all_y_true),
        pd.Series(all_y_pred_proba),
        pd.Series(all_y_pred)
    )
    print(f"  Overall Precision: {metrics.get('Precision', 0):.4f}")
    print(f"  Overall Recall: {metrics.get('Recall', 0):.4f}")
    print(f"  Overall AUC: {metrics.get('AUC', 0):.4f}")
    print(f"  Overall AUC-PR: {metrics.get('AUC_PR', 0):.4f}\n")

    # --- 顯示學習器統計資訊 ---
    print('[Step 7] Learner Statistics\n')
    stats = learner.get_statistics()
    print(f"  Total predictions made: {stats['total_predictions']:,}")
    print(f"  Total retrains triggered: {stats['retrain_count']}\n")

    print('Incremental Learning Simulation Completed!')


def main():
    """執行增量學習模擬"""
    processed_path = 'datasets/processed/processed_data.csv'
    if not os.path.exists(processed_path):
        print('預處理資料不存在，開始執行資料預處理...')
        handler = DataHandler(data_dir='datasets', offline_weeks=8)
        data = handler.load()
        data = handler.add_time_features(data)
        data = handler.handle_missing_values(data)
        data = handler.encode_categorical(data)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        data.to_csv(processed_path, index=False)
        print('已儲存預處理資料至 datasets/processed/processed_data.csv')
    else:
        print('已偵測到 processed_data.csv，直接使用。')

    run_incremental_learning_simulation(
        exp_id=EXP_ID,
        model_type=MODEL_TYPE,
        sampling_type=SAMPLING_TYPE,
        use_drift_detection=USE_DRIFT_DETECTION,
        use_cache=USE_CACHE
    )


if __name__ == '__main__':
    main()
"""
Run All Experiments (Static + Incremental)
一次執行所有靜態實驗和增量學習實驗

Usage: python run_all_experiments.py
"""
import os
import csv
import pandas as pd
from datetime import datetime
from src.data.datahandler import DataHandler
from src.features.engineer_with_cache import FeatureEngineerWithCache
from src.models.model import train_model, predict_model
from src.utils.evaluator import ModelEvaluator
from src.models.incremental_learner import IncrementalLearner

# ==================== Configs ====================
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

# 模型與採樣設定
MODEL_TYPE = 'xgb'          # lgb, xgb, rf, etc.
SAMPLING_TYPE = 'smote'     # none, undersample, oversample, smote, smoteenn etc.
USE_CACHE = True

# ==================== 靜態實驗配置 ====================
RUN_ALL_STATIC = False
STATIC_EXP_IDS = ['baseline', 'e1', 'e1a', 'e2', 'e2a', 'e3', 'e4', 'e5', 'e6', 'e6b', 'e7', 'e8']

# ==================== 增量學習配置 ====================
RUN_INCREMENTAL = True
INCREMENTAL_EXP_IDS = ['baseline', 'e1', 'e1a', 'e2', 'e2a', 'e3', 'e4', 'e5', 'e6', 'e6b', 'e7', 'e8']

# 增量學習專屬參數（僅在 RUN_INCREMENTAL=True 時生效）
USE_DRIFT_DETECTION = True     # 是否使用 ADWIN 漂移檢測
BATCH_DAYS = 7                  # 每個批次的天數
WINDOW_SIZE = 2                 # 訓練視窗大小 (保留最近 N 個批次)

# 輸出設定
OUTPUT_DIR = 'results/all_experiments'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
STATIC_CSV = f'{OUTPUT_DIR}/static_{MODEL_TYPE}_{SAMPLING_TYPE}_{TIMESTAMP}.csv'
INCREMENTAL_CSV = f'{OUTPUT_DIR}/incremental_{MODEL_TYPE}_{SAMPLING_TYPE}_{TIMESTAMP}.csv'
SUMMARY_CSV = f'{OUTPUT_DIR}/summary_{TIMESTAMP}.csv'


def run_static_experiment(exp_id, data, handler, model_type, sampling_type, use_cache):
    """執行單一靜態實驗"""
    config = EXPERIMENT_CONFIGS[exp_id]
    windows = config['windows']
    exp_name = config['name']

    print(f'\n{"=" * 60}')
    print(f'Static Experiment: {exp_id.upper()}')
    print(f'Config: {exp_name}')
    print(f'{"=" * 60}')

    # 特徵工程
    if windows:
        print(f'\n[1] Create {exp_name} Features')
        engineer = FeatureEngineerWithCache(windows=windows, use_cache=use_cache)
        processed_data = engineer.create_feature(data.copy(), data_type=f'static_{exp_id}')

        if processed_data.isnull().any().any():
            processed_data = handler.handle_missing_values(processed_data)
    else:
        print(f'\n[1] Skip Feature Engineering (Baseline)')
        processed_data = data.copy()

    # 切分資料
    train_data, test_data = handler.split_train_test(processed_data)

    # 訓練模型
    print(f'\n[2] Train {model_type.upper()} Model')
    model, feature_cols = train_model(
        train_data,
        model_type=model_type,
        use_sampling=True,
        sampling_type=sampling_type
    )

    # 預測
    print(f'\n[3] Predict')
    y_pred_proba, y_pred = predict_model(model, test_data, feature_cols, model_type=model_type)

    # 評估
    print(f'\n[4] Evaluate')
    y_true = test_data['isFraud']
    metrics = ModelEvaluator.evaluate(y_true, y_pred_proba, y_pred)

    print(f"  Precision: {metrics.get('Precision', 0):.4f}")
    print(f"  Recall: {metrics.get('Recall', 0):.4f}")
    print(f"  AUC: {metrics.get('AUC', 0):.4f}")
    print(f"  AUC-PR: {metrics.get('AUC_PR', 0):.4f}")

    return {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'windows': '+'.join(map(str, windows)) if windows else 'None',
        'model_type': model_type,
        'sampling_type': sampling_type,
        'precision': metrics.get('Precision', 0),
        'recall': metrics.get('Recall', 0),
        'auc': metrics.get('AUC', 0),
        'auc_pr': metrics.get('AUC_PR', 0),
        'mode': 'static'
    }


def run_incremental_experiment(exp_id, data, handler, model_type, sampling_type,
                               use_cache, use_drift_detection, batch_days, window_size):
    """執行單一增量學習實驗"""
    config = EXPERIMENT_CONFIGS[exp_id]
    windows = config['windows']
    exp_name = config['name']

    print(f'\n{"=" * 60}')
    print(f'Incremental Experiment: {exp_id.upper()}')
    print(f'Config: {exp_name}')
    print(f'Batch: {batch_days} days | Window: {window_size} batches')
    print(f'{"=" * 60}')

    # 特徵工程
    if windows:
        print(f'\n[1] Create {exp_name} Features')
        engineer = FeatureEngineerWithCache(windows=windows, use_cache=use_cache)
        processed_data = engineer.create_feature(data.copy(), data_type=f'incre_{exp_id}')

        if processed_data.isnull().any().any():
            processed_data = handler.handle_missing_values(processed_data)
    else:
        print(f'\n[1] Skip Feature Engineering (Baseline)')
        processed_data = data.copy()

    # 切分資料
    initial_train_data, stream_data = handler.split_train_test(processed_data)

    # 訓練初始模型
    print(f'\n[2] Train Initial {model_type.upper()} Model')
    base_model, feature_cols = train_model(
        initial_train_data,
        model_type=model_type,
        use_sampling=True,
        sampling_type=sampling_type
    )

    # 初始化增量學習器
    print(f'\n[3] Initialize Incremental Learner')
    learner = IncrementalLearner(
        base_model=base_model,
        feature_cols=feature_cols,
        model_type=model_type,
        sampling_type=sampling_type,
        use_drift_detection=use_drift_detection,
        window_size=window_size
    )

    # 模擬串流處理
    print(f'\n[4] Simulate Stream Processing')
    all_y_true = []
    all_y_pred_proba = []
    all_y_pred = []

    stream_batches = handler.create_batches(stream_data)
    total_batches = len(stream_batches)

    for i, batch_info in enumerate(stream_batches):
        batch_data = batch_info['data']

        if (i + 1) % 5 == 0 or i == 0:
            print(f'  Processing batch {i + 1}/{total_batches}...')

        y_pred_proba, y_pred = learner.predict_batch(batch_data)
        all_y_true.extend(batch_data['isFraud'].values)
        all_y_pred_proba.extend(y_pred_proba)
        all_y_pred.extend(y_pred)
        learner.update(batch_data, y_pred)

    # 評估
    print(f'\n[5] Evaluate')
    metrics = ModelEvaluator.evaluate(
        pd.Series(all_y_true),
        pd.Series(all_y_pred_proba),
        pd.Series(all_y_pred)
    )

    stats = learner.get_statistics()

    print(f"  Precision: {metrics.get('Precision', 0):.4f}")
    print(f"  Recall: {metrics.get('Recall', 0):.4f}")
    print(f"  AUC: {metrics.get('AUC', 0):.4f}")
    print(f"  AUC-PR: {metrics.get('AUC_PR', 0):.4f}")
    print(f"  Retrains: {stats['retrain_count']}")
    print(f"  Drifts: {stats.get('drift_count', 0)}")

    return {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'windows': '+'.join(map(str, windows)) if windows else 'None',
        'model_type': model_type,
        'sampling_type': sampling_type,
        'batch_days': batch_days,
        'window_size': window_size,
        'window_days': batch_days * window_size,
        'drift_detection': use_drift_detection,
        'precision': metrics.get('Precision', 0),
        'recall': metrics.get('Recall', 0),
        'auc': metrics.get('AUC', 0),
        'auc_pr': metrics.get('AUC_PR', 0),
        'total_predictions': stats['total_predictions'],
        'total_batches': stats['total_batches'],
        'retrain_count': stats['retrain_count'],
        'drift_count': stats.get('drift_count', 0),
        'mode': 'incremental'
    }


def main():
    """執行所有實驗"""
    print(f'\n{"=" * 80}')
    print(f'Run All Experiments')
    print(f'Static: {RUN_ALL_STATIC} | Incremental: {RUN_INCREMENTAL}')
    print(f'Model: {MODEL_TYPE} | Sampling: {SAMPLING_TYPE}')
    print(f'{"=" * 80}\n')

    # 確保預處理資料存在
    processed_path = 'datasets/processed/processed_data.csv'
    if not os.path.exists(processed_path):
        print('預處理資料不存在，開始執行資料預處理...')
        handler = DataHandler(data_dir='datasets', offline_weeks=8, batch_days=BATCH_DAYS)
        data = handler.load()
        data = handler.add_time_features(data)
        data = handler.handle_missing_values(data)
        data = handler.encode_categorical(data)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        data.to_csv(processed_path, index=False)
        print('已儲存預處理資料\n')
    else:
        print('已偵測到 processed_data.csv\n')

    # 載入資料
    data = pd.read_csv(processed_path)
    handler = DataHandler(data_dir='datasets', offline_weeks=8, batch_days=BATCH_DAYS)

    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    # ========== 執行靜態實驗 ==========
    if RUN_ALL_STATIC:
        print(f'\n{"#" * 80}')
        print(f'# Part 1: Static Experiments ({len(STATIC_EXP_IDS)} experiments)')
        print(f'{"#" * 80}')

        static_results = []
        for i, exp_id in enumerate(STATIC_EXP_IDS, 1):
            print(f'\n[Static {i}/{len(STATIC_EXP_IDS)}]')

            try:
                result = run_static_experiment(
                    exp_id, data, handler, MODEL_TYPE, SAMPLING_TYPE, USE_CACHE
                )
                static_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f'  Error in {exp_id}: {e}')
                continue

        # 儲存靜態實驗結果
        static_fieldnames = ['exp_id', 'exp_name', 'windows', 'model_type', 'sampling_type',
                             'precision', 'recall', 'auc', 'auc_pr', 'mode']

        with open(STATIC_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=static_fieldnames)
            writer.writeheader()
            writer.writerows(static_results)

        print(f'\n靜態實驗結果已儲存: {STATIC_CSV}')

    # ========== 執行增量學習實驗 ==========
    if RUN_INCREMENTAL:
        print(f'\n{"#" * 80}')
        print(f'# Part 2: Incremental Experiments ({len(INCREMENTAL_EXP_IDS)} experiments)')
        print(f'{"#" * 80}')

        incremental_results = []
        for i, exp_id in enumerate(INCREMENTAL_EXP_IDS, 1):
            print(f'\n[Incremental {i}/{len(INCREMENTAL_EXP_IDS)}]')

            try:
                result = run_incremental_experiment(
                    exp_id, data, handler, MODEL_TYPE, SAMPLING_TYPE, USE_CACHE,
                    USE_DRIFT_DETECTION, BATCH_DAYS, WINDOW_SIZE
                )
                incremental_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f'  Error in {exp_id}: {e}')
                continue

        # 儲存增量學習實驗結果
        incremental_fieldnames = ['exp_id', 'exp_name', 'windows', 'model_type', 'sampling_type',
                                  'batch_days', 'window_size', 'window_days', 'drift_detection',
                                  'precision', 'recall', 'auc', 'auc_pr',
                                  'total_predictions', 'total_batches', 'retrain_count',
                                  'drift_count', 'mode']

        with open(INCREMENTAL_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=incremental_fieldnames)
            writer.writeheader()
            writer.writerows(incremental_results)

        print(f'\n增量學習實驗結果已儲存: {INCREMENTAL_CSV}')

    # ========== 儲存總結 ==========
    summary_fieldnames = ['exp_id', 'mode', 'windows', 'precision', 'recall', 'auc', 'auc_pr']

    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # ========== 顯示結果摘要 ==========
    print(f'\n{"=" * 80}')
    print(f'All Experiments Completed!')
    print(f'{"=" * 80}')
    print(f'\nResults saved to:')
    if RUN_ALL_STATIC:
        print(f'  Static: {STATIC_CSV}')
    if RUN_INCREMENTAL:
        print(f'  Incremental: {INCREMENTAL_CSV}')
    print(f'  Summary: {SUMMARY_CSV}')

    # 顯示結果表格
    print(f'\n{"=" * 80}')
    print(f'Results Summary:')
    print(f'{"=" * 80}\n')
    print(f'{"-" * 100}')
    print(f'{"Exp":<10} {"Mode":<12} {"Windows":<15} {"Precision":<12} {"Recall":<12} {"AUC":<12} {"AUC-PR":<12}')
    print(f'{"-" * 100}')

    for r in all_results:
        print(f'{r["exp_id"]:<10} {r["mode"]:<12} {r["windows"]:<15} '
              f'{r["precision"]:<12.4f} {r["recall"]:<12.4f} '
              f'{r["auc"]:<12.4f} {r["auc_pr"]:<12.4f}')

    print(f'{"-" * 100}')


if __name__ == '__main__':
    main()
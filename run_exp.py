"""
Universal experiment runner
Usage: python run_exp.py
"""
import os
import csv
import pandas as pd
from src.data.datahandler import DataHandler
from src.features.engineer_with_cache import FeatureEngineerWithCache
from src.models.model import train_model, predict_model
from src.utils.evaluator import ModelEvaluator

# ==================== Configs ====================
EXP_IDS = ['baseline', 'e1', 'e1a', 'e2', 'e2a', 'e3', 'e4', 'e5', 'e6', 'e6b', 'e7', 'e8']
MODEL_TYPE = 'xgb'
SAMPLING_TYPE = 'smote'  # Options: 'none', 'undersample', 'oversample', 'smote', 'smoteenn'
USE_CACHE = True
OUTPUT_CSV = f'results_{MODEL_TYPE}_{SAMPLING_TYPE}.csv'
# ================================================

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

def run_experiment(
        exp_id,
        model_type,
        sampling_type,
        use_cache=True,
        **model_params
):
    """執行單一實驗"""
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment ID: {exp_id}. "
                         f"Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")

    config = EXPERIMENT_CONFIGS[exp_id]
    windows = config['windows']
    exp_name = config['name']

    print('=' * 60)
    print(f'{exp_id.upper()} Experiment: {exp_name}')
    print(f'Model: {model_type.upper()}')
    print('=' * 60 + '\n')

    # Step 1-3: Data Preparation
    print('[Step 1-3] Data Preparation\n')
    processed_path = 'datasets/processed/processed_data.csv'
    data = pd.read_csv(processed_path)
    handler = DataHandler(data_dir='datasets', offline_weeks=8)

    # Step 4: Feature Engineering (修改點：先計算特徵再切分)
    if windows:
        print(f'[Step 4] Create {exp_name} Features\n')
        engineer = FeatureEngineerWithCache(windows=windows, use_cache=use_cache)

        # 在整個資料集上計算特徵
        data = engineer.create_feature(data, data_type=f'static_{exp_id}')
        print(f'  Total features: {len(data.columns)}\n')

        # 處理新增特徵產生的 NaN
        if data.isnull().any().any():
            print('  Handling missing values in new features...')
            data = handler.handle_missing_values(data)
        else:
            print('  No missing values in new features.')
    else:
        print('[Step 4] Skip Feature Engineering (Baseline)\n')

    # 切分訓練/測試集（移到特徵工程之後）
    train_data, test_data = handler.split_train_test(data)

    # Step 5: 訓練模型
    print(f'[Step 5] Train {model_type.upper()} Model\n')
    print(f"  Using sampling: {sampling_type}")

    model, feature_cols = train_model(
        train_data,
        model_type=model_type,
        use_sampling=True,
        sampling_type=sampling_type,
        **model_params
    )

    # Step 6: Predict
    print('[Step 6] Predict\n')
    y_pred_proba, y_pred = predict_model(model, test_data, feature_cols, model_type=model_type)

    # Step 7: Evaluate
    print('[Step 7] Evaluate\n')
    y_true = test_data['isFraud']
    metrics = ModelEvaluator.evaluate(y_true, y_pred_proba, y_pred)

    precision = metrics.get('Precision', 0)
    recall = metrics.get('Recall', 0)
    auc = metrics.get('AUC', 0)
    auc_pr = metrics.get('AUC_PR', 0)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC: {auc:.4f}\n")
    print(f"  AUC-PR: {auc_pr:.4f}\n")

    print(f'{exp_id.upper()} Experiment Completed!')

    return {'precision': precision, 'recall': recall, 'AUC': auc, 'AUC_PR': auc_pr}


def main():
    """執行所有實驗並儲存結果"""
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

    print('\n' + '=' * 80)
    print(f'執行 {len(EXP_IDS)} 個實驗: {", ".join(EXP_IDS)}')
    print('=' * 80 + '\n')

    results = []
    for i, exp_id in enumerate(EXP_IDS, 1):
        print(f'[Progress: {i}/{len(EXP_IDS)}]\n')

        metrics = run_experiment(
            exp_id=exp_id,
            model_type=MODEL_TYPE,
            use_cache=USE_CACHE,
            sampling_type=SAMPLING_TYPE
        )

        config = EXPERIMENT_CONFIGS[exp_id]
        windows_size = '+'.join(map(str, config['windows'])) if config['windows'] else 'None'
        results.append({
            'exp_id': exp_id,
            'windows_size': windows_size,
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'AUC': round(metrics['AUC'], 4),
            'AUC_PR': round(metrics['AUC_PR'], 4),
        })

    # Write results to CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['exp_id', 'windows_size', 'precision', 'recall', 'AUC', 'AUC_PR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print('\n' + '=' * 80)
    print('All experiments completed!')
    print('=' * 80)
    print(f'\nThe results is saved as: {OUTPUT_CSV}')

    # Print results summary
    print('Result:\n')
    print('-' * 95)
    print(f"{'exp_id':<12} {'windows size':<20} {'Precision':<12} {'Recall':<12} {'AUC':<12} {'AUC_PR':<12}")
    print('-' * 95)
    for r in results:
        print(f"{r['exp_id']:<12} {r['windows_size']:<20} {r['precision']:<12} {r['recall']:<12} {r['AUC']:<12} {r['AUC_PR']:<12}")
    print('-' * 95)


if __name__ == '__main__':
    main()
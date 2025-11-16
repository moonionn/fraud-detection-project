"""
E1 實驗：Baseline + 1天窗口特徵（帶緩存版）
"""

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.features.engineer_with_cache import FeatureEngineerWithCache
from src.models.baseline import BaselineModel
from src.models.evaluator import ModelEvaluator


def run_e1_experiment(use_full_data=False):
    print('=' * 60)
    print('E1 Experiment: Baseline + 1-day Window Features (Cached)')
    print('=' * 60)

    # Step 1: Load Data
    print('\n[Step 1] Load Data')
    loader = DataLoader('datasets')
    data = loader.load()

    # Step 2: Preprocess
    print('\n[Step 2] Preprocess Data')
    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    # Step 3: Split Data
    print('\n[Step 3] Split Data')
    splitter = DataSplitter(offline_weeks=8)
    train_data, test_data = splitter.split_train_test(data)

    # 限制數據量（測試模式）
    if not use_full_data:
        print('\n[TEST MODE] Limiting data size...')
        train_data = train_data.head(50000).copy()
        test_data = test_data.head(50000).copy()
        print(f'  Train: {len(train_data):,} samples')
        print(f'  Test: {len(test_data):,} samples')

    # Step 4: Feature Engineering (1-day window) with Cache
    print('\n[Step 4] Create 1-day Window Features (with Cache)')
    engineer = FeatureEngineerWithCache(windows=[1], use_cache=True)

    # 使用緩存創建特徵
    train_data = engineer.create_feature(train_data, data_type='train')
    test_data = engineer.create_feature(test_data, data_type='test')

    print(f'\n  ✓ Total features: {len(train_data.columns)}')

    # Step 5: Train Model
    print('\n[Step 5] Train Model')
    model = BaselineModel()
    model.train(train_data, use_sampling=True)

    # Step 6: Predict
    print('\n[Step 6] Predict')
    y_pred_proba, y_pred = model.predict(test_data)

    # Step 7: Evaluate
    print('\n[Step 7] Evaluate')
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(test_data['isFraud'], y_pred_proba, y_pred)
    evaluator.print_metrics(metrics)

    print('\n' + '=' * 60)
    print('✓ E1 Experiment Completed!')
    print('=' * 60)

    return metrics


if __name__ == '__main__':
    metrics = run_e1_experiment(use_full_data=True)

    # 完整模式：改成 use_full_data=True
    # metrics = run_e1_experiment(use_full_data=True)
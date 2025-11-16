from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.features.egineer import FeatureEngineer
from src.models.baseline import BaselineModel
from src.models.evaluator import ModelEvaluator


def run_e1_experiment():
    print('=' * 60)
    print('E1 Experiment: Baseline + 1-day Window Features')
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

    # **測試版：限制數據量**
    print('\n[TEST MODE] Limiting data size...')
    train_data = train_data.head(50000).copy()
    test_data = test_data.head(50000).copy()
    print(f'  Train: {len(train_data):,} samples')
    print(f'  Test: {len(test_data):,} samples')

    # Step 4: Feature Engineering (1-day window)
    print('\n[Step 4] Create 1-day Window Features')
    engineer = FeatureEngineer(windows=[1])

    print('  Creating features for train data...')
    train_data = engineer.create_feature(train_data)

    print('  Creating features for test data...')
    test_data = engineer.create_feature(test_data)

    print(f'  Total features: {len(train_data.columns)}')

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
    metrics = run_e1_experiment()
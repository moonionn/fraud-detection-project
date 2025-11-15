from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.baseline import BaselineModel
from src.models.evaluator import ModelEvaluator


def run_base_line_experiment():
    print('|__Run Baseline Experiment (E0)__|')

    print('[Step 1] Load Data')
    loader = DataLoader('datasets')
    data = loader.load()

    print('[Step 2] Preprocess Data')
    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    print('[Step 3] Split Data')
    splitter = DataSplitter(offline_weeks=8)
    train_data, test_data = splitter.split_train_test(data)

    print('[Step 4] Train Baseline Model')
    model = BaselineModel()
    model.train(train_data, use_sampling=True)

    print('[Step 5] Predict Model')
    y_pred_proba, y_pred = model.predict(test_data)

    print('[Step 6] Evaluate Model')
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(test_data['isFraud'], y_pred_proba, y_pred)
    evaluator.print_metrics(metrics)

    print('SUCCESS: Baseline model training and evaluation completed.')

    return metrics

if __name__ == '__main__':
    run_base_line_experiment()


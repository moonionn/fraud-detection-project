import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from fontTools.designspaceLib.split import LOGGER
import time


class BaselineModel:

    def __init__(self, params=None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.model = None
        self.feature_cols = []
        self.train_time = 0

    def prepare_feature(self, data):
        exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT',
                        'TransactionDay', 'TIMESTAMP', 'Cents']
        self.feature_cols = [col for col in data.columns if col not in exclude_cols]

        X = data[self.feature_cols]
        y = data['isFraud']

        return X, y

    def train(self, train_data, use_sampling: bool = True):
        print('|__Train Baseline Model__|')

        X_train, y_train = self.prepare_feature(train_data)
        print(f'  Features: {len(self.feature_cols)}')
        print(f'  Training samples: {len(X_train):,} | Fraud rate: {y_train.mean():.2%}')

        if use_sampling:
            print(f'Applying undersampling...')
            sampler = RandomOverSampler(sampling_strategy=1.0, random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f'  After sampling: {len(X_train):,} | Fraud rate: {y_train.mean():.2%}')\

        print('Training the model...')
        train_data_lgb = lgb.Dataset(X_train, label=y_train)

        start_time = time.time()
        self.model = lgb.train(
            self.params,
            train_data_lgb,
            num_boost_round=500,
            valid_sets=[train_data_lgb],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        self.train_time = time.time() - start_time
        print(f'  Train time: {self.train_time:.2f} sec')
        print('SUCCESS: Model training completed.')

        return self

    def predict(self, test_data):
        print('|__Predict Baseline Model__|')
        if self.model is None:
            raise ValueError('FAILED: Model not trained. Call train() first.')
        X_test, _ = self.prepare_feature(test_data)

        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred_proba , y_pred


if __name__ == '__main__':
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.data.splitter import DataSplitter

    loader = DataLoader('datasets')
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    splitter = DataSplitter(offline_weeks=8)
    train_data, test_data = splitter.split_train_test(data)

    model = BaselineModel()
    model.train(train_data)

    y_pred_proba, y_pred = model.predict(test_data)

    print(f'Predictions completed: {len(y_pred)} samples predicted.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from fontTools.designspaceLib.split import LOGGER


class FeatureEngineer:

    def __init__(self, windows=[1, 3, 7, 14, 30]):
        self.windows = windows
        self.group_keys = ['card1', 'card2', 'addr1']
        self.agg_features = {
            'TransactionAmt': ['mean', 'std', 'min', 'max'],
            'TransactionID': ['count']
        }
        self.feature_names = []

    def create_feature(self, data):
        print('|___Create Window Features___|')
        print(f'  Windows: {self.windows} days')

        # Sort data by TransactionDT
        data = data.sort_values('TransactionDT').reset_index(drop=True)
        feature_count = 0

        for group_key in tqdm(self.group_keys, desc='Group Keys'):
            if group_key not in data.columns:
                continue
            for window_days in self.windows:
                window_seconds = window_days * 24 * 3600

                # Calculate rolling features for each aggregation
                for agg_col, agg_funcs in self.agg_features.items():
                    for agg_func in agg_funcs:
                        feat_name = f'{group_key}_{agg_col}_{agg_func}_{window_days}d'

                        # Compute rolling feature
                        data[feat_name] = data.groupby(group_key, observed=True).apply(
                            lambda x: self._rolling_agg(x, window_seconds, agg_col, agg_func),
                            include_groups=False
                        ).reset_index(level=0, drop=True)

                        self.feature_names.append(feat_name)
                        feature_count += 1
        print(f'  Feature Count: {feature_count}')
        print('SUCCESS: Created window features.')

        return data

    def _rolling_agg(self, group_data, window_seconds, value_col, agg_func):
        result = []

        for idx, row in group_data.iterrows():
            current_time = row['TransactionDT']
            window_start = current_time - window_seconds

            window_data = group_data[
                (group_data['TransactionDT'] >= window_start) &
                (group_data['TransactionDT'] < current_time)
            ]
            if len(window_data) == 0:
                result.append(0 if agg_func == 'count' else np.nan)
            else:
                if agg_func == 'count':
                    result.append(len(window_data))
                elif agg_func == 'mean':
                    result.append(window_data[value_col].mean())
                elif agg_func == 'std':
                    result.append(window_data[value_col].std())
                elif agg_func == 'min':
                    result.append(window_data[value_col].min())
                elif agg_func == 'max':
                    result.append(window_data[value_col].max())
                else:
                    result.append(np.nan)

        return pd.Series(result, index=group_data.index)

    def get_feature_names(self):
        return self.feature_names

if __name__ == '__main__':
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor

    loader = DataLoader('datasets')
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    test_data = data.head(10000).copy()

    engineer = FeatureEngineer(windows=[1, 7])
    test_data = engineer.create_feature(test_data)

    print('Result:')
    print(f'  Original Columns: {len(data.columns)}')
    print(f'  New Columns: {len(test_data.columns)}')
    print(f'  New Feature Columns: {len(engineer.get_feature_names())}')
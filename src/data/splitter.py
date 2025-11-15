import pandas as pd
from fontTools.designspaceLib.split import LOGGER


class DataSplitter:

    def __init__(self, offline_weeks=8, batch_days=7):
        self.offline_weeks = offline_weeks
        self.batch_days = batch_days

    def split_train_test(self, data):
        """
        按時間劃分訓練/測試集

        Args:
            data (pd.DataFrame): 包含 'TransactionDay' 欄位的數據集。

        Returns:
            offline_data, online_data (pd.DataFrame, pd.DataFrame): 劃分後的數據集。
        """
        print('|___Data Split___|')
        offline_days = self.offline_weeks * 7

        train_data = data[data['TransactionDay'] < offline_days].copy()
        test_data = data[data['TransactionDay'] >= offline_days].copy()

        print(f'Train (Day 0-{offline_days - 1}):')
        print(f'  Transactions: {len(train_data):,}')
        print(f'  Fraud Rate: {train_data["isFraud"].mean():.2%}')

        print(f'Test (Day {offline_days}+):')
        print(f'  Transactions: {len(test_data):,}')
        print(f'  Fraud Rate: {test_data["isFraud"].mean():.2%}')

        return train_data, test_data

    def create_batches(self, data):
        """
        將數據分批（用於增量學習）

        Args:
            data (pd.DataFrame): 包含 'TransactionDay' 欄位的數據集。

        Returns:
            batches (list): 每個批次的數據列表。
        """
        print('|___Create Batches__|')
        min_day = data['TransactionDay'].min()
        max_day = data['TransactionDay'].max()

        batches = []
        current_day = min_day

        while current_day <= max_day:
            batch_end = current_day + self.batch_days
            batch_data = data[
                (data['TransactionDay'] >= current_day) &
                (data['TransactionDay'] < batch_end)
            ].copy()
            if len(batch_data) > 0:
                batches.append({
                    'start_day': int(current_day),
                    'end_day': int(batch_end - 1),
                    'size': len(batch_data),
                    'fraud_rate': batch_data['isFraud'].mean(),
                    'data': batch_data
                })

            current_day = batch_end

        print(f'  Total Batches: {len(batches)}')
        print(f'  Batch Size (days): {self.batch_days}')
        print(f'SUCCESS: Created {len(batches)} batches of data.')

        return batches


if __name__ == '__main__':
    from loader import DataLoader
    from preprocessor import DataPreprocessor

    loader = DataLoader('datasets')
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    splitter = DataSplitter(offline_weeks=8, batch_days=7)
    train_data, test_data = splitter.split_train_test(data)
    batches = splitter.create_batches(test_data)

    print('First 3 Batch Info:')
    for i, batch in enumerate(batches[:3]):
        print(f'Batch {i}: Day {batch['start_day']}-{batch['end_day']} | '
                    f'Size: {batch['size']:,} | Fraud Rate: {batch['fraud_rate']:.2%}')
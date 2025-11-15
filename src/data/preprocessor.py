import pandas as pd
from fontTools.designspaceLib.split import LOGGER


class DataPreprocessor:

    def __init__(self):
        pass

    def add_time_features(self, data):
        print('|___Add Time Features___|')

        data['TimeDay'] = data['TransactionDT'] % 86400
        data['Cents'] = data['TransactionAmt'] % 1

        min_dt = data['TransactionDT'].min()
        data['TransactionDay'] = ((data['TransactionDT'] - min_dt) / 86400).astype('int32')

        print('   Added: TimeInDay, Cents, TransactionDay')
        print('SUCCESS: Time features added.')

        return data

    def process(self, data):
        data = self.add_time_features(data)
        return data


if __name__ == '__main__':
    from loader import DataLoader

    loader = DataLoader('./datasets')
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    print('Data processing completed.')
    print(f'  Shape: {data.shape}')
    print(f'  Time span (days): {data["TransactionDay"].max()} days')

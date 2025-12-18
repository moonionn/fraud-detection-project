import pandas as pd
import numpy as np


class DatasetStatistics:
    def __init__(self, data):
        self.data = data

    def print_basic_stats(self):
        print("=" * 60)
        print("IEEE-CIS Fraud Detection Dataset Statistics")
        print("=" * 60)

        print(f"Total Transactions: {len(self.data):,}")
        print(f"Total Features: {len(self.data.columns) - 1}")

        fraud_count = self.data['isFraud'].sum()
        fraud_rate = self.data['isFraud'].mean()
        print(f"Fraudulent Transactions: {fraud_count:,}")
        print(f"Legitimate Transactions: {len(self.data) - fraud_count:,}")
        print(f"Fraud Rate: {fraud_rate:.2%}")

        time_span = self.data['TransactionDay'].max() - self.data['TransactionDay'].min() + 1
        print(f"Time Span: {time_span} days (~{time_span / 30:.1f} months)")

        print("\n" + "=" * 60)
        print("Feature Categories")
        print("=" * 60)

        feature_categories = {
            'Transaction Info': ['TransactionAmt', 'ProductCD'],
            'Card Features': [c for c in self.data.columns if c.startswith('card')],
            'Address Features': [c for c in self.data.columns if c.startswith('addr')],
            'Distance Features': [c for c in self.data.columns if c.startswith('dist')],
            'Email Features': [c for c in self.data.columns if 'email' in c.lower()],
            'Count Features': [c for c in self.data.columns if c.startswith('C')],
            'Time Delta Features': [c for c in self.data.columns if c.startswith('D')],
            'Match Features': [c for c in self.data.columns if c.startswith('M')],
            'Vesta Features': [c for c in self.data.columns if c.startswith('V')],
            'Identity Features': [c for c in self.data.columns if c.startswith('id_')]
        }

        for category, features in feature_categories.items():
            count = len([f for f in features if f in self.data.columns])
            if count > 0:
                print(f"{category}: {count} features")

        print("=" * 60)


if __name__ == '__main__':
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor

    dataset_path = '/Users/moonion/ncue_paper/fraud-detection-project/datasets'
    loader = DataLoader(dataset_path)
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    stats = DatasetStatistics(data)
    stats.print_basic_stats()
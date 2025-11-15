import pandas as pd


class DataLoader:

    DTYPE = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'ProductCD': 'category',
        'card1': 'int16',
        'card2': 'float32',
        'card3': 'float32',
        'card4': 'category',
        'card5': 'float32',
        'card6': 'category',
        'addr1': 'float32',
        'addr2': 'float32',
        'dist1': 'float32',
        'dist2': 'float32',
        'P_emaildomain': 'category',
        'R_emaildomain': 'category',
    }

    DTYPE_ID = {
        'TransactionID': 'int32',
        'DeviceType': 'category',
        'DeviceInfo': 'category',
    }

    def __init__(self, data_dir='datasets'):
        self.data_dir = data_dir
        self.data = None

        # 動態添加 C, D, M, V 欄位類型
        self._setup_dtypes()

    def _setup_dtypes(self):
        # C1-C14: float32
        self.DTYPE.update({f'C{i}': 'float32' for i in range(1, 15)})

        # D1-D15: float32
        self.DTYPE.update({f'D{i}': 'float32' for i in range(1, 16)})

        # M1-M9: category
        self.DTYPE.update({f'M{i}': 'category' for i in range(1, 10)})

        # V1-V339: float32
        self.DTYPE.update({f'V{i}': 'float32' for i in range(1, 340)})

        # id_01 - id_38: float32
        self.DTYPE_ID.update({f'id_{i:02d}': 'category' for i in range(12, 39)})

        # Special cases for id_01 to id_11 treat as category
        ID_CATS = ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
                   'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35',
                   'id_36', 'id_37', 'id_38']
        self.DTYPE_ID.update({c: 'category' for c in ID_CATS})

    def load(self):
        print('|___Loading Data___|')
        train_trans = pd.read_csv(
            f'{self.data_dir}/train_transaction.csv',
            dtype=self.DTYPE,
            index_col='TransactionID'
        )
        print(f'  Transaction data: {train_trans.shape}')

        # Load identity data
        train_identity = pd.read_csv(
            f'{self.data_dir}/train_identity.csv',
            dtype=self.DTYPE_ID,
            index_col='TransactionID'
        )
        print(f'  Identity data: {train_identity.shape}')

        # Merge transaction and identity data
        self.data = train_trans.join(train_identity, how='left')
        self.data = self.data.reset_index()
        print(f'  Combined data: {self.data.shape}')

        return self.data

    def get_data(self):
        if self.data is None:
            raise ValueError('FAILED: Data not loaded. Call load() first.')
        return self.data


if __name__ == '__main__':
    data_loader = DataLoader(data_dir='datasets')
    data = data_loader.load()

    print(f'Data loaded: {data.shape}')
    print(f'  Fraud rate: {data["isFraud"].mean():.2%}')
    print(f'  Memory usage:{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
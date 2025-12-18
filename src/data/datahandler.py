import pandas as pd


class DataHandler:
    """整合資料載入、預處理、切分的完整流程"""

    def __init__(self, data_dir='datasets', offline_weeks=8, batch_days=7):
        self.data_dir = data_dir
        self.offline_weeks = offline_weeks
        self.batch_days = batch_days
        self.data = None
        self._setup_dtypes()

    def _setup_dtypes(self):
        """設定資料類型"""
        self.DTYPE = {
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

        self.DTYPE_ID = {
            'TransactionID': 'int32',
            'DeviceType': 'category',
            'DeviceInfo': 'category',
        }

        # C1-C14, D1-D15, V1-V339
        self.DTYPE.update({f'C{i}': 'float32' for i in range(1, 15)})
        self.DTYPE.update({f'D{i}': 'float32' for i in range(1, 16)})
        self.DTYPE.update({f'M{i}': 'category' for i in range(1, 10)})
        self.DTYPE.update({f'V{i}': 'float32' for i in range(1, 340)})
        self.DTYPE_ID.update({f'id_{i:02d}': 'category' for i in range(12, 39)})

        ID_CATS = ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
                   'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35',
                   'id_36', 'id_37', 'id_38']
        self.DTYPE_ID.update({c: 'category' for c in ID_CATS})

    def load(self):
        """載入資料"""
        print('|___Load Data___|')
        train_trans = pd.read_csv(
            f'{self.data_dir}/train_transaction.csv',
            dtype=self.DTYPE,
            index_col='TransactionID'
        )
        print(f'  Transaction data: {train_trans.shape}')

        train_identity = pd.read_csv(
            f'{self.data_dir}/train_identity.csv',
            dtype=self.DTYPE_ID,
            index_col='TransactionID'
        )
        print(f'  Identity data: {train_identity.shape}')

        self.data = train_trans.join(train_identity, how='left')
        self.data = self.data.reset_index()
        print(f'  Combined data: {self.data.shape}')

        return self.data

    @staticmethod
    def add_time_features(data):
        """新增時間特徵"""
        print('|___Add Time Features___|')

        data['TimeDay'] = data['TransactionDT'] % 86400
        data['Cents'] = data['TransactionAmt'] % 1

        min_dt = data['TransactionDT'].min()
        data['TransactionDay'] = ((data['TransactionDT'] - min_dt) / 86400).astype('int32')

        print('   Added: TimeInDay, Cents, TransactionDay')
        print('SUCCESS: Time features added.\n')

        return data

    @staticmethod
    def handle_missing_values(data):
        """統一處理缺失值"""
        print('|___Handle Missing Values___|')

        # 檢查缺失值數量
        nan_count = data.isnull().sum().sum()
        if nan_count == 0:
            print('  No missing values found.')
            return data

        print(f'  Found {nan_count:,} missing values, filling...')

        # 不處理的欄位 (ID 和目標變數)
        exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT']

        for col in data.columns:
            if col in exclude_cols:
                continue

            if data[col].isnull().any():
                # 類別型欄位:填補為 'missing'
                if data[col].dtype.name == 'category':
                    if 'missing' not in data[col].cat.categories:
                        data[col] = data[col].cat.add_categories(['missing'])
                    data[col] = data[col].fillna('missing')

                # 物件型欄位:填補為 'missing'
                elif data[col].dtype == 'object':
                    data[col] = data[col].fillna('missing')

                # 數值型欄位:填補為 0
                else:
                    data[col] = data[col].fillna(0)

        # 驗證是否還有缺失值
        remaining_nan = data.isnull().sum().sum()
        if remaining_nan > 0:
            print(f'  警告: 仍有 {remaining_nan} 個缺失值未處理')
            # 強制填補所有剩餘的 NaN
            data = data.fillna(0)
            print('  已強制將所有剩餘缺失值填補為 0')

        print('SUCCESS: Missing values handled.\n')
        return data

    def split_train_test(self, data):
        """切分訓練/測試集"""
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
        """建立批次資料"""
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
        print(f'SUCCESS: Created {len(batches)} batches of data.\n')

        return batches

    @staticmethod
    def encode_categorical(data):
        """將所有 category 欄位轉成數值型"""
        cat_cols = data.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            data[col] = data[col].astype('category').cat.codes
        print(f'  Encoded categorical columns: {list(cat_cols)}')
        return data

    def prepare_data(self):
        """
        執行完整的資料準備流程
        Steps:
            1. 載入資料
            2. 新增時間特徵
            3. 處理缺失值
            4. 欄位轉成數值型
            5. 切分訓練/測試集
        """
        data = self.load()
        data = self.add_time_features(data)
        data = self.handle_missing_values(data)
        data = self.encode_categorical(data)
        train_data, test_data = self.split_train_test(data)

        return train_data, test_data
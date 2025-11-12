import pandas as pd

# =================== Transaction data ===================
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

# C1-C14: float32
DTYPE.update({f'C{i}': 'float32' for i in range(1, 15)})

#D1-D15: float32
DTYPE.update({f'D{i}': 'float32' for i in range(1, 16)})

#M1-M9: category
DTYPE.update({f'M{i}': 'category' for i in range(1, 10)})

# V1-V339: float32
DTYPE.update({f'V{i}': 'float32' for i in range(1, 340)})

# =================== Identity data ===================
DTYPE_ID = {
    'TransactionID': 'int32',
    'DeviceType': 'category',
    'DeviceInfo': 'category',
}

# id_01 - id_38: float32
DTYPE_ID.update({f'id_{i:02d}': 'category' for i in range(12, 39)})

# Special cases for id_01 to id_11 treat as category
ID_CATS = ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
           'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35',
           'id_36', 'id_37', 'id_38']
DTYPE_ID.update({c: 'category' for c in ID_CATS})


def load_data(data_dir='datasets'):

    train_trans = pd.read_csv(f'{data_dir}/train_transaction.csv',
                              dtype=DTYPE, index_col='TransactionID')
    print(f'Transaction data shape: {train_trans.shape}')

    train_identity = pd.read_csv(f'{data_dir}/train_identity.csv',
                                 dtype=DTYPE_ID, index_col='TransactionID')
    print(f'Identity data shape: {train_identity.shape}')

    train = train_trans.join(train_identity, how='left')
    print(f'Combined train data shape: {train.shape}')

    train = train.reset_index()

    return train

def add_time_features(df):
    df['TIMESTAMP'] = df['TransactionDT'] % 86400

    df['Cents'] = df['TransactionAmt'] % 1

    min_dy = df['TransactionDT'].min()
    df['TransactionDT'] = ((df['TransactionDT'] - min_dy) / 86400).astype('int32')

    return df

if __name__ == '__main__':
    # TEST
    train = load_data('./datasets')
    train = add_time_features(train)

    print('|___Dataset Info___|')
    print(f'  Shape: {train.shape}')
    print(f'  Fraud count: {train["isFraud"].mean():.2%}')
    print(f'  Time span (days): {train["TransactionDT"].max()} days')
    print(f'  Memory usage: {train.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    print('SUCCESS: Got Dataset Info\n')
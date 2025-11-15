import matplotlib.pyplot as plt
from data_processing import load_data, add_time_features

# load data
data = load_data()
data = add_time_features(data)

print('|___Data Analysis___|')

# 1. Time span
print('|___1. Time Span___|')
daily_stats = data.groupby('TransactionDay').agg({
    'TransactionID': 'count',
    'isFraud': ['sum', 'mean'],
    'TransactionAmt': 'mean'
})
daily_stats.columns = ['TransactionCount', 'FraudCount', 'FraudRate', 'AvgTransactionAmt']

print(  f'Avg Daily Transactions per day: {daily_stats["TransactionCount"].mean():.0f}')
print(  f'Avg Daily Fraud Count per day: {daily_stats["FraudCount"].mean():.1f}')
print(  f'Overall Fraud Rate: {data["isFraud"].mean():.2%} ~ {daily_stats['FraudRate'].mean():.2%}')

# Fraud vs Normal
print('|___2. Fraud vs Normal___|')
fraud_stats = data.groupby('isFraud')['TransactionAmt'].describe()
print(  f'Fraud Stats: {fraud_stats}')

# Point features
print('|___3. Point Features___|')
cat_features = ['ProductCD', 'card4', 'card6']
for feat in cat_features:
    print(feat)
    print(data[feat].value_counts().head())

# Visualizations
print('|___4. Point Features___|')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Transaction Amount Pre Day
axes[0, 0].plot(daily_stats.index, daily_stats['AvgTransactionAmt'])
axes[0, 0].set_title('Avg Transaction Amount per Day')
axes[0, 0].set_xlabel('Transaction Day')
axes[0, 0].set_ylabel('Transaction Amount')
axes[0, 0].grid(alpha=0.3)

# Fraud Rate Pre Day
axes[0, 1].plot(daily_stats.index, daily_stats['FraudRate'] * 100, color='red')
axes[0, 1].set_title('Fraud Rate per Day')
axes[0, 1].set_xlabel('Transaction Day')
axes[0, 1].set_ylabel('Fraud Rate (%)')
axes[0, 1].grid(alpha=0.3)

# Transaction Count Distribution
data[data['TransactionAmt'] < 500].boxplot(column='TransactionAmt', by='isFraud', ax=axes[1, 0])
axes[1,0].set_title('Transaction Amount Span (< $500)')
axes[1,0].set_xlabel('isFraud')
axes[1,0].set_ylabel('Transaction Amount')

# ProductCD Distribution
product_fraud = data.groupby(['ProductCD', 'isFraud'], observed=True).size().unstack(fill_value=0)
product_fraud.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1,1].set_title('ProductCD Distribution by isFraud')
axes[1,1].set_xlabel('ProductCD')
axes[1,1].set_ylabel('Transaction Count')
axes[1,1].legend(['Normal', 'Fraud'])

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
print('SUCCESS: Picture saved as data_distribution.png\n')

plt.show()
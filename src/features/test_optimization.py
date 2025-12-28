# test_optimization.py
import pandas as pd
import os
from engineer_with_cache import FeatureEngineerWithCache

print("Testing optimized rolling aggregation...")

# 載入少量數據
project_root = '/Users/moonion/ncue_paper/fraud-detection-project'
data_path = os.path.join(project_root, 'datasets/processed/processed_data.csv')

data = pd.read_csv(data_path).head(10000)
print(f"Loaded {len(data)} rows for testing\n")

# 計算特徵
engineer = FeatureEngineerWithCache(windows=[1], use_cache=False)
result = engineer.create_feature(data.copy(), data_type='test_opt')

# 檢查結果
print("\nSample results:")
sample = result.head(10)
feature_cols = [col for col in result.columns if '_1d' in col]

for col in feature_cols[:5]:  # 只顯示前5個特徵
    print(f"\n{col}:")
    print(f"  Non-zero count: {(sample[col] != 0).sum()}/{len(sample)}")
    print(f"  Sample values: {sample[col].values[:3]}")

print("\nTest completed! If no errors, optimization is working correctly.")

if __name__ == "__main__":
    engineer = FeatureEngineerWithCache(windows=[1], use_cache=False)
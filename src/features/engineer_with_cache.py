import pandas as pd
import numpy as np
from tqdm import tqdm
from src.features.feature_cache import FeatureCacheManager

class FeatureEngineerWithCache:

    def __init__(self, windows=[1, 3, 7, 14, 30], use_cache=True):
        """
        初始化

        Args:
            windows: 時間窗口列表（天）
            use_cache: 是否使用緩存
        """
        self.windows = windows
        self.use_cache = use_cache
        self.group_keys = ['card1', 'card2', 'addr1']
        self.agg_features = {
            'TransactionAmt': ['mean', 'std', 'min', 'max'],
            'TransactionID': ['count']
        }
        self.feature_names = []
        self.cache_mgr = FeatureCacheManager() if use_cache else None

    def create_feature(self, data, data_type='train', force_compute=False):
        """
        創建多尺度窗口特徵（帶緩存）

        Args:
            data: DataFrame
            data_type: 數據類型（train/test）
            force_compute: 強制重新計算（忽略緩存）

        Returns:
            DataFrame: 添加特徵後的數據
        """
        # 檢查緩存
        if self.use_cache and not force_compute:
            if self.cache_mgr.cache_exists(self.windows, data_type):
                print(f'\n💡 Found cached features for {data_type} data')
                cached_data = self.cache_mgr.load_cache(self.windows, data_type)

                # 驗證數據大小是否一致
                if len(cached_data) == len(data):
                    print(f'  ✓ Cache validated')
                    return cached_data
                else:
                    print(f'  ⚠️  Cache size mismatch, recomputing...')

        # 計算特徵
        print(f'\n🔧 Computing window features for {data_type} data...')
        data = self._compute_features(data)

        # 保存緩存
        if self.use_cache:
            self.cache_mgr.save_cache(data, self.windows, data_type)

        return data

    def _compute_features(self, data):
        """實際計算特徵"""
        print(f'  Windows: {self.windows} days')

        # 確保按時間排序
        data = data.sort_values('TransactionDT').reset_index(drop=True)
        feature_count = 0

        for group_key in tqdm(self.group_keys, desc='Group Keys'):
            if group_key not in data.columns:
                continue

            for window_days in self.windows:
                window_seconds = window_days * 24 * 3600

                for agg_col, agg_funcs in self.agg_features.items():
                    for agg_func in agg_funcs:
                        feat_name = f'{group_key}_{agg_col}_{agg_func}_{window_days}d'

                        # 計算特徵
                        data[feat_name] = data.groupby(group_key, observed=True).apply(
                            lambda x: self._rolling_agg(x, window_seconds, agg_col, agg_func),
                            include_groups=False
                        ).reset_index(level=0, drop=True)

                        self.feature_names.append(feat_name)
                        feature_count += 1

        print(f'  ✓ Created {feature_count} features')

        return data

    def _rolling_agg(self, group_data, window_seconds, value_col, agg_func):
        """計算滾動窗口聚合"""
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
        """獲取特徵名稱"""
        return self.feature_names


if __name__ == '__main__':
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor

    # 測試
    loader = DataLoader('datasets')
    data = loader.load()

    preprocessor = DataPreprocessor()
    data = preprocessor.process(data)

    test_data = data.head(10000).copy()

    # 使用緩存
    engineer = FeatureEngineerWithCache(windows=[1], use_cache=True)

    # 第一次：計算並緩存
    print('\n--- First run ---')
    test_data = engineer.create_feature(test_data, data_type='test')

    # 第二次：從緩存讀取
    print('\n--- Second run ---')
    test_data2 = engineer.create_feature(data.head(10000).copy(), data_type='test')

    print(f'\n✓ Features match: {test_data.shape == test_data2.shape}')

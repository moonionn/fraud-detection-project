import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.feature_cache import FeatureCacheManager


class FeatureEngineerWithCache:

    def __init__(self, windows, use_cache=True):
        """
        初始化

        Args:
            windows: 時間窗口列表(天)
            use_cache: 是否使用緩存
        """
        if not windows:
            raise ValueError('Windows list cannot be empty!')
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
        創建多尺度窗口特徵(帶緩存)

        Args:
            data: DataFrame
            data_type: 資料類型(train/test)
            force_compute: 強制重新計算(忽略緩存)

        Returns:
            DataFrame: 新增特徵後的資料
        """
        # 檢查緩存
        if self.use_cache and not force_compute:
            if self.cache_mgr.cache_exists(self.windows, data_type):
                print(f'\nFound cached features for {data_type} data')
                cached_data = self.cache_mgr.load_cache(self.windows, data_type)

                # 驗證資料大小是否一致
                if len(cached_data) == len(data):
                    print(f'  ✓ Cache validated')
                    return cached_data
                else:
                    print(f'  Cache size mismatch, recomputing...')

        # 計算特徵
        print(f'\n🔧 Computing window features for {data_type} data...')
        data = self._compute_features(data)

        # 處理缺失值
        data = self._handle_missing_values(data)

        # 儲存緩存
        if self.use_cache:
            self.cache_mgr.save_cache(data, self.windows, data_type)

        return data

    def _compute_features(self, data):
        """實際計算特徵"""
        print(f'  Windows: {self.windows} days')

        self.feature_names = []

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

    def _handle_missing_values(self, data):
        """處理窗口特徵的缺失值"""
        nan_count = data.isnull().sum().sum()

        if nan_count > 0:
            print(f'  處理 {nan_count} 個缺失值...')

            # 只處理新建的窗口特徵
            for feat_name in self.feature_names:
                if feat_name in data.columns:
                    # count 特徵用 0 填補(代表沒有歷史交易)
                    if '_count_' in feat_name:
                        data[feat_name] = data[feat_name].fillna(0)
                        # std 特徵: 單筆交易無法計算標準差 → 填 0
                    elif '_std_' in feat_name:
                        data[feat_name] = data[feat_name].fillna(0)
                    # 統計特徵用 0 填補(代表窗口內無資料)
                    else:
                        data[feat_name] = data[feat_name].fillna(0)

            print(f'  ✓ 缺失值處理完成')

        return data

    def get_feature_names(self):
        """獲取特徵名稱"""
        return self.feature_names


if __name__ == '__main__':
    # 修正 import
    from src.data.datahandler import DataHandler

    # 測試
    handler = DataHandler('datasets')
    train_data, test_data = handler.prepare_data()

    test_data = test_data.head(10000).copy()

    # 使用緩存
    engineer = FeatureEngineerWithCache(windows=[1], use_cache=True)

    # 第一次:計算並緩存
    print('\n--- First run ---')
    test_data = engineer.create_feature(test_data, data_type='test')

    # 第二次:從緩存讀取
    print('\n--- Second run ---')
    test_data2 = engineer.create_feature(test_data.head(10000).copy(), data_type='test')
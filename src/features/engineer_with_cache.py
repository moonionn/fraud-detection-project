import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.feature_cache import FeatureCacheManager


class FeatureEngineerWithCache:

    def __init__(self, windows, use_cache=True, use_compression=True):
        """
        初始化

        Args:
            windows: 時間窗口列表(天)
            use_cache: 是否使用緩存
            use_compression: 是否壓縮快取檔案
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
        self.cache_mgr = FeatureCacheManager(use_compression=use_compression) if use_cache else None

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

                # 載入快取並合併到原始資料
                result = self.cache_mgr.load_cache(
                    self.windows,
                    data_type,
                    merge_with=data
                )

                # 驗證資料大小是否一致
                if len(result) == len(data):
                    print('  Cache validated')
                    # 更新 feature_names（從快取的欄位推斷）
                    self.feature_names = [col for col in result.columns
                                          if col not in data.columns]
                    return result
                else:
                    print('  Cache size mismatch, recomputing...')

        # 計算特徵
        print(f'\nComputing window features for {data_type} data...')
        data = self._compute_features(data)

        # 處理缺失值
        data = self._handle_missing_values(data)

        # 儲存緩存（只存窗口特徵）
        if self.use_cache:
            self.cache_mgr.save_cache(
                data,
                self.windows,
                data_type,
                feature_names=self.feature_names  # 只存窗口特徵
            )

        return data

    def _compute_features(self, data):
        """實際計算特徵（優化版）"""
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

                        # 使用優化的滾動聚合
                        data[feat_name] = data.groupby(group_key, observed=True).apply(
                            lambda x: self._rolling_agg_optimized(x, window_seconds, agg_col, agg_func),
                            include_groups=False
                        ).reset_index(level=0, drop=True)

                        self.feature_names.append(feat_name)
                        feature_count += 1

        print(f'  Created {feature_count} features')

        return data

    def _rolling_agg_optimized(self, group_data, window_seconds, value_col, agg_func):
        """優化的滾動窗口聚合 - O(n) 時間複雜度"""
        sorted_data = group_data.sort_values('TransactionDT').reset_index(drop=True)

        result = []

        # 預先準備所有交易的時間和值
        times = sorted_data['TransactionDT'].values
        values = sorted_data[value_col].values
        n = len(sorted_data)

        left_ptr = 0  # 窗口左指針

        for i in range(n):
            current_time = times[i]
            window_start = current_time - window_seconds

            # 移動左指針，移除窗口外的交易
            while left_ptr < i and times[left_ptr] < window_start:
                left_ptr += 1

            # 窗口範圍：[left_ptr, i)（不包括 i）
            window_values = values[left_ptr:i]

            # 計算聚合值
            if len(window_values) == 0:
                if agg_func == 'count':
                    result.append(0)
                else:
                    result.append(np.nan)
            else:
                if agg_func == 'count':
                    result.append(len(window_values))
                elif agg_func == 'mean':
                    result.append(np.mean(window_values))
                elif agg_func == 'std':
                    result.append(np.std(window_values, ddof=0))
                elif agg_func == 'min':
                    result.append(np.min(window_values))
                elif agg_func == 'max':
                    result.append(np.max(window_values))
                else:
                    result.append(np.nan)

        result_series = pd.Series(result, index=sorted_data.index)
        return result_series.reindex(group_data.index)

    def _handle_missing_values(self, data):
        """處理窗口特徵的缺失值"""
        nan_count = data.isnull().sum().sum()

        if nan_count > 0:
            print(f'  Handling {nan_count} missing values...')

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

            print('  Missing values handled')

        return data

    def get_feature_names(self):
        """獲取特徵名稱"""
        return self.feature_names
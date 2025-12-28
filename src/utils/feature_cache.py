import os
from os.path import exists
import pandas as pd
import pickle
import gzip
from pathlib import Path


class FeatureCacheManager:

    def __init__(self, cache_dir='datasets/cache', use_compression=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression

    def get_cache_key(self, windows, data_type='train'):
        windows_str = '_'.join(map(str, sorted(windows)))
        ext = '.pkl.gz' if self.use_compression else '.pkl'
        return f'features_{data_type}_w{windows_str}{ext}'

    def cache_exists(self, windows, data_type='train'):
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)
        return cache_file.exists()

    def load_cache(self, windows, data_type='train', merge_with=None):
        """
        載入快取特徵

        Args:
            windows: 時間窗口列表
            data_type: 資料類型
            merge_with: 如果提供，將快取特徵合併到此 DataFrame

        Returns:
            DataFrame: 快取的特徵或合併後的完整資料
        """
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)

        if not cache_file.exists():
            raise FileNotFoundError(f'Cache not found: {cache_file}')

        print(f'  Loading cache from: {cache_file.name}')

        # 根據是否壓縮選擇載入方式
        if self.use_compression:
            with gzip.open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
        else:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

        # 檢查 TransactionID 唯一性
        if cached_data['TransactionID'].duplicated().any():
            raise ValueError("Cache contains duplicate TransactionID!")

        # 如果提供原始資料，合併特徵
        if merge_with is not None:
            if merge_with['TransactionID'].duplicated().any():
                raise ValueError("Original data contains duplicate TransactionID!")

            result = merge_with.merge(
                cached_data,
                on='TransactionID',
                how='left'
            )
            print(f'  Merged cache with original data: {result.shape}')
            return result

        return cached_data

    def save_cache(self, data, windows, data_type='train', feature_names=None):
        """
        儲存快取特徵

        Args:
            data: 完整的 DataFrame
            windows: 時間窗口列表
            data_type: 資料類型
            feature_names: 如果提供，只儲存這些特徵欄位（+ TransactionID）
        """
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)

        # 如果提供 feature_names，只儲存這些欄位
        if feature_names:
            cols_to_save = ['TransactionID'] + feature_names
            cache_data = data[cols_to_save].copy()
            print(f'  Saving {len(feature_names)} features (+ TransactionID)')
        else:
            cache_data = data

        # 根據是否壓縮選擇儲存方式
        if self.use_compression:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = cache_file.stat().st_size / (1024 ** 2)
        print(f'  Saved cache: {file_size:.1f} MB')

    def clear_cache(self, windows=None, data_type=None):
        """清除快取"""
        if windows is None and data_type is None:
            # 清除所有 cache（支援 .pkl 和 .pkl.gz）
            for cache_file in self.cache_dir.glob('*.pkl*'):
                cache_file.unlink()
                print(f'  Removed: {cache_file.name}')
        else:
            cache_file = self.cache_dir / self.get_cache_key(windows, data_type)
            if cache_file.exists():
                cache_file.unlink()
                print(f'  Removed: {cache_file.name}')

    def list_cache(self):
        """列出所有快取檔案"""
        print(f'Cached features:')
        cache_files = list(self.cache_dir.glob('*.pkl*'))

        if not cache_files:
            print(f'  No cached features')
            return

        total_size = 0
        for cache_file in sorted(cache_files):
            file_size = cache_file.stat().st_size / (1024 ** 2)
            total_size += file_size
            print(f'  - {cache_file.name}: {file_size:.1f} MB')

        print(f'\nTotal cache size: {total_size:.1f} MB')


if __name__ == '__main__':
    cache_mgr = FeatureCacheManager()
    cache_mgr.list_cache()

    exists = cache_mgr.cache_exists([1, 3, 7], 'train')
    print(f'\nCache exists: {exists}')
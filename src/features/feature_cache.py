import os
from os.path import exists

import pandas as pd
import pickle
from pathlib import Path
import hashlib


class FeatureCacheManager:

    def __init__(self, cache_dir='outputs/features'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, windows, data_type='train'):
        windows_str = '_'.join(map(str, sorted(windows)))
        return f'features_{data_type}_w{windows_str}.pkl'

    def cache_exists(self, windows, data_type='train'):
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)
        return cache_file.exists()

    def load_cache(self, windows, data_type='train'):
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)

        if not cache_file.exists():
            raise FileNotFoundError(f'Cache not found: {cache_file}')
        print(f'  Loading cache features from: {cache_file.name}')

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded cache features: {cache_file.name}')

        return data

    def save_cache(self, data, windows, data_type='train'):
        cache_file = self.cache_dir / self.get_cache_key(windows, data_type)

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        file_size = cache_file.stat().st_size / (1024**2)
        print(f'  Saved cache features: {file_size: .1f} MB')

    def clear_cache(self, windows=None, data_type=None):
        if windows is None and data_type is None:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
                print(f'  Removed: {cache_file.name}')
        else:
            cache_file = self.cache_dir / self.get_cache_key(windows, data_type)
            if cache_file.exists():
                cache_file.unlink()
                print(f'  Removed: {cache_file.name}')

    def list_cache(self):
        print(f'List cached features')
        cache_files = list(self.cache_dir.glob('*.pkl'))

        if not cache_files:
            print(f'  No cached features')
            return
        for cache_file in sorted(cache_files):
            file_size = cache_file.stat().st_size / (1024**2)
            print(f'  - {cache_file.name}: {file_size:.1f} MB')


if __name__ == '__main__':
    cache_mgr = FeatureCacheManager()
    cache_mgr.list_cache()

    exists = cache_mgr.cache_exists([1, 3, 7], 'train')
    print(f'Cache exists: {exists}')

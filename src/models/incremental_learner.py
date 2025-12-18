"""
Incremental Learner
增量學習器（帶漂移檢測）
"""

import lightgbm as lgb
from PIL.ImageCms import Flags
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import time
from src.models.drift_detector import DriftDetector


class IncrementalLearner:
    """增量學習器"""

    def __init__(self, base_model, feature_cols, use_drift_detection=True):
        """
        初始化

        Args:
            base_model: 基礎模型（已訓練）
            feature_cols: 特徵欄位列表
            use_drift_detection: 是否使用漂移檢測
        """
        self.model = base_model
        self.feature_cols = feature_cols
        self.use_drift_detection = use_drift_detection
        self.drift_detector = DriftDetector(delta=0.002) if use_drift_detection else None

        # 統計
        self.retrain_count = 0
        self.total_predictions = 0
        self.batch_metrics = []

    def predict_batch(self, batch_data):
        """
        預測批次

        Args:
            batch_data: 批次數據

        Returns:
            y_pred_proba, y_pred
        """
        X_batch = batch_data[self.feature_cols]

        y_pred_proba = self.model.predict(X_batch)
        y_pred = (y_pred_proba >= 0.6).astype(int)

        self.total_predictions += len(batch_data)

        return y_pred_proba, y_pred

    def update(self, batch_data, drift_accumulation_threshold=1):
        """
        更新模型（如果檢測到漂移）

        Args:
            batch_data: 批次數據
            drift_accumulation_threshold: 漂移累積閾值（檢測到N次漂移後重訓練）

        Returns:
            retrained: 是否重新訓練
        """
        if not self.use_drift_detection:
            return False

        # 預測
        y_pred_proba, y_pred = self.predict_batch(batch_data)
        y_true = batch_data['isFraud'].values

        # 計算錯誤
        errors = (y_pred != y_true).astype(int)

        # 更新漂移檢測器，記錄漂移次數
        drift_detected = False
        # drift_count_in_batch = 0
        for error in errors:
            if self.drift_detector.update(error):
                drift_detected = True
                break

        # 如果檢測到漂移，重新訓練
        if drift_detected:
            print(f'  Drift detected in batch, retraining model...')
            self._retrain(batch_data)

            self.drift_detector.reset()
            return True

        return False

    def _retrain(self, new_data):
        """
        重新訓練模型

        Args:
            new_data: 新數據
        """
        X_new = new_data[self.feature_cols]
        y_new = new_data['isFraud']

        # Undersampling
        sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_new, y_new)

        print(f'    Training samples: {len(X_resampled):,} | Fraud: {y_resampled.mean():.2%}')

        # 訓練新模型
        train_data_lgb = lgb.Dataset(X_resampled, label=y_resampled)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        start_time = time.time()
        self.model = lgb.train(
            params,
            train_data_lgb,
            num_boost_round=300,  # 較少輪數（增量更新）
            valid_sets=[train_data_lgb],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        train_time = time.time() - start_time

        self.retrain_count += 1
        print(f'    ✓ Retrained in {train_time:.2f}s (Total retrains: {self.retrain_count})')

    def get_statistics(self):
        """獲取統計資訊"""
        stats = {
            'total_predictions': self.total_predictions,
            'retrain_count': self.retrain_count,
        }

        if self.drift_detector:
            stats.update(self.drift_detector.get_statistics())

        return stats


if __name__ == '__main__':
    print('IncrementalLearner - 需要實際數據測試')
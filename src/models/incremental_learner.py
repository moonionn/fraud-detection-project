"""
Incremental Learner
增量學習器（帶漂移檢測與訓練視窗）
"""
import time
import pandas as pd
from collections import deque
from src.models.drift_detector import DriftDetector
from src.models.model import train_model


class IncrementalLearner:
    """增量學習器"""

    def __init__(self, base_model, feature_cols, model_type='lgb', sampling_type='undersample',
                 use_drift_detection=True, window_size=2):
        """
        初始化

        Args:
            base_model: 基礎模型（已訓練）
            feature_cols: 特徵欄位列表
            model_type: 模型類型 ('lgb', 'xgb', etc.)
            sampling_type: 重新訓練時使用的採樣方法
            use_drift_detection: 是否使用漂移檢測
            window_size: 訓練視窗大小（保留最近 N 個批次）
        """
        self.model = base_model
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.sampling_type = sampling_type
        self.use_drift_detection = use_drift_detection
        self.drift_detector = DriftDetector(delta=0.002) if use_drift_detection else None

        # 訓練視窗（使用 deque 自動維護固定大小）
        self.training_window = deque(maxlen=window_size)
        self.window_size = window_size

        # 統計
        self.retrain_count = 0
        self.total_predictions = 0
        self.batch_count = 0
        self.batch_metrics = []

    def predict_batch(self, batch_data):
        """
        預測批次（不包含更新邏輯）

        Args:
            batch_data: 批次數據

        Returns:
            y_pred_proba, y_pred
        """
        X_batch = batch_data[self.feature_cols]

        if self.model_type == 'lgb':
            y_pred_proba = self.model.predict(X_batch)
        else:
            y_pred_proba = self.model.predict_proba(X_batch)[:, 1]

        y_pred = (y_pred_proba >= 0.5).astype(int)
        self.total_predictions += len(batch_data)
        return y_pred_proba, y_pred

    def update(self, batch_data, y_pred):
        """
        更新模型（使用已有的預測結果）

        Args:
            batch_data: 批次資料
            y_pred: 預測結果（來自 predict_batch）

        Returns:
            retrained: 是否重新訓練
        """
        self.batch_count += 1

        # 將當前批次加入訓練視窗
        self.training_window.append(batch_data.copy())

        if not self.use_drift_detection:
            return False

        # 計算批次錯誤率
        y_true = batch_data['isFraud'].values
        batch_error_rate = (y_pred != y_true).mean()

        # 用批次錯誤率更新 ADWIN
        drift_detected = self.drift_detector.update(batch_error_rate)

        if drift_detected:
            print(f'  [Batch {self.batch_count}] Drift detected! Batch error rate: {batch_error_rate:.4f}')
            self._retrain()
            self.drift_detector.reset()
            return True

        return False

    def _retrain(self):
        """
        使用訓練視窗內的資料重新訓練模型
        """
        # 合併訓練視窗內的所有批次
        training_data = pd.concat(list(self.training_window), ignore_index=True)

        print(f'    Retraining with {len(self.training_window)} batches ({len(training_data)} samples)...')
        start_time = time.time()

        new_model, _ = train_model(
            train_data=training_data,
            model_type=self.model_type,
            use_sampling=True,
            sampling_type=self.sampling_type,
            num_boost_round=300,
            n_estimators=300
        )
        self.model = new_model

        train_time = time.time() - start_time
        self.retrain_count += 1
        print(f'    Retrained in {train_time:.2f}s (Total retrains: {self.retrain_count})')

    def get_statistics(self):
        """獲取統計資訊"""
        stats = {
            'total_predictions': self.total_predictions,
            'total_batches': self.batch_count,
            'retrain_count': self.retrain_count,
            'window_size': self.window_size,
            'current_window_batches': len(self.training_window)
        }
        if self.drift_detector:
            stats.update(self.drift_detector.get_statistics())
        return stats


if __name__ == '__main__':
    print('IncrementalLearner - 需要實際數據測試')
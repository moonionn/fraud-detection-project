"""
Incremental Learner
增量學習器（帶漂移檢測）
"""
import time
from src.models.drift_detector import DriftDetector
# 引用通用的模型訓練函數
from src.models.model import train_model


class IncrementalLearner:
    """增量學習器"""

    def __init__(self, base_model, feature_cols, model_type='lgb', sampling_type='undersample', use_drift_detection=True):
        """
        初始化

        Args:
            base_model: 基礎模型（已訓練）
            feature_cols: 特徵欄位列表
            model_type: 模型類型 ('lgb', 'xgb', etc.)
            sampling_type: 重新訓練時使用的採樣方法
            use_drift_detection: 是否使用漂移檢測
        """
        self.model = base_model
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.sampling_type = sampling_type
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

        # 根據模型類型使用不同的預測方法
        if self.model_type == 'lgb':
            y_pred_proba = self.model.predict(X_batch)
        else:
            y_pred_proba = self.model.predict_proba(X_batch)[:, 1]

        y_pred = (y_pred_proba >= 0.5).astype(int)
        self.total_predictions += len(batch_data)
        return y_pred_proba, y_pred

    def update(self, batch_data):
        """
        更新模型（如果檢測到漂移）

        Args:
            batch_data: 批次數據

        Returns:
            retrained: 是否重新訓練
        """
        if not self.use_drift_detection:
            return False

        y_pred_proba, y_pred = self.predict_batch(batch_data)
        y_true = batch_data['isFraud'].values
        errors = (y_pred != y_true).astype(int)

        drift_detected = False
        for error in errors:
            if self.drift_detector.update(error):
                drift_detected = True
                break

        if drift_detected:
            print(f'  Drift detected in batch, retraining model...')
            self._retrain(batch_data)
            self.drift_detector.reset()
            return True

        return False

    def _retrain(self, new_data):
        """
        使用通用的 train_model 函數重新訓練模型

        Args:
            new_data: 新數據
        """
        print(f'    Retraining with model type: {self.model_type.upper()}')
        start_time = time.time()

        # 使用通用的 train_model 函數，它會處理採樣和模型訓練
        new_model, _ = train_model(
            train_data=new_data,
            model_type=self.model_type,
            use_sampling=True,
            sampling_type=self.sampling_type,
            num_boost_round=300,  # for lgb
            n_estimators=300      # for xgb/rf
        )
        self.model = new_model

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
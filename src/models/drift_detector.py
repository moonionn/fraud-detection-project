"""
Drift Detector using ADWIN
概念漂移檢測器
"""

import numpy as np
from river.drift import ADWIN


class DriftDetector:
    """漂移檢測器（基於 ADWIN）"""

    def __init__(self, delta=0.002):
        """
        初始化

        Args:
            delta: ADWIN 靈敏度參數（越小越敏感）
        """
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.drift_count = 0
        self.warning_count = 0

    def update(self, error):
        """
        更新漂移檢測器

        Args:
            error: 預測錯誤（0 or 1）

        Returns:
            drift_detected: 是否檢測到漂移
        """
        self.adwin.update(error)

        # 檢測漂移
        drift_detected = self.adwin.drift_detected

        if drift_detected:
            self.drift_count += 1
            print(f'  Concept Drift Detected! (Total: {self.drift_count})')
            # self.adwin = ADWIN(delta=self.delta)  # 重置檢測器
            return True

        return False

    def reset(self):
        self.adwin = ADWIN(delta=self.delta)
        print(f'  Drift Detector Reset!')

    def get_statistics(self):
        """獲取統計資訊"""
        return {
            'drift_count': self.drift_count,
            'warning_count': self.warning_count,
            'current_mean': self.adwin.estimation
        }


if __name__ == '__main__':
    # 測試漂移檢測
    detector = DriftDetector(delta=0.002)

    # 模擬穩定期
    print('Stable period:')
    for i in range(100):
        error = np.random.binomial(1, 0.1)  # 10% 錯誤率
        drift = detector.update(error)

    # 模擬漂移
    print('\nDrift period:')
    for i in range(100):
        error = np.random.binomial(1, 0.5)  # 50% 錯誤率
        drift = detector.update(error)

    stats = detector.get_statistics()
    print(f'\nStatistics: {stats}')
"""
Threshold Optimizer
動態閾值優化器（最大化 F1 分數）
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class ThresholdOptimizer:
    """閾值優化器"""

    @staticmethod
    def find_best_threshold(y_true, y_pred_proba, metric='f1', search_range=(0.1, 0.9), step=0.01):
        """
        尋找最佳閾值

        Args:
            y_true: 真實標籤
            y_pred_proba: 預測機率
            metric: 優化目標（'f1', 'precision', 'recall'）
            search_range: 搜尋範圍
            step: 搜尋步長

        Returns:
            best_threshold, best_score, all_results
        """
        thresholds = np.arange(search_range[0], search_range[1], step)
        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # 避免全為0或全為1的情況
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue

            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })

        # 根據指標排序
        if metric == 'f1':
            best = max(results, key=lambda x: x['f1'])
        elif metric == 'precision':
            best = max(results, key=lambda x: x['precision'])
        elif metric == 'recall':
            best = max(results, key=lambda x: x['recall'])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best['threshold'], best[metric], results

    @staticmethod
    def evaluate_with_threshold(y_true, y_pred_proba, threshold):
        """
        使用指定閾值評估

        Args:
            y_true: 真實標籤
            y_pred_proba: 預測機率
            threshold: 閾值

        Returns:
            metrics: 評估指標
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'threshold': threshold,
            'AUC': roc_auc_score(y_true, y_pred_proba),
            'F1': f1_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
        }

        return metrics

    @staticmethod
    def print_threshold_analysis(results):
        """
        打印閾值分析

        Args:
            results: 閾值搜尋結果
        """
        print('\n📊 Threshold Analysis (Top 10):')
        print('=' * 70)
        print(f"{'Threshold':>10} | {'F1':>8} | {'Precision':>10} | {'Recall':>8}")
        print('-' * 70)

        # 按 F1 排序
        sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:10]

        for r in sorted_results:
            print(f"{r['threshold']:>10.3f} | {r['f1']:>8.4f} | "
                  f"{r['precision']:>10.4f} | {r['recall']:>8.4f}")
        print('=' * 70)


if __name__ == '__main__':
    # 測試
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # 生成測試數據
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 訓練模型
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 尋找最佳閾值
    optimizer = ThresholdOptimizer()
    best_threshold, best_f1, results = optimizer.find_best_threshold(y_test, y_pred_proba)

    print(f'Best Threshold: {best_threshold:.3f}')
    print(f'Best F1: {best_f1:.4f}')

    # 打印分析
    optimizer.print_threshold_analysis(results)

    # 對比評估
    print('\n📊 Comparison:')
    metrics_05 = optimizer.evaluate_with_threshold(y_test, y_pred_proba, 0.5)
    metrics_best = optimizer.evaluate_with_threshold(y_test, y_pred_proba, best_threshold)

    print(
        f'\nThreshold 0.5: F1={metrics_05["F1"]:.4f}, Prec={metrics_05["Precision"]:.4f}, Rec={metrics_05["Recall"]:.4f}')
    print(
        f'Threshold {best_threshold:.3f}: F1={metrics_best["F1"]:.4f}, Prec={metrics_best["Precision"]:.4f}, Rec={metrics_best["Recall"]:.4f}')
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score


class ModelEvaluator:

    @staticmethod
    def evaluate(y_true, y_pred_proba, y_pred=None) -> dict:
        if y_pred is None:
            y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            'AUC': roc_auc_score(y_true, y_pred_proba),
            'AUC_PR': average_precision_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: dict):
        print('|__Evaluation Metrics__|')
        print(f'  AUC:       {metrics["AUC"]:.4%}')
        print(f'  AUC-PR:    {metrics["AUC_PR"]:.4f}')
        print(f'  F1 Score:  {metrics["F1"]:.4f}')
        print(f'  Precision: {metrics["Precision"]:.4f}')
        print(f'  Recall:    {metrics["Recall"]:.4f}')

if __name__ == '__main__':
    # Example usage
    import numpy as np

    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    y_pred = (y_pred_proba >= 0.5).astype(int)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred_proba, y_pred)
    evaluator.print_metrics(metrics)
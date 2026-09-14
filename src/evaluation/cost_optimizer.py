import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss

class CostMatrixOptimizer:
    """
    Business Cost-Sensitive Threshold Optimizer for Predictive Maintenance.
    Minimizes business loss:
    - False Negative (FN): Missed failure -> costly downtime & repair ($5,000 default)
    - False Positive (FP): False alarm -> inspection cost ($200 default)
    - True Positive (TP): Planned preventive maintenance ($200 default)
    - True Negative (TN): Normal operation ($0)
    """

    def __init__(self, fn_cost: float = 5000.0, fp_cost: float = 200.0, tp_cost: float = 200.0, tn_cost: float = 0.0):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost

    def compute_total_cost(self, y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))

        total_cost = (tp * self.tp_cost) + (fp * self.fp_cost) + (fn * self.fn_cost) + (tn * self.tn_cost)
        return total_cost, tp, fp, fn, tn

    def optimize_threshold(self, y_true: np.ndarray, y_probs: np.ndarray):
        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = 0.5
        min_cost = float('inf')
        best_confusion = (0, 0, 0, 0)

        cost_curve = []

        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)
            cost, tp, fp, fn, tn = self.compute_total_cost(y_true, y_pred)
            cost_curve.append({'threshold': th, 'cost': cost, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})

            if cost < min_cost:
                min_cost = cost
                best_threshold = th
                best_confusion = (tp, fp, fn, tn)

        df_cost = pd.DataFrame(cost_curve)
        
        # Calculate cost at standard 0.5 threshold
        cost_default, _, _, _, _ = self.compute_total_cost(y_true, (y_probs >= 0.5).astype(int))
        cost_savings = cost_default - min_cost

        return {
            'best_threshold': best_threshold,
            'min_cost': min_cost,
            'cost_at_0.5': cost_default,
            'cost_savings': cost_savings,
            'confusion_matrix': best_confusion,
            'cost_curve_df': df_cost
        }

    def evaluate_model_comprehensive(self, y_true: np.ndarray, y_probs: np.ndarray, model_name: str = "Model") -> dict:
        pr_auc = average_precision_score(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        brier = brier_score_loss(y_true, y_probs)

        cost_opt = self.optimize_threshold(y_true, y_probs)
        opt_th = cost_opt['best_threshold']

        y_pred_opt = (y_probs >= opt_th).astype(int)
        precision = precision_score(y_true, y_pred_opt, zero_division=0)
        recall = recall_score(y_true, y_pred_opt, zero_division=0)
        f1 = f1_score(y_true, y_pred_opt, zero_division=0)

        return {
            'Model': model_name,
            'PR-AUC': round(pr_auc, 4),
            'ROC-AUC': round(roc_auc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'Brier Score': round(brier, 4),
            'Optimal Threshold': round(opt_th, 4),
            'Total Loss ($)': round(cost_opt['min_cost'], 2),
            'Loss at 0.5 ($)': round(cost_opt['cost_at_0.5'], 2),
            'Cost Savings ($)': round(cost_opt['cost_savings'], 2),
            'TP': cost_opt['confusion_matrix'][0],
            'FP': cost_opt['confusion_matrix'][1],
            'FN': cost_opt['confusion_matrix'][2],
            'TN': cost_opt['confusion_matrix'][3]
        }

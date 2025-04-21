import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Tuple, Dict

def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    medium_thresh: float = 0.3,
    high_thresh:   float = 0.7
) -> Dict[str, float]:
    """
    Print AUC-ROC, AUC-PR, and per-risk-category precision/recall.
    Returns a dict of the main metrics.
    """
    roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr   = auc(recall, precision)

    print(f"AUC-ROC: {roc:.4f}")
    print(f"AUC-PR : {pr:.4f}")

    risk = pd.Series(['Low'] * len(y_pred), index=y_true.index)
    risk[y_pred >= medium_thresh] = 'Medium'
    risk[y_pred >= high_thresh]   = 'High'

    for cat in ['High','Medium','Low']:
        mask = (risk == cat)
        if mask.any():
            p = y_true.loc[mask].sum() / mask.sum()
            r = y_true.loc[mask].sum() / y_true.sum()
            print(f"{cat:6} | Count={mask.sum():4d} | Precision={p:.3f} | Recall={r:.3f}")

    return {'auc_roc': roc, 'auc_pr': pr}

import xgboost as xgb
from typing import Optional, Dict, Tuple
import pandas as pd

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series]  = None,
    params: Optional[Dict]       = None,
    early_stopping_rounds: int    = 10
) -> Tuple[xgb.XGBClassifier, pd.Series]:
    """
    Train an XGBoost classifier. If validation data is provided,
    uses early stopping.
    Returns: model and series of predicted churn probabilities on X_val.
    """
    default_params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'n_estimators':     100,
        'max_depth':        5,
        'learning_rate':    0.1,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'scale_pos_weight': float(sum(y_train==0)) / sum(y_train==1),
        'random_state':     42
    }
    if params:
        default_params.update(params)
    model = xgb.XGBClassifier(**default_params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = None

    return model, preds

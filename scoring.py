import pandas as pd
import shap
import numpy as np
from xgboost import XGBClassifier

def explain_and_score(
    model: XGBClassifier,
    X: pd.DataFrame,
    medium_thresh: float = 0.3,
    high_thresh:   float = 0.7
) -> pd.DataFrame:
    """
    Compute churn_prob, risk_category, and main SHAP driver for each row.
    """
    churn_prob = model.predict_proba(X)[:,1]
    risk = pd.Series(['Low'] * len(churn_prob))
    risk[churn_prob >= medium_thresh] = 'Medium'
    risk[churn_prob >= high_thresh]   = 'High'

    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    idx_feat   = np.argmin(shap_vals, axis=1)
    key_feat   = X.columns[idx_feat]

    return pd.DataFrame({
        'churn_prob': churn_prob,
        'risk_cat'  : risk,
        'key_feat'  : key_feat
    }, index=X.index)

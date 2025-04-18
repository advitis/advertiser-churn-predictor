# run_pipeline.py

import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

# 1) Load & split
data = pd.read_csv('../data/synthetic_ad_churn.csv')
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler().fit(X_train[numeric_cols])
X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 2) Train
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    scale_pos_weight=(sum(y_train==0)/sum(y_train==1)),
    random_state=42
)
# Train the model with default settings for compatibility
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:,1]

# 3) Evaluate
roc = roc_auc_score(y_test, y_pred_proba)
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
pr = auc(rec, prec)
print(f"AUC-ROC: {roc:.4f}")
print(f"AUC-PR : {pr:.4f}")

# 4) Risk & reasons
threshold_medium = 0.3
threshold_high = 0.7
risk = pd.Series(['Low']*len(y_pred_proba))
risk[y_pred_proba>=threshold_medium] = 'Medium'
risk[y_pred_proba>=threshold_high]   = 'High'

explainer = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test)

# Select the most negative SHAP feature per row
idx_feat = np.argmin(shap_vals, axis=1)
feat_names = X_test.columns[idx_feat]

results = pd.DataFrame({
    'churn_prob': y_pred_proba,
    'risk_cat' : risk,
    'key_feat' : feat_names
})

print(results.head())

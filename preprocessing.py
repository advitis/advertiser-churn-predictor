import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(
    path='data/synthetic_ad_churn.csv',
    test_size: float = 0.25,
    random_state: int = 42
):
    """
    Load CSV, split into train/test, and scale numeric features.
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler().fit(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, y_train, y_test

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def load_model(path: str):
    return joblib.load(path)

def evaluate(model_path: str, csv_path: str, label_col: str = "species", feature_cols=None):
    df = load_data(csv_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols]
    y = df[label_col]
    model = load_model(model_path)
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# ------- Config from env --------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-ga5")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")

# ------- Data path (re-use your GA data) --------
DATA_PATHS = ["data/iris.csv", "GA2_DVC/iris.csv", "GA4_CI/src/iris.csv", "/home/jupyter/data/iris.csv"]

def find_data():
    for p in DATA_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Could not find iris.csv in {DATA_PATHS}")

def load_iris(path):
    df = pd.read_csv(path)
    # Expected columns: sepal_length,sepal_width,petal_length,petal_width,species (or similar)
    # Normalize column names if needed
    cols = {c.lower().strip(): c for c in df.columns}
    # Try to map known features
    fx = []
    for name in ["sepal_length","sepal_width","petal_length","petal_width"]:
        cand = cols.get(name, None)
        if cand is None:
            raise ValueError(f"Missing column: {name} in {path}")
        fx.append(cand)
    # Try label
    label_col = cols.get("species", None)
    if label_col is None:
        raise ValueError("Missing 'species' column")

    X = df[fx].copy()
    y = df[label_col].astype("category").cat.codes.values  # encode into 0/1/2
    return X, y, fx

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = find_data()
    X, y, feature_names = load_iris(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Simple search space
    grid = [
        {"clf__C": 0.1, "clf__penalty": "l2", "clf__solver": "lbfgs"},
        {"clf__C": 1.0, "clf__penalty": "l2", "clf__solver": "lbfgs"},
        {"clf__C": 2.0, "clf__penalty": "l2", "clf__solver": "lbfgs"},
    ]

    best = {"acc": -1.0, "run_id": None, "params": None}

    for params in grid:
        with mlflow.start_run(nested=False) as run:
            run_id = run.info.run_id

            # Pipeline
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
            ])
            # Set params
            pipe.set_params(**params)

            # Train
            pipe.fit(X_tr, y_tr)

            # Eval
            y_pred = pipe.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="weighted")

            # Log params/metrics/artifacts
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_text("\n".join(feature_names), "features.txt")

            # Log model (as MLflow model)
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name=MODEL_NAME  # this will register/ version the model
            )

            if acc > best["acc"]:
                best.update({"acc": acc, "run_id": run_id, "params": params})

    print("[BEST] Accuracy:", best["acc"])
    print("[BEST] Run ID:", best["run_id"])
    print("[BEST] Params:", best["params"])
    # Save a small manifest for local debugging
    Path("models").mkdir(exist_ok=True, parents=True)
    with open("models/best_run.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()


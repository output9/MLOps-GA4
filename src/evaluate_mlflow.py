import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-ga5")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")

DATA_PATHS = [
    "data/iris.csv",
    "iris_feast/feature_repo/data/iris.csv",
]

def find_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find iris.csv in {DATA_PATHS}")

def load_iris(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    fx = [cols["sepal_length"], cols["sepal_width"], cols["petal_length"], cols["petal_width"]]
    y = df[cols["species"]].astype("category").cat.codes.values
    X = df[fx].copy()
    return X, y

def pick_best_version(client: MlflowClient):
    # Pick the latest "Production" first; if none, take the highest version
    mv_list = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not mv_list:
        raise RuntimeError("No versions available in the MLflow Model Registry")
    prods = [mv for mv in mv_list if mv.current_stage == "Production"]
    if prods:
        # if multiple, pick the newest version number
        best = sorted(prods, key=lambda mv: int(mv.version))[-1]
    else:
        best = sorted(mv_list, key=lambda mv: int(mv.version))[-1]
    return best.version

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    version = pick_best_version(client)
    model_uri = f"models:/{MODEL_NAME}/{version}"
    print(f"[INFO] Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    X, y = load_iris(find_data())
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    print(f"[EVAL] accuracy={acc:.4f}, f1_weighted={f1:.4f}")

if __name__ == "__main__":
    main()

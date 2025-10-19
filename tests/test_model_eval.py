
from pathlib import Path
from src.evaluate import evaluate

MODEL_PATH = Path("models/model.joblib")  # DVC-tracked pointer
DATA_PATH = Path("data/iris.csv")

def test_model_file_present():
    assert MODEL_PATH.exists(), "model.joblib not present (DVC pull should fetch it)"

def test_eval_accuracy_threshold():
    # Simple sanity threshold; adjust if your model is different
    acc = evaluate(str(MODEL_PATH), str(DATA_PATH), label_col="species")
    assert acc >= 0.85, f"Accuracy too low: {acc}"

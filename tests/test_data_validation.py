
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/iris.csv")  # DVC-tracked pointer

def test_file_present():
    assert DATA_PATH.exists(), "iris.csv not present (DVC pull should fetch it)"

def test_schema_and_nulls():
    df = pd.read_csv(DATA_PATH)
    expected_cols = {"sepal_length","sepal_width","petal_length","petal_width","species"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"
    assert df[expected_cols].isnull().sum().sum() == 0, "Nulls found in required columns"

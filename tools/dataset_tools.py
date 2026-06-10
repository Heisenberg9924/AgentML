import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from typing import Any


def load_dataset(path: str) -> pd.DataFrame:
    """Loading a dataset from a CSV file"""

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset from {path}: {e}")
        
def dataset_summary(df: pd.DataFrame) -> dict:
    """ Generating a summary of the dataset"""

    numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    summary = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb" : round(
            df.memory_usage(deep=True).sum() / (1024 * 1024),
            2
        )
    }

    return summary

def missing_value_report(df: pd.DataFrame) -> dict:
    """ Generating a missing value summary of the dataset"""

    missing_values = df.isnull().sum()

    missing_percentages = (missing_values / len(df)) * 100

    report = {}

    for col in df.columns:
        if missing_values[col] > 0:
            report[col] = {
                "missing_count": int(missing_values[col]),
                "missing_percentage": round(missing_percentages[col], 2)
            }
    return report

from pandas.api.types import (
    is_numeric_dtype
)


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str
) -> str:
    """
    Infer whether the task is
    classification or regression.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"not found in dataset."
        )

    target = df[target_column]

    # ---------------------------------
    # Non-numeric target
    # ---------------------------------

    if not is_numeric_dtype(target):
        return "classification"

    unique_values = target.nunique()
    total_rows = len(target)

    unique_ratio = (
        unique_values / total_rows
    )

    # ---------------------------------
    # Numeric target but behaves like
    # a class label
    # ---------------------------------

    if (
        unique_values <= 20
        and unique_ratio < 0.05
    ):
        return "classification"

    # ---------------------------------
    # Otherwise assume regression
    # ---------------------------------

    return "regression"
    
def split_features_target(df: pd.DataFrame, target_column: str):
    """Splitting the dataset into features and target"""

    X = df.drop(columns = [target_column])
    y = df[target_column]

    return X, y

def train_test_split_dataset(
    X,
    y,
    test_size=0.2,
    random_state: int = 42,
    stratify=None
):
    """Splitting the dataset into training and testing sets"""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test
'''
if __name__ == "__main__":
    # Example usage using the Iris Dataset

    iris : Any =  load_iris()
    df = pd.DataFrame(
        data = iris["data"],
        columns = iris["feature_names"],
    )
    df["target"]  = iris["target"]
    print("Dataset summary:", dataset_summary(df))
    print("Problem type:", detect_problem_type(df, "target"))
    '''



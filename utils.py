# utils.py
import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset and return a pandas DataFrame.
    Automatically strips whitespace from column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values or inconsistent data.
    - Converts numeric columns where possible.
    - Keeps NaN for missing numeric data (handled later in model/training).
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # If conversion fails, we leave the column as-is
            pass
    return df


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return list of columns with numeric data types.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for numeric columns (like pandas describe()).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    desc = numeric_df.describe().T  # transpose for readability
    return desc


def safe_mean(series: pd.Series) -> float:
    """
    Compute mean safely (ignoring NaNs).
    """
    return series.dropna().mean()


def safe_std(series: pd.Series) -> float:
    """
    Compute standard deviation safely (ignoring NaNs).
    """
    return series.dropna().std()


def check_missing_values(df: pd.DataFrame):
    """
    Print the number of missing values per column.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\n⚠️ Missing values detected:\n")
        print(missing)
    else:
        print("\n✅ No missing values found.\n")

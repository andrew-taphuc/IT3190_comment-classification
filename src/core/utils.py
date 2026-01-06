"""
Module chứa các utility functions.
"""
import os
import pandas as pd
from typing import Tuple, Optional


def ensure_dir(path: str):
    """Đảm bảo thư mục tồn tại."""
    os.makedirs(path, exist_ok=True)


def load_data_split(
    path: str,
    text_col: str = "text",
    label_col: str = "label"
) -> Tuple[pd.Series, pd.Series]:
    """
    Load data split từ CSV file.
    
    Args:
        path: Đường dẫn đến file CSV
        text_col: Tên cột chứa text
        label_col: Tên cột chứa label
    
    Returns:
        Tuple (X, y) với X là Series text và y là Series labels
    """
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"{path} phải có cột '{text_col}' và '{label_col}'. "
            f"Các cột hiện có: {list(df.columns)}"
        )
    
    df = df.dropna(subset=[text_col, label_col]).copy()
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)
    return X, y


def get_label_distribution(y: pd.Series) -> dict:
    """
    Tính phân bố labels.
    
    Returns:
        Dict với keys là labels và values là counts và percentages
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    return {
        label: {
            "count": int(counts[label]),
            "percentage": float(percentages[label])
        }
        for label in counts.index
    }


def print_label_distribution(y: pd.Series, name: str = ""):
    """In phân bố labels."""
    dist = get_label_distribution(y)
    print(f"\n📊 Label Distribution {name}:")
    for label, stats in dist.items():
        print(f"  {label:15s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")


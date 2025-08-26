from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def qlike(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    """QLIKE on variances."""
    true_var = np.square(y_true)
    pred_var = np.square(y_pred)
    return float(np.mean(np.log(pred_var + eps) + true_var / (pred_var + eps)))


def summarize_metrics(
    df: pd.DataFrame, truth_col: str, pred_cols: Dict[str, str]
) -> pd.DataFrame:
    """Summarize metrics for multiple prediction columns."""
    records = []
    for name, col in pred_cols.items():
        mask = df[col].notna()
        if mask.any():
            y_true = df.loc[mask, truth_col]
            y_pred = df.loc[mask, col]
            records.append(
                {"model": name, "MSE": mse(y_true, y_pred), "QLIKE": qlike(y_true, y_pred)}
            )
    return pd.DataFrame.from_records(records)

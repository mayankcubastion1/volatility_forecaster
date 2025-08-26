from __future__ import annotations

from typing import List
import pandas as pd
import pyreadr

REQUIRED_COLS: List[str] = [
    "stock_id",
    "time_id",
    "seconds_in_bucket",
    "WAP",
    "BidAskSpread",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
]


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[REQUIRED_COLS].copy()


def _coerce_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df["stock_id"] = df["stock_id"].astype(int)
    df["time_id"] = df["time_id"].astype(int)
    df["seconds_in_bucket"] = df["seconds_in_bucket"].astype(int)
    numeric_cols = [
        "WAP",
        "BidAskSpread",
        "bid_size1",
        "ask_size1",
        "bid_size2",
        "ask_size2",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values(["stock_id", "time_id", "seconds_in_bucket"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    df = pd.read_csv(path)
    df = _validate_columns(df)
    df = _coerce_and_sort(df)
    return df


def try_load_rdata(path: str) -> pd.DataFrame:
    """Load and validate RData using pyreadr."""
    result = pyreadr.read_r(path)
    if len(result) == 0:
        raise ValueError("No objects found in RData file.")
    df = None
    for obj in result.values():
        if isinstance(obj, pd.DataFrame):
            df = obj
            break
    if df is None:
        raise ValueError("No DataFrame object in RData file.")
    df = _validate_columns(df)
    df = _coerce_and_sort(df)
    return df

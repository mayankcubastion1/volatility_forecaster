from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex seconds to 1..600, forward-fill, and compute log returns."""
    numeric_cols = [
        "WAP",
        "BidAskSpread",
        "bid_size1",
        "ask_size1",
        "bid_size2",
        "ask_size2",
    ]
    group_cols = ["stock_id", "time_id"]

    def _process(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("seconds_in_bucket").reindex(range(1, 601))
        group[group_cols[0]] = group[group_cols[0]].iloc[0]
        group[group_cols[1]] = group[group_cols[1]].iloc[0]
        group[numeric_cols] = group[numeric_cols].ffill().bfill()
        group["log_return"] = np.log(group["WAP"]).diff().fillna(0.0)
        group.reset_index(inplace=True)
        group.rename(columns={"index": "seconds_in_bucket"}, inplace=True)
        return group

    return df.groupby(group_cols, group_keys=False).apply(_process)


def bucketize_30s(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 30-second buckets and compute order flow."""
    df = df.copy()
    df["time_bucket"] = ((df["seconds_in_bucket"] - 1) // 30) + 1
    df["order_flow"] = (
        df["bid_size1"] + df["ask_size1"] + df["bid_size2"] + df["ask_size2"]
    )
    return df


def aggregate_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seconds to 30-second buckets."""
    agg = (
        df.groupby(["stock_id", "time_id", "time_bucket"])
        .agg(
            realized_vol=("log_return", lambda x: np.sqrt(np.sum(np.square(x)))),
            price=("WAP", "mean"),
            BidAskSpread=("BidAskSpread", "mean"),
            order=("order_flow", "mean"),
        )
        .reset_index()
    )
    return agg


def add_lags_and_rolls(bucket_df: pd.DataFrame, include_bas: bool = True) -> pd.DataFrame:
    """Add lagged and rolling features for HAV model."""
    df = bucket_df.copy()
    group_cols = ["stock_id", "time_id"]

    df["previous_volatility"] = df.groupby(group_cols)["realized_vol"].shift(1)
    df["mean_prev_5_vol"] = (
        df.groupby(group_cols)["realized_vol"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    if include_bas:
        df["previous_BidAskSpread"] = df.groupby(group_cols)["BidAskSpread"].shift(1)
        df["mean_prev_5_bas"] = (
            df.groupby(group_cols)["BidAskSpread"]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return df

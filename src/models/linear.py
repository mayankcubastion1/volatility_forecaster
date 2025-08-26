from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class LinearConfig:
    use_order_mean: bool = True
    use_bas_mean: bool = True


class LinearVolModel:
    """Linear regression on exogenous features."""

    def __init__(self, config: LinearConfig):
        self.config = config
        self.reg = LinearRegression()
        self.feature_cols = self._feature_cols()

    def _feature_cols(self) -> list:
        cols = ["price"]
        if self.config.use_order_mean:
            cols.append("order")
        if self.config.use_bas_mean:
            cols.append("BidAskSpread")
        return cols

    def fit(self, df: pd.DataFrame) -> "LinearVolModel":
        train_df = df.dropna(subset=self.feature_cols + ["realized_vol"])
        if train_df.empty:
            raise ValueError("Not enough data to fit Linear model.")
        self.reg.fit(train_df[self.feature_cols], train_df["realized_vol"])
        return self

    def predict_insample(self, df: pd.DataFrame) -> pd.Series:
        mask = df[self.feature_cols].notna().all(axis=1)
        preds = pd.Series(index=df.index, dtype=float)
        if mask.any():
            preds.loc[mask] = self.reg.predict(df.loc[mask, self.feature_cols])
        return preds

    def forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        last_row = df.iloc[-1][self.feature_cols].values.reshape(1, -1)
        pred = self.reg.predict(last_row)[0]
        return np.repeat(pred, horizon)

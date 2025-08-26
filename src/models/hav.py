from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class HAVConfig:
    use_bas: bool = False


class HAVModel:
    """HAR-style linear regression model."""

    def __init__(self, config: HAVConfig):
        self.config = config
        self.reg = LinearRegression()

    def _feature_cols(self) -> list:
        cols = ["previous_volatility", "mean_prev_5_vol"]
        if self.config.use_bas:
            cols.extend(["previous_BidAskSpread", "mean_prev_5_bas"])
        return cols

    def fit(self, df: pd.DataFrame) -> "HAVModel":
        cols = self._feature_cols()
        train_df = df.dropna(subset=cols + ["realized_vol"])
        if train_df.empty:
            raise ValueError("Not enough data to fit HAV model.")
        self.reg.fit(train_df[cols], train_df["realized_vol"])
        return self

    def predict_insample(self, df: pd.DataFrame) -> pd.Series:
        cols = self._feature_cols()
        mask = df[cols].notna().all(axis=1)
        preds = pd.Series(index=df.index, dtype=float)
        if mask.any():
            preds.loc[mask] = self.reg.predict(df.loc[mask, cols])
        return preds

    def forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        cols = self._feature_cols()
        vols = df["realized_vol"].tolist()
        bas_value = df["BidAskSpread"].iloc[-1] if self.config.use_bas else None
        preds = []
        for _ in range(horizon):
            prev = vols[-1]
            mean5 = np.mean(vols[-5:])
            features = [prev, mean5]
            if self.config.use_bas:
                features.extend([bas_value, bas_value])
            pred = self.reg.predict([features])[0]
            preds.append(pred)
            vols.append(pred)
        return np.array(preds)

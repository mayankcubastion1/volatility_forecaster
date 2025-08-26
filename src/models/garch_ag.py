from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class AGConfig:
    ar_order: int
    garch_p: int
    garch_q: int


class AR_GARCH_VolModel:
    """AR-GARCH model on log-volatility."""

    def __init__(self, config: AGConfig, epsilon: float = 1e-8):
        self.config = config
        self.epsilon = epsilon
        self.model = None
        self.result = None

    def fit(self, vol_series: pd.Series) -> "AR_GARCH_VolModel":
        y = np.log(vol_series + self.epsilon)
        am = arch_model(
            y,
            mean="AR",
            lags=self.config.ar_order,
            vol="GARCH",
            p=self.config.garch_p,
            q=self.config.garch_q,
            rescale=False,
        )
        self.result = am.fit(disp="off")
        self.model = am
        return self

    def predict_insample(self) -> pd.Series:
        if self.result is None:
            raise ValueError("Model is not fitted.")
        forecast = self.result.forecast(horizon=1, reindex=True)
        nobs = self.result.model.nobs
        mean_series = forecast.mean["h.1"].iloc[:nobs]
        return np.exp(mean_series)

    def forecast(self, horizon: int) -> np.ndarray:
        if self.result is None:
            raise ValueError("Model is not fitted.")
        forecast = self.result.forecast(horizon=horizon, reindex=True)
        mean_forecast = forecast.mean.iloc[-1].values
        return np.exp(mean_forecast)

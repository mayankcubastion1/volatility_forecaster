from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_insample(
    df: pd.DataFrame,
    truth_col: str,
    ag_col: str,
    hav_col: str,
    lr_col: str,
    title: str,
) -> go.Figure:
    """Overlay in-sample predictions with truth."""
    fig = go.Figure()
    x = df["time_bucket"]
    fig.add_trace(go.Scatter(x=x, y=df[truth_col], name="True", mode="lines+markers"))
    if ag_col in df:
        fig.add_trace(go.Scatter(x=x, y=df[ag_col], name="AG", mode="lines"))
    if hav_col in df:
        fig.add_trace(go.Scatter(x=x, y=df[hav_col], name="HAV", mode="lines"))
    if lr_col in df:
        fig.add_trace(go.Scatter(x=x, y=df[lr_col], name="Linear", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Time Bucket", yaxis_title="Volatility")
    return fig


def plot_future(
    horizon: int, ag: np.ndarray, hav: np.ndarray, lr: np.ndarray, title: str
) -> go.Figure:
    """Plot future forecasts."""
    x = list(range(1, horizon + 1))
    fig = go.Figure()
    if ag is not None:
        fig.add_trace(go.Scatter(x=x, y=ag, name="AG", mode="lines+markers"))
    if hav is not None:
        fig.add_trace(go.Scatter(x=x, y=hav, name="HAV", mode="lines+markers"))
    if lr is not None:
        fig.add_trace(go.Scatter(x=x, y=lr, name="Linear", mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="Horizon", yaxis_title="Volatility")
    return fig

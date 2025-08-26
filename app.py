import streamlit as st
import pandas as pd
import numpy as np

from src.data import load_csv, try_load_rdata
from src.features import (
    compute_log_returns,
    bucketize_30s,
    aggregate_buckets,
    add_lags_and_rolls,
)
from src.models.garch_ag import AGConfig, AR_GARCH_VolModel
from src.models.hav import HAVConfig, HAVModel
from src.models.linear import LinearConfig, LinearVolModel
from src.metrics import summarize_metrics
from src.plotting import plot_insample, plot_future


st.set_page_config(page_title="Intraday Volatility Forecasting", layout="wide")


def generate_synthetic() -> pd.DataFrame:
    """Generate a small synthetic dataset for demo purposes."""
    seconds = np.arange(1, 601)
    wap = 100 + np.cumsum(np.random.normal(0, 0.1, size=600))
    bas = np.abs(np.random.normal(0, 0.01, size=600)) + 0.01
    size1 = np.random.randint(1, 10, size=600)
    size2 = np.random.randint(1, 10, size=600)
    df = pd.DataFrame(
        {
            "stock_id": 0,
            "time_id": 0,
            "seconds_in_bucket": seconds,
            "WAP": wap,
            "BidAskSpread": bas,
            "bid_size1": size1,
            "ask_size1": size1,
            "bid_size2": size2,
            "ask_size2": size2,
        }
    )
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load CSV or RData depending on file extension."""
    if path.lower().endswith(".csv"):
        return load_csv(path)
    if path.lower().endswith((".rdata", ".rda")):
        return try_load_rdata(path)
    raise ValueError("File must be .csv, .rdata, or .rda")


# Sidebar: data loading
st.sidebar.header("Data Loader")
data_path = st.sidebar.text_input("CSV or RData path")
load_button = st.sidebar.button("Load data")
demo_button = st.sidebar.button("Generate synthetic demo data")

if "data" not in st.session_state:
    st.session_state["data"] = None

if load_button:
    try:
        st.session_state["data"] = load_data(data_path)
        st.success("Data loaded successfully.")
    except Exception as exc:
        st.session_state["data"] = None
        st.error(f"Failed to load data: {exc}")

if demo_button:
    st.session_state["data"] = generate_synthetic()
    st.success("Synthetic demo data generated.")

data = st.session_state["data"]

if data is not None:
    # Sidebar: selections
    st.sidebar.header("Window Selection")
    stock_ids = sorted(data["stock_id"].unique())
    stock_id = st.sidebar.selectbox("Stock ID", stock_ids)
    time_ids = sorted(data[data["stock_id"] == stock_id]["time_id"].unique())
    time_id = st.sidebar.selectbox("Time ID", time_ids)

    # Sidebar: model controls
    st.sidebar.header("Models")
    ar_order = st.sidebar.number_input("AG AR order", min_value=0, max_value=5, value=1)
    garch_p = st.sidebar.number_input("GARCH p", min_value=1, max_value=5, value=1)
    garch_q = st.sidebar.number_input("GARCH q", min_value=1, max_value=5, value=1)

    hav_use_bas = st.sidebar.checkbox("HAV include BAS lags", value=False)
    lin_use_order = st.sidebar.checkbox("Linear include order", value=True)
    lin_use_bas = st.sidebar.checkbox("Linear include BAS", value=True)

    horizon = st.sidebar.number_input("Forecast horizon", 1, 40, 5)

    # Filter selected window
    window_df = data[(data["stock_id"] == stock_id) & (data["time_id"] == time_id)]
    if window_df.empty:
        st.error("No data for selected stock_id/time_id.")
        st.stop()

    # Feature pipeline
    try:
        df = compute_log_returns(window_df)
        df = bucketize_30s(df)
        bucket_df = aggregate_buckets(df)
        bucket_df = add_lags_and_rolls(bucket_df, include_bas=True)
    except Exception as exc:
        st.error(f"Feature processing error: {exc}")
        st.stop()

    # Fit models
    bucket_df["AG"] = np.nan
    bucket_df["HAV"] = np.nan
    bucket_df["Linear"] = np.nan

    # AG Model
    try:
        ag_model = AR_GARCH_VolModel(AGConfig(ar_order, garch_p, garch_q))
        ag_model.fit(bucket_df["realized_vol"])
        bucket_df["AG"] = ag_model.predict_insample()
        ag_forecast = ag_model.forecast(int(horizon))
    except Exception as exc:
        st.error(f"AG model failed: {exc}")
        ag_forecast = np.array([np.nan] * int(horizon))

    # HAV Model
    try:
        hav_model = HAVModel(HAVConfig(use_bas=hav_use_bas))
        hav_model.fit(bucket_df)
        bucket_df["HAV"] = hav_model.predict_insample(bucket_df)
        hav_forecast = hav_model.forecast(bucket_df, int(horizon))
    except Exception as exc:
        st.error(f"HAV model failed: {exc}")
        hav_forecast = np.array([np.nan] * int(horizon))

    # Linear Model
    try:
        lin_model = LinearVolModel(
            LinearConfig(
                use_order_mean=lin_use_order, use_bas_mean=lin_use_bas
            )
        )
        lin_model.fit(bucket_df)
        bucket_df["Linear"] = lin_model.predict_insample(bucket_df)
        lin_forecast = lin_model.forecast(bucket_df, int(horizon))
    except Exception as exc:
        st.error(f"Linear model failed: {exc}")
        lin_forecast = np.array([np.nan] * int(horizon))

    # Data preview
    st.subheader("Bucketed Data")
    st.dataframe(bucket_df)

    # In-sample plot
    st.subheader("In-sample Predictions")
    insample_fig = plot_insample(
        bucket_df,
        truth_col="realized_vol",
        ag_col="AG",
        hav_col="HAV",
        lr_col="Linear",
        title=f"Stock {stock_id}, Time {time_id}",
    )
    st.plotly_chart(insample_fig, use_container_width=True)

    # Metrics
    st.subheader("Metrics")
    metrics_df = summarize_metrics(
        bucket_df,
        truth_col="realized_vol",
        pred_cols={"AG": "AG", "HAV": "HAV", "Linear": "Linear"},
    )
    metrics_df["MSE"] = metrics_df["MSE"].round(6)
    metrics_df["QLIKE"] = metrics_df["QLIKE"].round(6)
    st.table(metrics_df)

    # Future forecast plot
    st.subheader("Future Forecast")
    future_fig = plot_future(
        int(horizon),
        ag_forecast,
        hav_forecast,
        lin_forecast,
        f"{horizon}-step Ahead Forecast",
    )
    st.plotly_chart(future_fig, use_container_width=True)

else:
    st.info(
        "Load a dataset using the sidebar. You may also generate synthetic demo data."
    )

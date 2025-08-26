# Intraday Volatility Forecasting App

Streamlit app to forecast short-horizon realized volatility in 10-minute windows from limit order book data.

## Features
- Reconstruct 600-second windows with forward-filled snapshots.
- Aggregate into 20 × 30-second buckets.
- AR+GARCH (log-vol), HAR-style, and linear models.
- Metrics: MSE and QLIKE.
- Interactive charts for in-sample and future forecasts.

## Installation

```bash
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

## Expected CSV schema
Columns (all required):

| column          | description                                   |
|-----------------|-----------------------------------------------|
| `stock_id`      | integer stock identifier                      |
| `time_id`       | integer window identifier                     |
| `seconds_in_bucket` | 1..600 seconds inside the window        |
| `WAP`           | weighted average price                        |
| `BidAskSpread`  | bid-ask spread                                |
| `bid_size1` / `ask_size1` / `bid_size2` / `ask_size2` | level sizes |

RData files should contain a DataFrame with the same columns.

## Notes
The AG model uses log-realized-volatility with an AR(p) mean and GARCH(p, q) variance via the `arch` package, serving as a practical analogue to ARMA-GARCH.

Run the app and explore different model settings and forecast horizons interactively.

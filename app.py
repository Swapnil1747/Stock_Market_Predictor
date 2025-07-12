import numpy as np
import pandas as pd
import yfinance as yf
import calendar
from datetime import timedelta, date
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ─── App Header ──────────────────────────────────────
st.header('📈 Stock Market Predictor App')

# ─── App Description & Author ──────────────────────────────
st.subheader("""
Welcome to the **Stock Market Price Predictor** app!  
This interactive tool allows you to:

- **Select multiple stocks** and compare their performance.  
- **Auto-refresh** data on demand.  
- **Visualize historical trends** with interactive charts.  
- **Download predictions** and error metrics as CSV.  

Built with caching, LSTM modeling, and rich analytics.  
""")
st.markdown("**Author:** Swapnil Mishra")

# ─── Sidebar Controls & Caching ───────────────────────────────
@st.cache_data
def get_data(tickers, start, end):
    return {t: yf.download(t, start, end)['Close'] for t in tickers}

@st.cache_data
def predict_series(series, model):
    scaler = MinMaxScaler((0,1))
    values = series.values
    scaled = scaler.fit_transform(values.reshape(-1,1))
    X = np.array([scaled[i-100:i] for i in range(100, len(scaled))])
    preds = model.predict(X) * (1/scaler.scale_[0])
    return preds.flatten(), values[100:]

# ─── Sidebar Inputs ──────────────────────────
st.sidebar.subheader("📦 Stock Selection & Controls")

popular = {
    'Google (GOOG)': 'GOOG', 'Apple (AAPL)': 'AAPL', 'Tesla (TSLA)': 'TSLA',
    'Amazon (AMZN)': 'AMZN', 'Microsoft (MSFT)': 'MSFT', 'NVIDIA (NVDA)': 'NVDA',
    'Meta (META)': 'META', 'Reliance (RELIANCE.NS)': 'RELIANCE.NS',
    'TCS (TCS.NS)': 'TCS.NS'
}
ticker_choices = st.sidebar.multiselect("Choose stocks", list(popular.keys()), default=list(popular.keys())[:2])
custom = st.sidebar.text_input("Add custom stock (optional)")
tickers = [popular[c] for c in ticker_choices]
if custom:
    tickers.append(custom.upper())

date_col1, date_col2 = st.sidebar.columns(2)
with date_col1:
    start = st.sidebar.date_input("Start", datetime.date(2012,1,1))
with date_col2:
    end = st.sidebar.date_input("End", datetime.date.today())
if start > end:
    st.sidebar.error("End date must be after start date.")
    st.stop()
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Load model (.h5 format recommended)
try:
    model = load_model('stock_model.h5', compile=False)
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# Fetch data
if not tickers:
    st.error("Select at least one stock to visualize.")
    st.stop()

data_dict = get_data(tickers, start, end)

# ─── Historical Interactive Chart ───────────────────
st.subheader("📈 Historical Closing Prices")
for tkr, series in data_dict.items():
    plt.plot(series.index, series.values, label=tkr)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()

# ─── Predictions & Backtesting Metrics ─────────────────────
st.subheader("🔮 Predictive Analysis & Backtesting")
results = []
for tkr, series in data_dict.items():
    preds, actual = predict_series(series, model)
    mape = mean_absolute_percentage_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    results.append({'Ticker': tkr, 'MAPE': mape, 'RMSE': rmse})
    plt.figure(figsize=(8,4))
    plt.plot(actual, 'g', label='Actual')
    plt.plot(preds,  'b', label='Predicted')
    plt.title(f'Predicted vs Actual for {tkr}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

# Display metrics table
metrics_df = pd.DataFrame(results)
st.subheader("📈 Backtesting Metrics")
st.dataframe(metrics_df)
to_metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Metrics CSV", to_metrics_csv, "metrics.csv", "text/csv")

# ─── Next-Day Forecasts & Download ─────────────────
st.subheader("Next-Day Forecasts")
all_preds = []
for tkr, series in data_dict.items():
    last100 = series.dropna().iloc[-100:]
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(last100.values.reshape(-1,1))
    inp = scaled.reshape(1, scaled.shape[0], 1)
    raw = model.predict(inp)[0,0]
    price = float(raw * (1/scaler.scale_[0]))
    last_ts = series.dropna().index[-1]
    yr, mo, dy = last_ts.year, last_ts.month, last_ts.day
    ld = calendar.monthrange(yr, mo)[1]
    if dy == ld:
        nd = date(yr+1,1,1) if mo==12 else date(yr,mo+1,1)
    else:
        nd = (last_ts + timedelta(days=1)).date()
    dt = pd.Timestamp(nd).replace(hour=16,minute=0)
    all_preds.append({'Ticker':tkr,'Next Date':dt,'Predicted Close':price})
pred_df = pd.DataFrame(all_preds)
st.dataframe(pred_df)
to_forecast_csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecasts CSV", to_forecast_csv, "forecasts.csv", "text/csv")

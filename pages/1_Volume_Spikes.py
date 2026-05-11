import pandas as pd
import streamlit as st

from data import (
    current_refresh_bucket,
    fetch_data,
    force_refresh,
    format_adtv,
    format_pct,
    format_volume,
    load_ticker_info,
    load_tickers,
)

st.set_page_config(page_title="Volume Spikes", page_icon="🔥", layout="wide")

ADTV_BASELINES = {"5d": "5d ADTV", "21d": "21d ADTV", "63d": "63d ADTV"}

title_col, refresh_col = st.columns([6, 1])
title_col.title("Volume Spikes")
if refresh_col.button("Refresh now", use_container_width=True):
    force_refresh()

yahoo_tickers = load_tickers()
ticker_info = load_ticker_info()
bucket = current_refresh_bucket()

with st.spinner(f"Fetching data for {len(yahoo_tickers)} stocks…"):
    data = fetch_data(yahoo_tickers, ticker_info, bucket)

if data.empty:
    st.error("No data returned. Check your internet connection or try again.")
    st.stop()

# Exclude mega-caps: keep companies at or below $5B (and unknown market caps).
MAX_MARKET_CAP = 5_000_000_000
data = data[(data["Market Cap"] <= MAX_MARKET_CAP) | data["Market Cap"].isna()]

# --- Sidebar controls ---
st.sidebar.header("Spike filters")

baseline_label = st.sidebar.selectbox("ADTV baseline", list(ADTV_BASELINES.keys()), index=1)
baseline_col = ADTV_BASELINES[baseline_label]

spike_multiple = st.sidebar.slider(
    "Spike multiple (last session vs ADTV)",
    min_value=1.5, max_value=10.0, value=3.0, step=0.5,
)

min_traded = st.sidebar.number_input(
    "Min session traded value ($)",
    min_value=0, value=50_000, step=10_000,
)

pop_threshold = st.sidebar.slider("Price-pop threshold (%)", 0, 50, 10)

st.sidebar.divider()
search = st.sidebar.text_input("Search ticker", "").upper()
sector_options = ["All", "Basic Materials", "Energy", "Healthcare", "Technology"]
selected_sector = st.sidebar.selectbox("Sector", sector_options)

# --- Compute spike multiple ---
working = data.copy()
# Avoid div-by-zero: rows with zero baseline ADTV cannot be ranked.
working = working[working[baseline_col] > 0]
working["Spike Multiple"] = working["Last Session Traded Value"] / working[baseline_col]

# --- Apply filters ---
filtered = working[
    (working["Spike Multiple"] >= spike_multiple)
    & (working["Last Session Traded Value"] >= min_traded)
].copy()

if search:
    filtered = filtered[filtered["Ticker"].str.contains(search, na=False)]
if selected_sector != "All":
    filtered = filtered[filtered["Sector"] == selected_sector]

filtered = filtered.sort_values("Spike Multiple", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1

# --- Pop badge ---
filtered["Pop"] = filtered["Last Session % Change"].apply(
    lambda x: "🔥" if pd.notna(x) and x >= pop_threshold else ""
)

# --- Header caption ---
last_session_dates = data["Last Session Date"].dropna()
latest_session = last_session_dates.max() if not last_session_dates.empty else "—"
st.caption(
    f"Showing **{len(filtered)}** spikes  ·  baseline = **{baseline_label} ADTV**  ·  "
    f"latest session in dataset: **{latest_session}**  ·  refresh bucket: {bucket}  ·  "
    "auto-refreshes after 5pm AEST"
)

if filtered.empty:
    st.info("No tickers match the current spike criteria. Try lowering the spike multiple or min traded value.")
    st.stop()

# --- Display table ---
TABLE_COLS = [
    "Pop", "Ticker", "Company", "Sector",
    "Last Session Date", "Last Session Traded Value", "Last Session % Change",
    "Spike Multiple", "Volume",
    "5d ADTV", "21d ADTV", "63d ADTV",
    "1W % Change", "1M % Change", "3M % Change",
]

styled = filtered[TABLE_COLS].style.format({
    "Last Session Traded Value": lambda x: format_adtv(x),
    "Last Session % Change": lambda x: format_pct(x),
    "Spike Multiple": lambda x: f"{x:.1f}x" if pd.notna(x) else "—",
    "Volume": lambda x: format_volume(x),
    "5d ADTV": lambda x: format_adtv(x),
    "21d ADTV": lambda x: format_adtv(x),
    "63d ADTV": lambda x: format_adtv(x),
    "1W % Change": lambda x: format_pct(x),
    "1M % Change": lambda x: format_pct(x),
    "3M % Change": lambda x: format_pct(x),
})

st.dataframe(styled, use_container_width=True, height=700)

# --- Download ---
csv_export = filtered[TABLE_COLS].to_csv(index=False)
st.download_button("Download CSV", csv_export, file_name="volume_spikes.csv", mime="text/csv")

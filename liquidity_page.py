import streamlit as st
import pandas as pd
from datetime import datetime

from data import (
    PERIODS,
    current_refresh_bucket,
    fetch_data,
    force_refresh,
    format_adtv,
    format_dollar,
    format_volume,
    load_ticker_info,
    load_tickers,
)

st.set_page_config(page_title="Liquidity Screener", page_icon="💧", layout="wide")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
title_col, refresh_col = st.columns([6, 1])
title_col.title("Liquidity Screener")
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

now = datetime.now().strftime("%d %b %Y  %H:%M")
st.caption(
    f"Data refreshed: **{now}**  ·  {len(data)} stocks loaded  ·  "
    f"Auto-refreshes after 5pm AEST (bucket: {bucket})"
)

# --- Sidebar filters ---
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search ticker", "").upper()

ALLOWED_SECTORS = {"Energy", "Healthcare", "Basic Materials", "Technology"}
available_sectors = set(data["Sector"].dropna().unique()) & ALLOWED_SECTORS
sector_options = ["All"] + sorted(available_sectors)
selected_sector = st.sidebar.selectbox("Sector", sector_options)

adtv_col = st.sidebar.selectbox("Filter ADTV by", list(PERIODS.keys()), index=1)
min_adtv = st.sidebar.number_input(
    f"Min {adtv_col} ($)", min_value=0.0, value=0.0, step=10_000.0, format="%.0f"
)

# --- Apply filters ---
filtered = data.copy()
if search:
    filtered = filtered[filtered["Ticker"].str.contains(search, na=False)]
if selected_sector != "All":
    filtered = filtered[filtered["Sector"] == selected_sector]
else:
    filtered = filtered[filtered["Sector"].isin(ALLOWED_SECTORS)]
filtered = filtered[filtered[adtv_col] >= min_adtv]

# --- Sort by 21d ADTV descending by default ---
filtered = filtered.sort_values("21d ADTV", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1  # 1-based

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks shown", len(filtered))
col2.metric("Median 5d ADTV", format_dollar(filtered["5d ADTV"].median()))
col3.metric("Median 21d ADTV", format_dollar(filtered["21d ADTV"].median()))
col4.metric("Median 63d ADTV", format_dollar(filtered["63d ADTV"].median()))

# --- Display table ---
TABLE_COLS = ["Ticker", "Company", "Industry", "Market Cap", "Cash", "Last Price", "Volume", "1W % Change", "1M % Change", "3M % Change", "5d ADTV", "21d ADTV", "63d ADTV"]

styled = filtered[TABLE_COLS].style.format({
    "Market Cap": lambda x: f"${x / 1_000_000:,.1f}m" if pd.notna(x) else "—",
    "Cash": lambda x: f"${x / 1_000_000:,.1f}m" if pd.notna(x) else "—",
    "Last Price": lambda x: f"${x:,.3f}" if pd.notna(x) else "—",
    "Volume": lambda x: format_volume(x),
    "1W % Change": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
    "1M % Change": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
    "3M % Change": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
    "5d ADTV": lambda x: format_adtv(x),
    "21d ADTV": lambda x: format_adtv(x),
    "63d ADTV": lambda x: format_adtv(x),
})

st.dataframe(
    styled,
    use_container_width=True,
    height=700,
)

# --- Download ---
csv_export = filtered[TABLE_COLS].to_csv(index=False)
st.download_button("Download CSV", csv_export, file_name="liquidity_screener.csv", mime="text/csv")

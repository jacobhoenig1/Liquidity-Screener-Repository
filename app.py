import json
import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Liquidity Screener", layout="wide")

APP_DIR = Path(__file__).parent
CSV_PATH = APP_DIR / "Healthcare Cos.csv"
TICKER_INFO_PATH = APP_DIR / "ticker_info.json"
PERIODS = {"5d ADTV": 5, "21d ADTV": 21, "63d ADTV": 63}
CHANGE_PERIODS = {"1W % Change": 5, "1M % Change": 21, "3M % Change": 63}
HISTORY_DAYS = 100  # calendar days to fetch (~70 trading days)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_tickers() -> list[str]:
    df = pd.read_csv(CSV_PATH)
    return df["Companies"].tolist()


def load_ticker_info() -> dict:
    if TICKER_INFO_PATH.exists():
        with open(TICKER_INFO_PATH) as f:
            return json.load(f)
    return {}


def format_dollar(val):
    """Format a number as $X.XK / $X.XM / $X.XB."""
    if pd.isna(val) or val == 0:
        return "—"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"${val / 1_000_000_000:,.1f}B"
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:,.1f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def format_adtv(val):
    """Format ADTV: $X.XM if >= $1M, $X.XK if < $1M."""
    if pd.isna(val) or val == 0:
        return "—"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    return f"${val / 1_000:.1f}K"


def format_volume(val):
    if pd.isna(val) or val == 0:
        return "—"
    if val >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:,.1f}K"
    return f"{val:,.0f}"


CHUNK_SIZE = 25  # download in small batches to avoid timeouts


def _download_chunk(tickers: list[str]) -> pd.DataFrame:
    """Download a chunk of tickers and return raw DataFrame."""
    return yf.download(
        tickers,
        period=f"{HISTORY_DAYS}d",
        group_by="ticker",
        threads=True,
        progress=False,
    )


def _extract_rows(raw, tickers: list[str], ticker_info: dict) -> list[dict]:
    """Extract ADTV rows from a yfinance download result."""
    rows = []
    for yahoo_tick in tickers:
        asx_tick = yahoo_tick.replace(".AX", "")
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[yahoo_tick].copy()
            df = df.dropna(subset=["Close", "Volume"])
            if df.empty:
                continue

            traded_value = df["Close"] * df["Volume"]
            last_price = df["Close"].iloc[-1]
            last_volume = df["Volume"].iloc[-1]

            info = ticker_info.get(yahoo_tick, {})
            row = {
                "Ticker": asx_tick,
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "Last Price": last_price,
                "Volume": last_volume,
            }
            for label, days in PERIODS.items():
                recent = traded_value.tail(days)
                row[label] = recent.mean() if len(recent) > 0 else 0
            for label, days in CHANGE_PERIODS.items():
                if len(df) > days:
                    prev_price = df["Close"].iloc[-(days + 1)]
                    row[label] = ((last_price - prev_price) / prev_price) * 100
                else:
                    row[label] = None
            rows.append(row)
        except Exception:
            continue
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers_yahoo: list[str], ticker_info: dict) -> pd.DataFrame:
    """Download price history in chunks and compute ADTV metrics."""
    all_rows = []
    chunks = [tickers_yahoo[i:i + CHUNK_SIZE] for i in range(0, len(tickers_yahoo), CHUNK_SIZE)]
    progress = st.progress(0, text="Downloading stock data…")

    for idx, chunk in enumerate(chunks):
        try:
            raw = _download_chunk(chunk)
            all_rows.extend(_extract_rows(raw, chunk, ticker_info))
        except Exception:
            pass
        progress.progress((idx + 1) / len(chunks), text=f"Downloaded {min((idx + 1) * CHUNK_SIZE, len(tickers_yahoo))}/{len(tickers_yahoo)} stocks…")

    progress.empty()
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Liquidity Screener")

yahoo_tickers = load_tickers()
ticker_info = load_ticker_info()

with st.spinner(f"Fetching data for {len(yahoo_tickers)} stocks…"):
    data = fetch_data(yahoo_tickers, ticker_info)

if data.empty:
    st.error("No data returned. Check your internet connection or try again.")
    st.stop()

now = datetime.now().strftime("%d %b %Y  %H:%M")
st.caption(f"Data refreshed: **{now}**  ·  {len(data)} stocks loaded  ·  Cached for 1 hour")

# --- Sidebar filters ---
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search ticker", "").upper()

sector_options = ["Basic Materials", "Healthcare", "Technology"]
selected_sector = st.sidebar.selectbox("Sector", sector_options)

adtv_col = st.sidebar.selectbox("Filter ADTV by", list(PERIODS.keys()), index=1)
min_adtv = st.sidebar.number_input(
    f"Min {adtv_col} ($)", min_value=0.0, value=0.0, step=10_000.0, format="%.0f"
)

# --- Apply filters ---
filtered = data.copy()
if search:
    filtered = filtered[filtered["Ticker"].str.contains(search, na=False)]
filtered = filtered[filtered["Sector"] == selected_sector]
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
TABLE_COLS = ["Ticker", "Industry", "Last Price", "Volume", "1W % Change", "1M % Change", "3M % Change", "5d ADTV", "21d ADTV", "63d ADTV"]

styled = filtered[TABLE_COLS].style.format({
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

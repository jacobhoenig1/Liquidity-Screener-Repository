import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).parent
CSV_PATH = APP_DIR / "Healthcare Cos.csv"
TICKER_INFO_PATH = APP_DIR / "ticker_info.json"

PERIODS = {"5d ADTV": 5, "21d ADTV": 21, "63d ADTV": 63}
CHANGE_PERIODS = {"1W % Change": 5, "1M % Change": 21, "3M % Change": 63}
HISTORY_DAYS = 100  # calendar days to fetch (~70 trading days)
CHUNK_SIZE = 25
MAX_MARKET_CAP = 5_000_000_000  # exclude mega-caps from the screener
METADATA_WORKERS = 32  # threads for per-ticker fast_info calls
DOWNLOAD_WORKERS = 5  # concurrent yf.download chunk requests


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_tickers() -> list[str]:
    df = pd.read_csv(CSV_PATH)
    return df["Companies"].tolist()


def load_ticker_info() -> dict:
    if TICKER_INFO_PATH.exists():
        with open(TICKER_INFO_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def format_dollar(val):
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


def format_pct(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.1f}%"


# ---------------------------------------------------------------------------
# 5pm AEST refresh bucket
# ---------------------------------------------------------------------------
def current_refresh_bucket() -> str:
    """Cache key that flips at 17:00 Australia/Sydney each day."""
    now_aest = datetime.now(ZoneInfo("Australia/Sydney"))
    return (now_aest - timedelta(hours=17)).date().isoformat()


# ---------------------------------------------------------------------------
# Fetch + compute
# ---------------------------------------------------------------------------
def _download_chunk(tickers: list[str]) -> pd.DataFrame:
    return yf.download(
        tickers,
        period=f"{HISTORY_DAYS}d",
        group_by="ticker",
        threads=True,
        progress=False,
    )


def _fetch_market_cap(yahoo_tick: str):
    try:
        return yf.Ticker(yahoo_tick).fast_info.get("marketCap", None)
    except Exception:
        return None


def _build_price_row(yahoo_tick: str, df: pd.DataFrame, ticker_info: dict) -> dict | None:
    df = df.dropna(subset=["Close", "Volume"])
    if df.empty:
        return None

    traded_value = df["Close"] * df["Volume"]
    last_price = df["Close"].iloc[-1]
    last_volume = df["Volume"].iloc[-1]
    last_session_date = df.index[-1].date()
    last_session_traded_value = float(last_price) * float(last_volume)

    if len(df) >= 2:
        prev_close = df["Close"].iloc[-2]
        last_session_pct = ((last_price - prev_close) / prev_close) * 100
    else:
        last_session_pct = None

    info = ticker_info.get(yahoo_tick, {})
    row = {
        "Ticker": yahoo_tick.replace(".AX", ""),
        "Company": info.get("name", "Unknown"),
        "Sector": info.get("sector", "Unknown"),
        "Industry": info.get("industry", "Unknown"),
        "Market Cap": None,
        "Last Price": last_price,
        "Volume": last_volume,
        "Last Session Date": last_session_date,
        "Last Session Traded Value": last_session_traded_value,
        "Last Session % Change": last_session_pct,
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
    return row


def _download_and_build(chunk: list[str], ticker_info: dict) -> list[tuple[str, dict]]:
    """Download one chunk of price history and build rows (no metadata calls)."""
    raw = _download_chunk(chunk)
    rows: list[tuple[str, dict]] = []
    for yahoo_tick in chunk:
        try:
            df = raw.copy() if len(chunk) == 1 else raw[yahoo_tick].copy()
            row = _build_price_row(yahoo_tick, df, ticker_info)
            if row is not None:
                rows.append((yahoo_tick, row))
        except Exception:
            continue
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers_yahoo: list[str], ticker_info: dict, refresh_bucket: str) -> pd.DataFrame:
    """Download price history in parallel chunks and compute all metrics.

    `refresh_bucket` participates in the cache key so the cache invalidates
    automatically after each 17:00 Australia/Sydney boundary. The argument is
    not used inside the function body.
    """
    _ = refresh_bucket
    chunks = [tickers_yahoo[i:i + CHUNK_SIZE] for i in range(0, len(tickers_yahoo), CHUNK_SIZE)]
    progress = st.progress(0, text="Downloading stock data…")

    # Phase 1: download price history for all chunks concurrently.
    pending: list[tuple[str, dict]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = [ex.submit(_download_and_build, chunk, ticker_info) for chunk in chunks]
        for fut in as_completed(futures):
            try:
                pending.extend(fut.result())
            except Exception:
                pass
            done += 1
            progress.progress(
                done / len(chunks) * 0.7,
                text=f"Downloaded {min(done * CHUNK_SIZE, len(tickers_yahoo))}/{len(tickers_yahoo)} stocks…",
            )

    if not pending:
        progress.empty()
        return pd.DataFrame()

    # Phase 2: fetch market caps in parallel and drop mega-caps.
    progress.progress(0.7, text=f"Fetching market caps for {len(pending)} stocks…")
    pending_ticks = [t for t, _ in pending]
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as ex:
        market_caps = list(ex.map(_fetch_market_cap, pending_ticks))

    all_rows = []
    for (_, row), cap in zip(pending, market_caps):
        if cap is not None and cap > MAX_MARKET_CAP:
            continue
        row["Market Cap"] = cap
        all_rows.append(row)

    progress.empty()
    return pd.DataFrame(all_rows)


def force_refresh():
    """Clear all cached data and rerun the page."""
    st.cache_data.clear()
    st.rerun()

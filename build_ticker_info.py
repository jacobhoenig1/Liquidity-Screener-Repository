"""
Pre-cache sector and industry data from Yahoo Finance into ticker_info.json.
Run this locally whenever you update Healthcare Cos.csv with new tickers.

Usage:
    python build_ticker_info.py
"""

import json
import pandas as pd
import yfinance as yf
from pathlib import Path

CSV_PATH = Path(__file__).parent / "Healthcare Cos.csv"
OUT_PATH = Path(__file__).parent / "ticker_info.json"

tickers_df = pd.read_csv(CSV_PATH)
yahoo_tickers = tickers_df["Companies"].tolist()

# Load existing data so we only fetch new tickers
existing = {}
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        existing = json.load(f)

ticker_info = dict(existing)
new_tickers = [t for t in yahoo_tickers if t not in ticker_info]

print(f"{len(yahoo_tickers)} tickers in CSV, {len(existing)} already cached, {len(new_tickers)} to fetch")

for i, tick in enumerate(new_tickers):
    try:
        info = yf.Ticker(tick).info
        ticker_info[tick] = {
            "sector": info.get("sector", "Unknown") or "Unknown",
            "industry": info.get("industry", "Unknown") or "Unknown",
        }
    except Exception:
        ticker_info[tick] = {"sector": "Unknown", "industry": "Unknown"}

    if (i + 1) % 10 == 0 or i == len(new_tickers) - 1:
        print(f"  {i + 1}/{len(new_tickers)}: {tick} -> {ticker_info[tick]['sector']} / {ticker_info[tick]['industry']}")

# Remove tickers no longer in CSV
ticker_info = {k: v for k, v in ticker_info.items() if k in yahoo_tickers}

with open(OUT_PATH, "w") as f:
    json.dump(ticker_info, f, indent=2)

print(f"\nSaved {len(ticker_info)} tickers to {OUT_PATH}")

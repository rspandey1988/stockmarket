import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests

# Read Telegram credentials from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# Nifty 50 tickers
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS", "MARUTI.NS",
    "NESTLEIND.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "WIPRO.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "GRASIM.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "ADANIPORTS.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "BPCL.NS", "TECHM.NS", "HEROMOTOCO.NS",
    "HCLTECH.NS", "M&M.NS", "EICHERMOT.NS", "CIPLA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS",
    "TATAMOTORS.NS", "SHREECEM.NS", "IOC.NS", "TATACONSUM.NS", "UPL.NS", "VEDL.NS", "BRITANNIA.NS",
    "BAJAJ-AUTO.NS"
]

# WMA calculation
def compute_wma_and_slope(df):
    df['WMA'] = df['Close'].rolling(window=30).apply(
        lambda prices: np.dot(prices, np.arange(1, 31)) / np.arange(1, 31).sum(), raw=True
    )
    df['WMA_Slope'] = df['WMA'].diff()
    return df

# Runtime signal detection
def check_runtime_signal(ticker):
    try:
        df = yf.download(ticker, period="90d", interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 40:
            return

        df = compute_wma_and_slope(df)
        current = df.iloc[-1]
        close, wma, slope = current['Close'], current['WMA'], current['WMA_Slope']

        if pd.isna(wma) or pd.isna(slope):
            return

        if close > wma and slope > 0:
            send_telegram_message(f"📈 BUY Signal for {ticker}: Price={close:.2f}, WMA={wma:.2f}, Slope={slope:.2f}")
        elif close < wma and slope < 0:
            send_telegram_message(f"📉 SELL Signal for {ticker}: Price={close:.2f}, WMA={wma:.2f}, Slope={slope:.2f}")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Run for all tickers
for ticker in nifty50_tickers:
    check_runtime_signal(ticker)

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import threading
import time
import os

# List of Nifty 50 tickers
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS", "MARUTI.NS", "NESTLEIND.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS", "WIPRO.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "GRASIM.NS", "DIVISLAB.NS", "DRREDDY.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS",
    "COALINDIA.NS", "BPCL.NS", "TECHM.NS", "HEROMOTOCO.NS", "HCLTECH.NS", "M&M.NS", "EICHERMOT.NS",
    "CIPLA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", "TATAMOTORS.NS", "SHREECEM.NS",
    "IOC.NS", "TATACONSUM.NS", "UPL.NS", "VEDL.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS"
]

# Telegram setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Failed to send message: {response.text}")
    except Exception as e:
        print(f"Error sending message: {e}")

# Compute 30-week WMA and slope
def compute_wma_and_slope(df):
    df['WMA'] = df['Close'].rolling(window=30).apply(
        lambda prices: np.dot(prices, np.arange(1, 31)) / np.arange(1, 31).sum(), raw=True
    )
    df['WMA_Slope'] = df['WMA'].diff()
    return df

# Monitoring function for each ticker
def monitor_ticker(ticker):
    in_position = False
    print(f"Starting monitoring for {ticker}")
    while True:
        try:
            # Fetch 1-minute interval data for today
            df = yf.download(ticker, period='1d', interval='1m', auto_adjust=True, progress=False)
            if df.empty:
                print(f"No data for {ticker}")
                time.sleep(60)
                continue

            df = compute_wma_and_slope(df)
            latest = df.iloc[-1]
            close = float(latest['Close'])
            wma = float(latest['WMA'])
            slope = float(latest['WMA_Slope'])

            # Generate signals
            if not in_position and close > wma and slope > 0:
                message = f"🚀 *BUY* signal for {ticker} at {close:.2f}"
                send_telegram_message(message)
                in_position = True
                print(f"Buy signal for {ticker} at {close:.2f}")
            elif in_position and close < wma and slope < 0:
                message = f"🔻 *SELL* signal for {ticker} at {close:.2f}"
                send_telegram_message(message)
                in_position = False
                print(f"Sell signal for {ticker} at {close:.2f}")
        except Exception as e:
            print(f"Error monitoring {ticker}: {e}")

        time.sleep(60)  # Wait 1 minute before next check

# Run monitoring for all tickers in parallel
threads = []

for ticker in nifty50_tickers:
    t = threading.Thread(target=monitor_ticker, args=(ticker,))
    t.daemon = True  # Daemon threads will exit when main thread exits
    t.start()
    threads.append(t)

# Keep main thread alive
try:
    while True:
        time.sleep(600)  # Sleep for 10 minutes; keep script running
except KeyboardInterrupt:
    print("Stopping monitoring.")

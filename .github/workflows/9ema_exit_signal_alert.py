import yfinance as yf
import pandas as pd
import requests
import time
import os

# Telegram setup
TELEGRAM_TOKEN = os.getenv("SWINGTRADE_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("SWINGTRADE_TELEGRAM_CHAT_ID")

# List of Nifty 50 stocks (symbols on Yahoo Finance)
nifty50_symbols = [
    '^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HDFC.NS',
    # Add remaining Nifty 50 stocks symbols here
]

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=payload)

def fetch_stock_data(symbol):
    # Fetch latest 20 days of data for daily timeframe
    df = yf.download(symbol, period='30d', interval='1d')
    return df

def check_breakdown(df):
    if len(df) < 10:
        return False, None

    # Calculate 9-day EMA
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    # Find the breakdown candle (close > EMA on previous day)
    for i in range(1, len(df)):
        prev_close = df.iloc[i - 1]['Close']
        prev_ema = df.iloc[i - 1]['EMA9']
        curr_close = df.iloc[i]['Close']
        curr_low = df.iloc[i]['Low']
        date = df.index[i]

        # Break down condition: previous close above EMA, current close below EMA
        if prev_close > prev_ema and curr_close < prev_ema:
            # Store the low of the breakdown candle
            return True, df.iloc[i]
    return False, None

def monitor_stocks():
    alerts_sent = set()

    for symbol in nifty50_symbols:
        df = fetch_stock_data(symbol)
        breakdown, candle = check_breakdown(df)
        if breakdown:
            low_breakdown_candle = candle['Low']
            date_of_candle = candle.name.strftime('%Y-%m-%d')
            # Fetch latest data to check if today's close is below low
            latest_df = yf.download(symbol, period='2d', interval='1d')
            latest_close = latest_df['Close'][-1]

            if latest_close < low_breakdown_candle:
                message = (f"🚨 Exit Alert for {symbol}\n"
                           f"Breakdown Candle Date: {date_of_candle}\n"
                           f"Breakdown Candle Low: {low_breakdown_candle}\n"
                           f"Latest Close: {latest_close}\n"
                           f"Time: {latest_df.index[-1].strftime('%Y-%m-%d')}")
                if symbol not in alerts_sent:
                    send_telegram_message(message)
                    alerts_sent.add(symbol)

# Run daily or as needed
if __name__ == "__main__":
    while True:
        monitor_stocks()
        print("Checked all stocks. Waiting for next check...")
        time.sleep(24 * 60 * 60)  # Run every 24 hours

import yfinance as yf
import pandas as pd
import requests
import logging
import os

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Telegram setup
SWINGTRADE_TELEGRAM_BOT_TOKEN = os.getenv("SWINGTRADE_TELEGRAM_BOT_TOKEN")
SWINGTRADE_TELEGRAM_CHAT_ID = os.getenv("SWINGTRADE_TELEGRAM_CHAT_ID")

# List of Nifty 50 stocks
nifty50_symbols = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "LT.NS",
    "HDFC.NS", "MARUTI.NS", "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS",
    "ADANIPORTS.NS", "TITAN.NS", "NESTLEIND.NS", "TATASTEEL.NS", "DIVISLAB.NS",
    "M&M.NS", "DRREDDY.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "CIPLA.NS",
    "EICHERMOT.NS", "SHREECEM.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS", "BPCL.NS", "IOCL.NS",
    "POWERGRID.NS", "TATACONSUM.NS", "ADANIGREEN.NS", "APOLLOHOSP.NS",
    "BRITANNIA.NS", "HINDALCO.NS", "NESTLEIND.NS", "TATAPOWER.NS", "UPL.NS",
    "VEDL.NS", "COALINDIA.NS"
]

def send_telegram_message(message):
    """Send message via Telegram."""
    url = f'https://api.telegram.org/bot{SWINGTRADE_TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': SWINGTRADE_TELEGRAM_CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logging.info("Telegram message sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def fetch_stock_data(symbol, period='30d'):
    """Fetch stock data from Yahoo Finance."""
    try:
        logging.info(f"Fetching data for {symbol} for period {period}...")
        df = yf.download(symbol, period=period, interval='1d', auto_adjust=True)
        if df.empty:
            logging.warning(f"No data fetched for {symbol}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def check_breakdown(df):
    """Check for EMA9 crossover indicating breakdown."""
    if len(df) < 10:
        logging.info("Not enough data for breakdown check.")
        return False, None

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    for i in range(1, len(df)):
        prev_close = float(df['Close'].iloc[i - 1])
        prev_ema = float(df['EMA9'].iloc[i - 1])
        curr_close = float(df['Close'].iloc[i])
        curr_low = float(df['Low'].iloc[i])
        date = df.index[i]

        if prev_close > prev_ema and curr_close < prev_ema:
            logging.info(f"Breakdown detected on {date.date()}: Close {curr_close} crossed below EMA9 {prev_ema}.")
            return True, df.iloc[i]
    logging.info("No breakdown detected in data.")
    return False, None

def monitor_stocks():
    alerts_sent = set()

    for symbol in nifty50_symbols:
        logging.info(f"Processing {symbol}...")
        df = fetch_stock_data(symbol)
        if df.empty:
            logging.warning(f"Skipping {symbol} due to no data.")
            continue

        breakdown, candle = check_breakdown(df)
        if breakdown:
            low_breakdown_candle = float(candle['Low'])
            date_of_candle = candle.name.strftime('%Y-%m-%d')
            latest_df = yf.download(symbol, period='2d', interval='1d', auto_adjust=True)
            if latest_df.empty:
                logging.warning(f"No latest data for {symbol}. Skipping.")
                continue

            latest_close = float(latest_df['Close'].iloc[-1])
            logging.info(f"{symbol} latest close: {latest_close} (type: {type(latest_close)})")

            if latest_close < low_breakdown_candle:
                message = (
                    f"🚨 Exit Alert for {symbol}\n"
                    f"Breakdown Candle Date: {date_of_candle}\n"
                    f"Breakdown Candle Low: {low_breakdown_candle}\n"
                    f"Latest Close: {latest_close}\n"
                    f"Time: {latest_df.index[-1].strftime('%Y-%m-%d')}"
                )
                if symbol not in alerts_sent:
                    send_telegram_message(message)
                    logging.info(f"Alert sent for {symbol}.")
                    alerts_sent.add(symbol)
                else:
                    logging.info(f"Alert already sent for {symbol}, skipping.")
            else:
                logging.info(f"{symbol} latest close {latest_close} is not below breakdown low {low_breakdown_candle}.")
        else:
            logging.info(f"No breakdown for {symbol}.")

if __name__ == "__main__":
    logging.info("Starting stock analysis...")
    monitor_stocks()
    logging.info("Analysis complete.")

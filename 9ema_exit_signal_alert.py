import yfinance as yf
import pandas as pd
import requests
import logging
import os
import glob
import matplotlib.pyplot as plt

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("SWINGTRADE_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("SWINGTRADE_TELEGRAM_CHAT_ID")

# Cache directory
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Dictionary of stock tickers to names
nifty50_tickers = {
    "AAPL": "Apple Stock",
}

def clear_cache():
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    for file_path in cache_files:
        try:
            os.remove(file_path)
            logging.info(f"Deleted cache file: {file_path}")
        except Exception as e:
            logging.warning(f"Failed to delete {file_path}: {e}")

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logging.info("Telegram message sent successfully.")
        return True
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
        return False

def load_cached_data(symbol):
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logging.debug(f"Loaded cached data for {symbol} with {len(df)} rows.")
            return df
        except Exception as e:
            logging.warning(f"Failed to load cache for {symbol}: {e}")
    else:
        logging.debug(f"No cache found for {symbol}")
    return None

def save_cache_data(symbol, df):
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")
    df.to_csv(filepath)
    logging.info(f"Saved cache for {symbol} with {len(df)} rows.")

def fetch_data_for_symbol(symbol):
    df_cached = load_cached_data(symbol)
    if df_cached is not None:
        logging.info(f"Using cached data for {symbol} with {len(df_cached)} rows.")
        return df_cached
    try:
        df_new = yf.download(symbol, period='10d', interval='1d', auto_adjust=True)
        if not df_new.empty:
            logging.info(f"Fetched {len(df_new)} rows for {symbol}")
            save_cache_data(symbol, df_new)
        else:
            logging.warning(f"No data received for {symbol}")
        return df_new
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def check_breakdown(df):
    """Check for EMA9 crossover indicating breakdown."""
    if len(df) < 10:
        logging.debug(f"Not enough data ({len(df)}) to check for breakdown.")
        return False, None

    # Check columns existence
    if 'Close' not in df.columns or 'Low' not in df.columns:
        logging.warning("Missing 'Close' or 'Low' columns in data.")
        return False, None

    # Convert to numeric safely
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

    # Drop NaNs
    df.dropna(subset=['Close', 'Low'], inplace=True)

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i - 1]
        prev_ema = df['EMA9'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]
        date = df.index[i]
        logging.debug(f"Date: {date.date()}, Prev Close: {prev_close}, Prev EMA: {prev_ema}, Curr Close: {curr_close}")

        if prev_close >= prev_ema and curr_close <= prev_ema:
            logging.info(f"Breakdown detected on {date.date()}")
            return True, df.iloc[i]
    return False, None

# Prepare summary list
summary_list = []

def process_stock(ticker, name):
    """Process each stock and update summary info"""
    record = {
        'Stock': name,
        'Ticker': ticker,
        'Exit Triggered': 'No',
        'Alert Sent': 'No',
        'Breakdown Candle Date': None,
        'Breakdown Candle Low': None,
        'Current Close Price': None,
        'Date': None
    }

    df = fetch_data_for_symbol(ticker)
    if df is None or df.empty:
        logging.warning(f"No data to process for {name} ({ticker})")
        return record

    # Check if enough data exists
    if len(df) < 10:
        logging.info(f"Not enough data ({len(df)}) for {name} ({ticker}) to perform analysis.")
        return record

    breakdown, candle = check_breakdown(df)
    if breakdown:
        low_breakdown_candle = candle['Low']
        date_of_candle = candle.name.strftime('%Y-%m-%d')
        latest_close = df['Close'].iloc[-1]
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        
        logging.info(f"Latest close for {name} ({ticker}): {latest_close}")

        if latest_close < low_breakdown_candle:
            message = (
                f"🚨 Exit Alert for {name} ({ticker})\n"
                f"Breakdown Candle Date: {date_of_candle}\n"
                f"Breakdown Candle Low: {low_breakdown_candle}\n"
                f"Latest Close: {latest_close}\n"
                f"Date: {latest_date}"
            )
            logging.info(message)
            sent = send_telegram_message(message)
            if sent:
                logging.info(f"Telegram alert sent for {name} ({ticker})")
            else:
                logging.warning(f"Failed to send Telegram alert for {name} ({ticker})")
            record.update({
                'Exit Triggered': 'Yes',
                'Alert Sent': 'Yes' if sent else 'No',
                'Breakdown Candle Date': date_of_candle,
                'Breakdown Candle Low': low_breakdown_candle,
                'Current Close Price': latest_close,
                'Date': latest_date
            })
        else:
            logging.info(f"No trigger: latest close {latest_close} is not below breakdown low {low_breakdown_candle}")
    else:
        logging.info(f"No breakdown detected for {name} ({ticker})")
    return record

if __name__ == "__main__":

    logging.info("Starting stock analysis with detailed logs...")
    for ticker, name in nifty50_tickers.items():
        rec = process_stock(ticker, name)
        summary_list.append(rec)

    # Create DataFrame from summary
    summary_df = pd.DataFrame(summary_list)

    # Print the summary
    print("\nSummary of Exit Signals:\n")
    print(summary_df)

    # Plot with highlighting
    fig, ax = plt.subplots(figsize=(14, max(6, len(summary_df) * 0.5)))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Highlight rows where Exit Triggered == 'Yes'
    for i, row in enumerate(summary_df.itertuples(index=False)):
        if row._2 == 'Yes':  # 'Exit Triggered' column
            for j in range(len(summary_df.columns)):
                cell = table[(i + 1, j)]
                cell.set_facecolor('red')
                cell.set_text_props(color='white')

    #plt.title('Stock Exit Signal Summary', fontsize=14)
    #plt.tight_layout()
    #plt.show()

    logging.info("Analysis complete.")

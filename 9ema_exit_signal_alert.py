import os
import glob
import requests
import yfinance as yf
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import ta

# Set default font to avoid font matching delays
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Delete matplotlib font cache to force rebuild
cache_dir = os.path.expanduser("~/.cache/matplotlib")
font_cache_file = os.path.join(cache_dir, "fontlist-v310.json")  # adjust if needed

if os.path.exists(font_cache_file):
    os.remove(font_cache_file)
    print(f"Deleted font cache: {font_cache_file}")

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("SWINGTRADE_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("SWINGTRADE_TELEGRAM_CHAT_ID")

# Cache directory
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Dictionary of stock tickers to names
nifty50_tickers = {
    "APLAPOLLO.NS": "Apollo Hospitals",
    "GRWRHITECH.NS": "G R Woth Tech",
    "ABSLAMC.NS": "Aditya Birla Sun Life AMC",
    "TDPOWERSYS.NS": "Tata Power Sectors",
    "JMFINANCIL.NS": "JM Financial",
    "WOCKPHARMA.NS": "Wockhardt Pharma",
    "KITEX.NS": "Kitex Garments",
    "CARERATING.NS": "CARE Ratings",
    "MANAPPURAM.NS": "Manappuram Finance",
    "CHOLAFIN.NS": "Cholamandalam Finance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "AUBANK.NS": "AU Small Finance Bank",
    "SUPREMEIND.NS": "Supreme Industries"
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
            # Verify and fix index if necessary
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                # Get index as strings to filter valid dates
                index_strs = df.index.astype(str)
                valid_dates = []
                for val in index_strs:
                    try:
                        # Try parsing each index value
                        pd.to_datetime(val)
                        valid_dates.append(val)
                    except:
                        # skip invalid entries
                        pass
                # Keep only valid date entries
                df = df.loc[[idx in valid_dates for idx in index_strs]].copy()
                # Convert index to datetime
                df.index = pd.to_datetime(df.index)
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
    """Check for EMA9 crossover indicating breakdown using ta library."""
    if len(df) < 10:
        logging.debug(f"Not enough data ({len(df)}) to check for breakdown.")
        return False, None

    if not isinstance(df, pd.DataFrame):
        logging.warning("Provided data is not a DataFrame.")
        return False, None

    expected_cols = {'Close', 'Low'}
    if not expected_cols.issubset(df.columns):
        logging.warning(f"DataFrame missing expected columns. Found columns: {df.columns}")
        return False, None

    try:
        close_series = pd.to_numeric(df['Close'], errors='coerce')
        low_series = pd.to_numeric(df['Low'], errors='coerce')
    except Exception as e:
        logging.warning(f"Failed to convert price columns to numeric: {e}")
        return False, None

    # Drop NaN values
    df_clean = df.loc[close_series.notna() & low_series.notna()].copy()
    df_clean['Close'] = close_series.loc[df_clean.index]
    df_clean['Low'] = low_series.loc[df_clean.index]

    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean.index):
        df_clean.index = pd.to_datetime(df_clean.index)

    # Calculate EMA9 using ta library
    df_clean['EMA9'] = ta.trend.ema_indicator(df_clean['Close'], window=9)

    # Check for crossover
    for i in range(1, len(df_clean)):
        prev_close = df_clean['Close'].iloc[i - 1]
        prev_ema = df_clean['EMA9'].iloc[i - 1]
        curr_close = df_clean['Close'].iloc[i]
        curr_ema = df_clean['EMA9'].iloc[i]
        date = df_clean.index[i]

        # Detect crossover from above/equal to below
        if prev_close > prev_ema and curr_close < curr_ema:
            logging.info(f"Breakdown detected on {date.date()}: "
                         f"prev_close={prev_close}, prev_ema={prev_ema}, "
                         f"curr_close={curr_close}, curr_ema={curr_ema}")
            return True, {
                'Low': df_clean['Low'].iloc[i],
                'date': date
            }

    return False, None




def process_stock(ticker, name):
    """Process each stock and update the summary info."""
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

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logging.warning(f"Failed to convert index for {name} ({ticker}): {e}")
            return record

    # Check if enough data exists
    if len(df) < 10:
        logging.info(f"Not enough data ({len(df)}) for {name} ({ticker}) to perform analysis.")
        return record

    # Check for breakdown
    breakdown, candle_info = check_breakdown(df)
    if breakdown:
        low_value = float(candle_info['Low'])  # ensure scalar float
        date_of_candle = candle_info['date'].strftime('%Y-%m-%d')

        try:
            latest_close = float(df['Close'].iloc[-1])
        except (ValueError, TypeError) as e:
            logging.warning(f"Invalid latest close price for {name} ({ticker}): {e}")
            return record

        latest_date = df.index[-1]
        latest_date_str = latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else 'Unknown'

        logging.info(f"Latest close for {name} ({ticker}): {latest_close}")# Fetch historical data for the last 5 days (to handle weekends/holidays)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        # Drop any rows without closing price
        hist = hist.dropna(subset=["Close"])

        # Get the most recent closing price
        latest_close = hist["Close"][-1]
        latest_date = hist.index[-1].date()

        print(f"Latest trading day: {latest_date}")
        print(f"{ticker} closing price: {latest_close:.2f}")

        # Compare latest close with breakdown low
        if latest_close < low_value:
            message = (
                f"🚨 Exit Alert for {name} ({ticker})\n"
                f"Breakdown Candle Date: {date_of_candle}\n"
                f"Breakdown Candle Low: {low_value}\n"
                f"Latest Close: {latest_close}\n"
                f"Date: {latest_date}"
            )
            logging.info(message)
            sent = send_telegram_message(message)
            if sent:
                logging.info(f"Telegram alert sent for {name} ({ticker})")
                record['Exit Triggered'] = 'Yes'
                record['Alert Sent'] = 'Yes'
            else:
                logging.warning(f"Failed to send Telegram alert for {name} ({ticker})")
                record['Exit Triggered'] = 'Yes'
                record['Alert Sent'] = 'No'

            # Update record with details
            record.update({
                'Breakdown Candle Date': date_of_candle,
                'Breakdown Candle Low': low_value,
                'Current Close Price': latest_close,
                'Date': latest_date_str
            })
        else:
            logging.info(f"No trigger: latest close {latest_close} is not below breakdown low {low_value}")
    else:
        logging.info(f"No breakdown detected for {name} ({ticker})")

    return record


if __name__ == "__main__":
    # Optional: clear cache before starting
    # clear_cache()

    logging.info("Starting stock analysis with detailed logs...")
    summary_list = []

    for ticker, name in nifty50_tickers.items():
        rec = process_stock(ticker, name)
        summary_list.append(rec)

    # Create DataFrame from summary
    summary_df = pd.DataFrame(summary_list)

    # Print the summary
    print("\nSummary of Exit Signals:\n")
    print(summary_df)

    logging.info("Analysis complete.")

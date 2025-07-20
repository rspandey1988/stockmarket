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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
    "SUPREMEIND.NS": "Supreme Industries",
    "SUNDARMFIN.NS": "Sundaram Finance",
    "CLSEL.NS" : "Chamanlal Seta",
    "THYROCARE.NS": "THYROCARE"
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
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
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
        curr_close = df_clean['Close'].iloc[i]
        curr_ema = df_clean['EMA9'].iloc[i]
        date = df_clean.index[i]

        print(f"Date: {date.date()} | Close: {curr_close:.2f} | EMA9: {curr_ema:.2f}")

        # Detect crossover from above/equal to below
        if curr_close < curr_ema:
            logging.info(f"Breakdown detected on {date.date()}: "
                         f"curr_close={curr_close}, curr_ema={curr_ema}")
            return True, {
                'Low': df_clean['Low'].iloc[i],
                'date': date
            }

    return False, None


def process_stock(ticker, name):
    """Process each stock and update the summary info with exit condition based on next candle."""
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
        date_of_candle = candle_info['date']
        low_of_candle = float(candle_info['Low'])

        # Log breakdown candle info
        logging.info(f"Breakdown Candle Date: {date_of_candle}")
        logging.info(f"Breakdown Candle Low: {low_of_candle}")

        # Fetch the next day's candle data
        next_day_start = date_of_candle + pd.Timedelta(days=1)
        next_day_end = next_day_start + pd.Timedelta(days=2)  # buffer for weekends/holidays

        next_candle_df = yf.download(ticker, start=next_day_start, end=next_day_end, interval='1d')

        if not next_candle_df.empty:
            next_candle = next_candle_df.iloc[0]
            # Debug print to verify structure
            print("Next candle data:\n", next_candle)

            # Extract date
            next_candle_date = next_candle_df.index[0].date()

            # Convert to float explicitly
            try:
                close_price = float(next_candle['Close'])
                open_price = float(next_candle['Open'])
            except Exception as e:
                logging.warning(f"Error converting next candle data to float: {e}")
                close_price = None
                open_price = None

            # Log details
            logging.info(f"Next Candle Date: {next_candle_date}")
            logging.info(f"Next Candle Close Price: {close_price}")

            if close_price is not None and open_price is not None:
                # Check if bearish
                if close_price < open_price:
                    # Check if close < breakdown candle low
                    if close_price < low_of_candle:
                        # Trigger exit alert
                        message = (
                            f"🚨 Exit Confirmation for {name} ({ticker})\n"
                            f"Breakdown Candle Date: {date_of_candle}\n"
                            f"Breakdown Candle Low: {low_of_candle}\n"
                            f"Next Candle Date: {next_candle_date}\n"
                            f"Next Candle Close: {close_price}\n"
                            f"Next Candle is Bearish and closes below breakdown low."
                        )
                        sent = send_telegram_message(message)
                        if sent:
                            logging.info(f"Exit confirmation sent for {name} ({ticker})")
                            record['Exit Triggered'] = 'Yes'
                            record['Alert Sent'] = 'Yes'
                        else:
                            logging.warning(f"Failed to send exit confirmation for {name} ({ticker})")
                            record['Exit Triggered'] = 'Yes'
                            record['Alert Sent'] = 'No'
                else:
                    logging.info("Next candle is not bearish; no exit triggered.")
            else:
                logging.warning("Could not process next candle prices due to conversion error.")
        else:
            logging.info(f"No data available for next candle after {date_of_candle} for {name} ({ticker})")
    else:
        # No breakdown detected
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
    sent = send_telegram_message(summary_df)
    if sent:
        print("\n successfully sent message\n")

    logging.info("Analysis complete.")

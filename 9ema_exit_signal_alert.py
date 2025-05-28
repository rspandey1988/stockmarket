"""
Breakdown EMA Strategy — Detailed Description
Overview
The Breakdown EMA Strategy is a technical trading approach designed to identify potential bearish exit signals in stocks based on exponential moving average (EMA) crossovers and price breakdowns. It aims to assist traders in systematically detecting points where an existing long position might be at risk, enabling timely exit decisions to preserve gains or minimize losses.

Core Principles

Trend Confirmation via EMA9:
The 9-day EMA acts as a dynamic support/resistance level, capturing short-term momentum. When the price is above EMA9, the stock is typically in a bullish or neutral trend; when it crosses below, it suggests potential trend reversal or weakening momentum.



Crossover Signal (Breakdown):
A breakdown occurs when the stock's daily closing price crosses from above to below the EMA9, signaling a possible shift to bearish momentum.



Confirmation of Breakdown:
To avoid false signals, the system checks if the latest closing price remains below the lowest low of the candle during the breakdown. If the current close stays below this level, it confirms a genuine breakdown, indicating a strong bearish move.



Step-by-Step Logic

Detection of EMA9 Crossover (Breakdown Candle):


The system scans recent data for instances where:

The previous day's close was above EMA9.

The current day's close has fallen below EMA9.


Such a crossover indicates a potential shift from bullish to bearish momentum.
The lowest price during this breakdown day (the candle's low) is recorded as the breakdown candle low.


Monitoring for Confirmation:


After detecting the breakdown candle, the system continually checks the latest available data.
It fetches the most recent two days' data to analyze the current close.
If the latest close remains below the recorded breakdown candle low, it confirms that the price has indeed broken down and stayed below critical support.


Triggering an Exit Alert:


Upon confirmation, a Telegram alert is sent to notify the trader to exit the position.
The alert includes:

The stock symbol.

The breakdown candle date.

The breakdown candle low.

The latest close price.

The date of the latest data point.




Avoiding Duplicate Alerts:


The system maintains a set of stocks for which alerts have already been sent during that execution to prevent repeated notifications.


Underlying Assumptions & Rationale

Trend Reversal Signal:
The crossover from above to below EMA9 is considered an early warning of a potential bearish reversal.



Breakdown Confirmation:
Waiting for the latest close to fall below the breakdown candle's low reduces the chance of false signals caused by minor fluctuations or whipsaws.



Exit Timing:
This method aims to provide a systematic way to exit trades before significant downside occurs, based on technical breakdowns rather than subjective judgment.



Advantages of the Strategy

Early Exit Recognition:
Detects potential downtrends promptly, allowing traders to protect profits or prevent further losses.



Objective & Systematic:
Uses clear rules to identify breakdowns, removing emotional biases.



Adaptable:
Can be integrated with other indicators or rules for more comprehensive strategies.



Limitations & Considerations

False Signals:
Like all technical strategies, false breakdowns can occur, especially in volatile markets.



Market Conditions:
Works best in trending markets; sideways or choppy markets may generate frequent false signals.



Parameter Sensitivity:
The choice of EMA period and look-back window influences sensitivity; tuning is recommended based on the asset and timeframe.



Time Frame:
The current implementation uses daily data, suitable for swing or position traders. Shorter or longer periods can be adapted.



Summary
The Breakdown EMA Strategy systematically identifies potential exit points in stocks by monitoring the crossing of the price below the EMA9 and confirming the breakdown through the latest close remaining below the breakdown candle's low. This approach aims to help traders mitigate downside risk with timely alerts, leveraging objective technical signals for disciplined decision-making.
"""

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
    "APLAPOLLO.NS",        # APL Apollo Tubes Ltd
    "GARWAREHITECH.NS",    # Garware Hi-Tech Films Ltd
    "ABSLAMC.NS",          # Aditya Birla Sun Life AMC Ltd
    "TDPOWER.NS",          # TD Power Systems Ltd
    "JMFINANCIL.NS",       # JM Financial Ltd
    "WOCKPHARMA.NS",       # Wockhardt Ltd
    "KITEX.NS",            # Kitex Garments Ltd
    "CARERATING.NS",       # Care Ratings Ltd
    "MANAPPURAM.NS",       # Manappuram Finance Ltd
    "CHOLAFIN.NS",         # Cholamandalam Investment and Finance Company Ltd
    "HDFCLIFE.NS",         # HDFC Life Insurance Company Ltd
    "AUBANK.NS",           # AU Small Finance Bank Ltd
    "SUPREMEIND.NS"        # Supreme Industries Ltd
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

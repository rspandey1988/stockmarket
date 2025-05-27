"""
Weinstein-Based Weekly Trading Strategy
Overview
This strategy is a rules-based, technical indicator-driven approach designed for weekly trading of stocks, specifically applied to the Nifty 50 universe. It aims to identify high-probability buy and sell signals based on the combination of moving averages, trend slopes, and breakdown signals, with real-time alerts via Telegram.

Core Components
1. Trend Detection Using WMA and EMA

Weighted Moving Average (WMA):
Calculated over the last 30 weeks, giving more weight to recent prices to detect the underlying trend.  

EMA9 (Exponential Moving Average):
A short-term trend indicator that reacts faster to recent price changes, helping to confirm the trend direction.

2. Slope of WMA

WMA Slope:
The first difference of the WMA, indicating the trend's momentum. A positive slope suggests an uptrend; negative indicates a downtrend.

Trading Rules
Buy Signal Conditions
A buy is triggered when all of the following conditions are met:


Price above EMA9:
The current weekly closing price exceeds the short-term EMA, indicating upward momentum.



Price above WMA:
The weekly closing price is above the long-term trend indicator, reinforcing upward trend presence.



Positive WMA Slope:
The WMA is trending upward, suggesting increasing bullish momentum.



Action:  


Buy at current weekly close price if these conditions are satisfied, provided no current position exists.

Position sizing:
Allocate the maximum possible shares with the available cash.

Notification:  


Sends a Telegram message indicating a buy signal with the ticker and buy price.

Sell Signal Conditions
The exit is triggered under a breakdown scenario:


Tracking breakdown candle:
When in a position, if the weekly close drops below the EMA9, record the lowest low during this period as a breakdown candle.



Breakdown confirmation:
If, after a breakdown candle, the weekly close falls below this recorded low, it signals a bearish breakdown.



Action:  



Sell all held shares at the current weekly close price.

Record profit or loss.

Send a Telegram message indicating the sell with the ticker and sell price.



Final Position Closure:  


If at the end of the data, the position is still open, the system sells at the last available close price.

Additional Features

Breakdown Candle Tracking:
The lowest low during the period when the weekly close drops below EMA9 is used as a threshold. Falling below this indicates a bearish breakdown.



Progressive Exit Strategy:
The exit is triggered only after a confirmed breakdown below the recorded low, reducing false break signals.



Risk Management and Assumptions

Position sizing is based solely on available capital and share price.

The strategy relies on weekly data, making it suitable for longer-term traders.

Alerts via Telegram keep the trader informed about all buy and sell signals.

No explicit stop-loss or take-profit levels are implemented; the system relies on breakdown signals for exits.

Summary
This Weinstein-based weekly trading strategy combines trend-following indicators with a breakdown confirmation to identify high-probability entry and exit points. By focusing on the trend's momentum and sudden breakdowns, it aims to capture sustained upward moves while avoiding false signals. The use of Telegram alerts ensures timely execution and monitoring.

Would you like me to craft this into a formal documentation or presentation?


Regenerate
Copy message


"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

# --- Telegram setup ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send message: {e}")

# --- Nifty 50 list ---
nifty50_tickers = [
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

def compute_wma_and_slope(df):
    df['WMA'] = df['Close'].rolling(window=30).apply(
        lambda prices: np.dot(prices, np.arange(1, 31)) / np.arange(1, 31).sum(), raw=True
    )
    df['WMA_Slope'] = df['WMA'].diff(1)
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    return df

def detect_weinstein_signals(ticker, start, end, capital=100000):
    try:
        df = yf.download(ticker, start=start, end=end, interval='1wk', auto_adjust=True, progress=False)

        if df.empty or len(df) < 40:
            return {
                "Ticker": ticker,
                "Trades": 0,
                "Total Profit": 0,
                "CAGR (%)": 0,
                "Successful Trades": 0,
                "Trade Details": []
            }

        df = compute_wma_and_slope(df)

        in_position = False
        entry_price = 0
        trades = 0
        cash = capital
        shares = 0
        entry_date = None
        trade_details = []

        # For tracking breakdown candles
        breakdown_candle_low = None

        for i in range(30, len(df)):
            close = float(df["Close"].iloc[i])
            wma = float(df["WMA"].iloc[i])
            slope = float(df["WMA_Slope"].iloc[i])
            ema9 = float(df["EMA9"].iloc[i])
            low = float(df["Low"].iloc[i])  # Weekly low

            if pd.isna(close) or pd.isna(wma) or pd.isna(slope) or pd.isna(ema9):
                continue

            # --- Buy condition ---
            if not in_position and close > ema9 and close > wma and slope > 0:
                entry_price = close
                shares = cash // entry_price
                if shares > 0:
                    cash -= shares * entry_price
                    in_position = True
                    entry_date = df.index[i]
                    # Send buy alert
                    send_telegram_message(f"🟢 *BUY* {ticker} at {entry_price:.2f} on {entry_date.date()}")
                    # Reset breakdown candle info
                    breakdown_candle_low = None
                    current_trade = {
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": None,
                        "Exit Price": None,
                        "Profit": None
                    }

            # --- Track breakdown candle ---
            if in_position and close < ema9:
                if breakdown_candle_low is None:
                    breakdown_candle_low = low

            # --- Exit condition ---
            if in_position and breakdown_candle_low is not None:
                if close < breakdown_candle_low:
                    exit_price = close
                    cash += shares * exit_price
                    trades += 1
                    profit = (exit_price - entry_price) * shares
                    if profit > 0:
                        # Send sell alert
                        send_telegram_message(f"🔴 *SELL* {ticker} at {exit_price:.2f} on {df.index[i].date()}")
                        current_trade["Exit Date"] = df.index[i]
                        current_trade["Exit Price"] = exit_price
                        current_trade["Profit"] = profit
                        trade_details.append(current_trade)
                    in_position = False
                    shares = 0
                    breakdown_candle_low = None

        # Final sell if still holding position
        if in_position:
            final_price = float(df["Close"].iloc[-1])
            cash += shares * final_price
            trades += 1
            profit = (final_price - entry_price) * shares
            if profit > 0:
                # Send final sell alert
                send_telegram_message(f"🔴 *SELL* {ticker} at {final_price:.2f} on {df.index[-1].date()}")
                current_trade["Exit Date"] = df.index[-1]
                current_trade["Exit Price"] = final_price
                current_trade["Profit"] = profit
                trade_details.append(current_trade)

        total_profit = cash - capital
        successful_trades = len([t for t in trade_details if t['Profit'] and t['Profit'] > 0])
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((cash / capital) ** (1 / years) - 1) * 100

        return {
            "Ticker": ticker,
            "Trades": trades,
            "Total Profit": round(total_profit, 2),
            "CAGR (%)": round(cagr, 2),
            "Successful Trades": successful_trades,
            "Trade Details": trade_details
        }

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {
            "Ticker": ticker,
            "Trades": 0,
            "Total Profit": 0,
            "CAGR (%)": 0,
            "Successful Trades": 0,
            "Trade Details": []
        }

# --- Run from one month ago to today ---
end_date_dt = datetime.today()
start_date_dt = end_date_dt - timedelta(days=30)
start_date = start_date_dt.strftime("%Y-%m-%d")
end_date = end_date_dt.strftime("%Y-%m-%d")
initial_capital = 100000

results = []
for ticker in nifty50_tickers:
    print(f"Processing {ticker}...")
    result = detect_weinstein_signals(ticker, start=start_date, end=end_date, capital=initial_capital)
    results.append(result)

# Convert results to DataFrame
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="CAGR (%)", ascending=False)

# Print summary
print(df_result[['Ticker', 'Trades', 'Total Profit', 'CAGR (%)', 'Successful Trades']])

# Save detailed trade info for the top stock
top_stock = df_result.iloc[0]['Ticker']
top_stock_details = next(r for r in results if r['Ticker'] == top_stock)
trade_details_df = pd.DataFrame(top_stock_details['Trade Details'])
trade_details_df.to_csv(f"{top_stock}_profitable_trade_details.csv", index=False)

# Portfolio summary
total_profit = df_result["Total Profit"].sum()
final_value = initial_capital + total_profit
years_total = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
portfolio_cagr = ((final_value / initial_capital) ** (1 / years_total) - 1) * 100

print(f"\nTotal Portfolio Profit: ₹{round(total_profit, 2)}")
print(f"Portfolio CAGR: {round(portfolio_cagr, 2)}%")

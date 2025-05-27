import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- Telegram setup ---
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # Replace
CHAT_ID = 'YOUR_CHAT_ID'  # Replace

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send message: {e}")

# Full NSE/Nifty list (example)
nifty50_tickers = [
    # Your current list; I keep it short here for brevity
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"
]

# Initialize position status dict
position_status = {ticker: False for ticker in nifty50_tickers}

def compute_wma_and_slope(df):
    df['WMA'] = df['Close'].rolling(window=30).apply(
        lambda prices: np.dot(prices, np.arange(1, 31)) / np.arange(1, 31).sum(), raw=True
    )
    df['WMA_Slope'] = df['WMA'].diff(1)
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    return df

def detect_weinstein_signals(ticker, start, end, capital=100000, position_status=None):
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
        in_position = position_status.get(ticker, False)

        trades = 0
        cash = capital
        shares = 0
        entry_price = 0
        entry_date = None
        trade_details = []

        breakdown_candle_low = None

        for i in range(30, len(df)):
            close = float(df["Close"].iloc[i])
            wma = float(df["WMA"].iloc[i])
            slope = float(df["WMA_Slope"].iloc[i])
            ema9 = float(df["EMA9"].iloc[i])
            low = float(df["Low"].iloc[i])

            if pd.isna(close) or pd.isna(wma) or pd.isna(slope) or pd.isna(ema9):
                continue

            # --- BUY logic ---
            if not in_position and close > ema9 and close > wma and slope > 0:
                # Only buy if previously not in position
                if not position_status.get(ticker, False):
                    entry_price = close
                    shares = cash // entry_price
                    if shares > 0:
                        cash -= shares * entry_price
                        in_position = True
                        position_status[ticker] = True
                        entry_date = df.index[i]
                        send_telegram_message(f"🟢 *BUY* {ticker} at {entry_price:.2f} on {entry_date.date()}")
                        current_trade = {
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": None,
                            "Exit Price": None,
                            "Profit": None
                        }
                        breakdown_candle_low = None

            # --- Track breakdown candle ---
            if in_position and close < ema9:
                if breakdown_candle_low is None:
                    breakdown_candle_low = low

            # --- SELL logic ---
            if in_position and breakdown_candle_low is not None:
                if close < breakdown_candle_low:
                    exit_price = close
                    cash += shares * exit_price
                    trades += 1
                    profit = (exit_price - entry_price) * shares
                    # Only alert if profit
                    if profit > 0:
                        send_telegram_message(f"🔴 *SELL* {ticker} at {exit_price:.2f} on {df.index[i].date()}")
                        current_trade["Exit Date"] = df.index[i]
                        current_trade["Exit Price"] = exit_price
                        current_trade["Profit"] = profit
                        trade_details.append(current_trade)
                    in_position = False
                    position_status[ticker] = False
                    shares = 0
                    breakdown_candle_low = None

        # Final position check
        if in_position:
            final_price = float(df["Close"].iloc[-1])
            cash += shares * final_price
            trades += 1
            profit = (final_price - entry_price) * shares
            if profit > 0:
                send_telegram_message(f"🔴 *SELL* {ticker} at {final_price:.2f} on {df.index[-1].date()}")
                current_trade["Exit Date"] = df.index[-1]
                current_trade["Exit Price"] = final_price
                current_trade["Profit"] = profit
                trade_details.append(current_trade)
            in_position = False
            position_status[ticker] = False

        total_profit = cash - capital
        successful_trades = len([t for t in trade_details if t['Profit'] and t['Profit'] > 0])
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years > 0:
            cagr = ((cash / capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0

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

# --- Run for date starting 1 May 2025 ---
start_date = "2020-05-01"
end_date = datetime.today().strftime("%Y-%m-%d")
initial_capital = 100000

# Run analysis for all tickers
results = []
for ticker in nifty50_tickers:
    print(f"Processing {ticker}...")
    result = detect_weinstein_signals(ticker, start=start_date, end=end_date, capital=initial_capital, position_status=position_status)
    results.append(result)

# Summary output
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="CAGR (%)", ascending=False)

print(df_result[['Ticker', 'Trades', 'Total Profit', 'CAGR (%)', 'Successful Trades']])

# Save top stock trade details
top_stock = df_result.iloc[0]['Ticker']
top_stock_details = next(r for r in results if r['Ticker'] == top_stock)
trade_details_df = pd.DataFrame(top_stock_details['Trade Details'])
trade_details_df.to_csv(f"{top_stock}_trade_details.csv", index=False)

# Portfolio summary
total_profit = df_result["Total Profit"].sum()
final_value = initial_capital + total_profit
years_total = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
if years_total > 0:
    portfolio_cagr = ((final_value / initial_capital) ** (1 / years_total) - 1) * 100
else:
    portfolio_cagr = 0

print(f"\nTotal Portfolio Profit: ₹{round(total_profit, 2)}")
print(f"Portfolio CAGR: {round(portfolio_cagr, 2)}%")

"""
Processing RELIANCE.NS...
Processing TCS.NS...
Processing INFY.NS...
Processing HDFCBANK.NS...
        Ticker  Trades  Total Profit  CAGR (%)  Successful Trades
2      INFY.NS       1      56171.67      9.17                  1
0  RELIANCE.NS       1      51492.96      8.52                  1
1       TCS.NS       3      30547.59      5.39                  2
3  HDFCBANK.NS       8      -2423.87     -0.48                  3

Total Portfolio Profit: ₹135788.35
Portfolio CAGR: 18.42%
"""

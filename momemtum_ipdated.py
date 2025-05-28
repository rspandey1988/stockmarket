import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# Telegram token and chat id
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')  # Or get from env
CHAT_ID = os.getenv('CHAT_ID')

# Initialize trade log list
trade_log = []

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send message: {e}")

def log_trade(ticker, action, price, date, shares=0):
    # action: "BUY" or "SELL"
    trade = {
        'Ticker': ticker,
        'Action': action,
        'Price': price,
        'Date': date,
        'Shares': shares
    }
    trade_log.append(trade)

def compute_indicators(df):
    df['200EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['50EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['20EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['9EMA'] = df['Close'].ewm(span=9, adjust=False).mean()
    return df

def detect_signals(ticker, start, end, capital=100000):
    trades_this_stock = 0
    try:
        df = yf.download(ticker, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
        if df.empty or len(df) < 200:
            print(f"Data insufficient for {ticker}")
            return {"Ticker": ticker, "Trades": 0, "Total Profit": 0}

        df = compute_indicators(df)

        in_position = False
        cash = capital
        shares = 0
        entry_price = 0
        total_trades = 0
        breakdown_candle_low = None

        for i in range(200, len(df)):
            date_str = df.index[i].date().isoformat()
            close = float(df['Close'].iloc[i])
            dma_200 = float(df['200EMA'].iloc[i])
            dma_50 = float(df['50EMA'].iloc[i])
            dma_20 = float(df['20EMA'].iloc[i])
            dma_9 = float(df['9EMA'].iloc[i])
            ema_9 = float(df['9EMA'].iloc[i])
            low = float(df['Low'].iloc[i])

            # --- BUY logic ---
            if (not in_position and 
                not pd.isna(dma_200) and 
                not pd.isna(dma_50) and 
                not pd.isna(dma_20) and 
                not pd.isna(dma_9) and 
                not pd.isna(close)):
                if (close > dma_200 and close > dma_50 and close > dma_20 and close > dma_9):
                    entry_price = close
                    shares = cash // entry_price
                    if shares > 0:
                        cash -= shares * entry_price
                        in_position = True
                        send_telegram_message(f"🟢 BUY {ticker} at ${entry_price:.2f} on {date_str}")
                        log_trade(ticker, 'BUY', entry_price, date_str, shares)

            # --- Track breakdown candle ---
            if in_position and not pd.isna(ema_9):
                if close < ema_9:
                    if breakdown_candle_low is None:
                        breakdown_candle_low = low

            # --- SELL logic ---
            if in_position and breakdown_candle_low is not None:
                if not pd.isna(close) and not pd.isna(breakdown_candle_low):
                    if close < breakdown_candle_low:
                        exit_price = close
                        cash += shares * exit_price
                        total_trades += 1
                        profit = (exit_price - entry_price) * shares
                        if profit > 0:
                            send_telegram_message(f"🔴 SELL {ticker} at ${exit_price:.2f} on {date_str}")
                        log_trade(ticker, 'SELL', exit_price, date_str, shares)
                        in_position = False
                        shares = 0
                        breakdown_candle_low = None

        # Close position at end
        if in_position:
            final_price = float(df['Close'].iloc[-1])
            final_date = df.index[-1].date().isoformat()
            if not pd.isna(final_price):
                cash += shares * final_price
                total_trades += 1
                profit = (final_price - entry_price) * shares
                send_telegram_message(f"🔴 SELL (end) {ticker} at ${final_price:.2f} on {final_date}")
                log_trade(ticker, 'SELL', final_price, final_date, shares)
                in_position = False

        total_profit = cash - capital
        return {"Ticker": ticker, "Trades": total_trades, "Total Profit": round(total_profit, 2)}

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {"Ticker": ticker, "Trades": 0, "Total Profit": 0}

# --- Main ---
start_date = "2010-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
initial_capital = 100000

# List of stocks
# Nifty 50 stock list
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "ITC.NS", "BHARTIARTL.NS", "HCLTECH.NS", "ASIANPAINT.NS", "LT.NS",
    "MARUTI.NS", "HDFCLIFE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "POWERGRID.NS", "BAJAJFINSV.NS",
    "TATASTEEL.NS", "ADANIGREEN.NS", "NTPC.NS", "JSWSTEEL.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "CIPLA.NS",
    "INDUSINDBK.NS", "EICHERMOT.NS", "TECHM.NS", "TATAMOTORS.NS", "GRASIM.NS",
    "BPCL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "HINDALCO.NS", "SBILIFE.NS",
    "SHREECEM.NS", "UPL.NS", "VEDL.NS", "M&M.NS", "IOC.NS"
]

# Run the backtest
results = []
for ticker in nifty50_tickers:
    print(f"Processing {ticker}...")
    result = detect_signals(ticker, start=start_date, end=end_date, capital=initial_capital)
    results.append(result)

# Save trade log to CSV
trades_df = pd.DataFrame(trade_log)
trades_df.to_csv("trade_log.csv", index=False)

# Summary
df_results = pd.DataFrame(results)
print(df_results[['Ticker', 'Trades', 'Total Profit']])

total_profit = df_results["Total Profit"].sum()
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
Processing ICICIBANK.NS...
Processing HINDUNILVR.NS...
Processing KOTAKBANK.NS...
Processing SBIN.NS...
Processing AXISBANK.NS...
Processing BAJFINANCE.NS...
Processing ITC.NS...
Processing BHARTIARTL.NS...
Processing HCLTECH.NS...
Processing ASIANPAINT.NS...
Processing LT.NS...
Processing MARUTI.NS...
Processing HDFCLIFE.NS...
Processing WIPRO.NS...
Processing ULTRACEMCO.NS...
Processing ONGC.NS...
Processing TITAN.NS...
Processing SUNPHARMA.NS...
Processing NESTLEIND.NS...
Processing POWERGRID.NS...
Processing BAJAJFINSV.NS...
Processing TATASTEEL.NS...
Processing ADANIGREEN.NS...
Processing NTPC.NS...
Processing JSWSTEEL.NS...
Processing DIVISLAB.NS...
Processing DRREDDY.NS...
Processing HEROMOTOCO.NS...
Processing BAJAJ-AUTO.NS...
Processing BRITANNIA.NS...
Processing CIPLA.NS...
Processing INDUSINDBK.NS...
Processing EICHERMOT.NS...
Processing TECHM.NS...
Processing TATAMOTORS.NS...
Processing GRASIM.NS...
Processing BPCL.NS...
Processing ADANIPORTS.NS...
Processing COALINDIA.NS...
Processing HINDALCO.NS...
Processing SBILIFE.NS...
Processing SHREECEM.NS...
Processing UPL.NS...
Processing VEDL.NS...
Processing M&M.NS...
Processing IOC.NS...
           Ticker  Trades  Total Profit
0     RELIANCE.NS      32     278296.15
1          TCS.NS      17     561284.34
2         INFY.NS      21     149985.25
3     HDFCBANK.NS       8     578164.48
4    ICICIBANK.NS      76      28996.16
5   HINDUNILVR.NS       9     707037.26
6    KOTAKBANK.NS      10     598083.44
7         SBIN.NS      24     148465.67
8     AXISBANK.NS      21     196560.36
9   BAJFINANCE.NS      13   10336620.96
10         ITC.NS       7     388572.98
11  BHARTIARTL.NS      27     204435.39
12     HCLTECH.NS      12    1484435.04
13  ASIANPAINT.NS       5     898633.69
14          LT.NS      19     351688.38
15      MARUTI.NS      12     831901.03
16    HDFCLIFE.NS       6      26466.78
17       WIPRO.NS      13     150752.51
18  ULTRACEMCO.NS       8     802326.85
19        ONGC.NS      35      48194.60
20       TITAN.NS       7    1447190.50
21   SUNPHARMA.NS       2     716542.08
22   NESTLEIND.NS       4     746799.16
23   POWERGRID.NS      10     560699.27
24  BAJAJFINSV.NS       8    2685032.64
25   TATASTEEL.NS      35     497014.37
26  ADANIGREEN.NS       5    1798390.96
27        NTPC.NS      58      63877.21
28    JSWSTEEL.NS      21     992841.19
29    DIVISLAB.NS       3    1983123.66
30     DRREDDY.NS      10     255789.03
31  HEROMOTOCO.NS      17     152465.57
32  BAJAJ-AUTO.NS       8     590644.27
33   BRITANNIA.NS      10    2659268.05
34       CIPLA.NS      11     325009.43
35  INDUSINDBK.NS       9     212106.44
36   EICHERMOT.NS      11    4804479.28
37       TECHM.NS       7     938366.13
38  TATAMOTORS.NS      24     173746.14
39      GRASIM.NS      10     603320.15
40        BPCL.NS      20     601857.70
41  ADANIPORTS.NS      21     269348.67
42   COALINDIA.NS      27      79210.26
43    HINDALCO.NS      16     384079.42
44     SBILIFE.NS      13      84723.44
45    SHREECEM.NS       5    1603440.84
46         UPL.NS      18     186200.01
47        VEDL.NS      11     380296.91
48         M&M.NS      11     407881.87
49         IOC.NS      16     478957.90

Total Portfolio Profit: ₹45453603.87
Portfolio CAGR: 48.8%

"""

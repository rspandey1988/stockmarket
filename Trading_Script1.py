"""
Weinstein Weekly Trading Strategy Backtest on Nifty 50 Stocks

This script implements a weekly trading strategy inspired by Weinstein's methodology,
applied to the top 10 Nifty 50 stocks. It downloads weekly OHLC data from Yahoo Finance
and uses technical indicators such as Weighted Moving Average (WMA), its slope, and
Exponential Moving Average (EMA) to generate buy and sell signals.

Key features:
- Calculates 30-week WMA, its slope, and 9-week EMA for trend detection.
- Enters a position when the price crosses above WMA with a positive slope.
- Tracks breakdown candles when price drops below the 9-week EMA during a position.
- Sells when the price drops below the recorded breakdown candle low.
- Closes open positions at the end of the data period.
- Records detailed trade data (entry/exit dates, prices, profit) for all trades.
- Computes overall performance metrics including total profit, CAGR, and number of trades.
- Processes multiple stocks and outputs a sorted summary based on CAGR.
- Saves detailed trade history for the top-performing stock.
- Plots portfolio value over the years to visualize growth.
- Displays final total portfolio value and CAGR.

Usage:
- Ensure `yfinance`, `pandas`, `numpy`, and `matplotlib` packages are installed.
- Adjust the `nifty50_tickers` list if needed.
- Set the desired start and end date for backtesting.
- Run the script; results will be printed, saved as CSV, and plots will be shown.

Note:
- The script logs all trades (both profitable and unprofitable) for comprehensive analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# List of top 10 Nifty 50 tickers
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "LT.NS"
]

def compute_wma_and_slope(df):
    """
    Computes the 30-week Weighted Moving Average (WMA), its slope, and the 9-week EMA.
    Args:
        df (DataFrame): DataFrame containing 'Close' prices.
    Returns:
        DataFrame: with added 'WMA', 'WMA_Slope', and 'EMA9' columns.
    """
    df['WMA'] = df['Close'].rolling(window=30).apply(
        lambda prices: np.dot(prices, np.arange(1, 31)) / np.arange(1, 31).sum(), raw=True
    )
    df['WMA_Slope'] = df['WMA'].diff(1)
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    return df

def detect_weinstein_signals(ticker, start, end, capital=100000):
    """
    Performs backtest for a single stock using Weinstein's weekly strategy.
    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.
        capital (float): Initial capital for trading.
    Returns:
        dict: Contains overall stats, detailed trade data, and portfolio value timeline.
    """
    try:
        df = yf.download(ticker, start=start, end=end, interval='1wk', auto_adjust=True, progress=False)

        if df.empty or len(df) < 40:
            return {
                "Ticker": ticker,
                "Trades": 0,
                "Total Profit": 0,
                "CAGR (%)": 0,
                "Successful Trades": 0,
                "Trade Details": [],
                "Portfolio Timeline": []
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

        # To record portfolio value over time
        portfolio_timeline = []

        for i in range(30, len(df)):
            close = float(df["Close"].iloc[i])
            wma = float(df["WMA"].iloc[i])
            slope = float(df["WMA_Slope"].iloc[i])
            ema9 = float(df["EMA9"].iloc[i])
            low = float(df["Low"].iloc[i])  # Weekly low

            if pd.isna(close) or pd.isna(wma) or pd.isna(slope) or pd.isna(ema9):
                continue

            # Buy condition
            if not in_position and close > wma and slope > 0:
                entry_price = close
                shares = cash // entry_price
                if shares > 0:
                    cash -= shares * entry_price
                    in_position = True
                    entry_date = df.index[i]
                    # Reset breakdown candle info
                    breakdown_candle_low = None
                    current_trade = {
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": None,
                        "Exit Price": None,
                        "Profit": None
                    }

            # Track breakdown candle if price drops below EMA9 during position
            if in_position and close < ema9:
                if breakdown_candle_low is None:
                    breakdown_candle_low = low

            # Check for exit condition
            if in_position and breakdown_candle_low is not None:
                if close < breakdown_candle_low:
                    exit_price = close
                    cash += shares * exit_price
                    trades += 1
                    profit = (exit_price - entry_price) * shares
                    # Log all trades
                    current_trade["Exit Date"] = df.index[i]
                    current_trade["Exit Price"] = exit_price
                    current_trade["Profit"] = profit
                    trade_details.append(current_trade)
                    # Reset position and breakdown candle
                    in_position = False
                    shares = 0
                    breakdown_candle_low = None

            # Record portfolio value at each step
            total_value = cash + (shares * close) if in_position else cash
            portfolio_timeline.append({
                'Date': df.index[i],
                'Portfolio Value': total_value
            })

        # Close open position at the end
        if in_position:
            final_price = float(df["Close"].iloc[-1])
            cash += shares * final_price
            trades += 1
            profit = (final_price - entry_price) * shares
            # Log all trades
            current_trade["Exit Date"] = df.index[-1]
            current_trade["Exit Price"] = final_price
            current_trade["Profit"] = profit
            trade_details.append(current_trade)
            total_value = cash

        total_profit = total_value - capital
        successful_trades = len([t for t in trade_details if t["Profit"] > 0])
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((total_value / capital) ** (1 / years) - 1) * 100

        return {
            "Ticker": ticker,
            "Trades": trades,
            "Total Profit": round(total_profit, 2),
            "CAGR (%)": round(cagr, 2),
            "Successful Trades": successful_trades,
            "Trade Details": trade_details,
            "Portfolio Timeline": portfolio_timeline
        }

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {
            "Ticker": ticker,
            "Trades": 0,
            "Total Profit": 0,
            "CAGR (%)": 0,
            "Successful Trades": 0,
            "Trade Details": [],
            "Portfolio Timeline": []
        }

# Run strategy for selected stocks
start_date = "2010-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
initial_capital = 100000

results = []
for ticker in nifty50_tickers:
    print(f"Processing {ticker}...")
    result = detect_weinstein_signals(ticker, start=start_date, end=end_date, capital=initial_capital)
    results.append(result)

# Convert results to DataFrame and sort by CAGR
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="CAGR (%)", ascending=False)

# Display summary
print(df_result[['Ticker', 'Trades', 'Total Profit', 'CAGR (%)', 'Successful Trades']])

# Save detailed trade logs for the top stock
top_stock = df_result.iloc[0]['Ticker']
top_stock_data = next(r for r in results if r['Ticker'] == top_stock)
trade_details_df = pd.DataFrame(top_stock_data['Trade Details'])
trade_details_df.to_csv(f"{top_stock}_all_trade_details.csv", index=False)

# --- Display total portfolio value and CAGR ---
final_value = top_stock_data['Portfolio Timeline'][-1]['Portfolio Value']
cagr_percent = top_stock_data['CAGR (%)']

print(f"\nTotal Portfolio Value at End: ₹{final_value:.2f}")
print(f"Portfolio CAGR: {cagr_percent:.2f}%")

# Prepare data for yearly growth plot
portfolio_timeline = top_stock_data['Portfolio Timeline']
dates = [entry['Date'] for entry in portfolio_timeline]
values = [entry['Portfolio Value'] for entry in portfolio_timeline]

# Convert dates to pandas datetime for grouping
df_portfolio = pd.DataFrame({'Date': dates, 'Value': values})
df_portfolio['Year'] = pd.to_datetime(df_portfolio['Date']).dt.year

# Get the last portfolio value for each year
year_end_values = df_portfolio.groupby('Year').last()

# Plot yearly growth
plt.figure(figsize=(12, 6))
plt.plot(year_end_values.index, year_end_values['Value'], marker='o', linestyle='-')
plt.title(f"Yearly Portfolio Growth for {top_stock}")
plt.xlabel("Year")
plt.ylabel("Portfolio Value (₹)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(year_end_values.index)
plt.tight_layout()
plt.show()



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
Processing LT.NS...
          Ticker  Trades  Total Profit  CAGR (%)  Successful Trades
5  HINDUNILVR.NS       1    1106929.17     17.54                  1
1         TCS.NS       2     795854.69     15.29                  2
6   KOTAKBANK.NS       5     795577.96     15.29                  3
4   ICICIBANK.NS       7     776661.50     15.13                  4
3    HDFCBANK.NS      15     399141.85     11.00                  4
0    RELIANCE.NS       7     380730.91     10.73                  2
9          LT.NS       7     351502.33     10.28                  3
2        INFY.NS      13     160999.70      6.42                  4
7        SBIN.NS      21     112711.63      5.02                  6
8    AXISBANK.NS      17      25758.26      1.50                  5

Total Portfolio Value at End: ₹1206929.17
Portfolio CAGR: 17.54%


"""

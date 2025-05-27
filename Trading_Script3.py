import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Full Nifty 50 stock list
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

# Calculate 30-week WMA, slope, and 9-week EMA
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

            # BUY CONDITION:
            # 1. Close > WEMA (EMA9)
            # 2. Close > WMA
            # 3. Slope > 0
            if not in_position and close > ema9 and close > wma and slope > 0:
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

            # Track breakdown candle if close below 9 WEMA
            if in_position and close < ema9:
                if breakdown_candle_low is None:
                    breakdown_candle_low = low

            # Exit condition:
            # When a breakdown candle was identified AND next candle closes below that candle's low
            if in_position and breakdown_candle_low is not None:
                if close < breakdown_candle_low:
                    exit_price = close
                    cash += shares * exit_price
                    trades += 1
                    profit = (exit_price - entry_price) * shares
                    if profit > 0:
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
                current_trade["Exit Date"] = df.index[-1]
                current_trade["Exit Price"] = final_price
                current_trade["Profit"] = profit
                trade_details.append(current_trade)

        total_profit = cash - capital
        successful_trades = len(trade_details)
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

# Run strategy for all Nifty 50 stocks
start_date = "2010-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
initial_capital = 100000

results = []
for ticker in nifty50_tickers:
    print(f"Processing {ticker}...")
    result = detect_weinstein_signals(ticker, start=start_date, end=end_date, capital=initial_capital)
    results.append(result)

# Convert results to DataFrame
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="CAGR (%)", ascending=False)

# Display summary
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
Processing HDFC.NS...

1 Failed download:
['HDFC.NS']: YFTzMissingError('possibly delisted; no timezone found')
Processing MARUTI.NS...
Processing BAJFINANCE.NS...
Processing HCLTECH.NS...
Processing WIPRO.NS...
Processing ADANIPORTS.NS...
Processing TITAN.NS...
Processing NESTLEIND.NS...
Processing TATASTEEL.NS...
Processing DIVISLAB.NS...
Processing M&M.NS...
Processing DRREDDY.NS...
Processing BAJAJ-AUTO.NS...
Processing HEROMOTOCO.NS...
Processing CIPLA.NS...
Processing EICHERMOT.NS...
Processing SHREECEM.NS...
Processing TATAMOTORS.NS...
Processing ULTRACEMCO.NS...
Processing JSWSTEEL.NS...
Processing HDFCLIFE.NS...
Processing SBILIFE.NS...
Processing INDUSINDBK.NS...
Processing BPCL.NS...
Processing IOCL.NS...

1 Failed download:
['IOCL.NS']: HTTPError('HTTP Error 404: ')
Processing POWERGRID.NS...
Processing TATACONSUM.NS...
Processing ADANIGREEN.NS...
Processing APOLLOHOSP.NS...
Processing BRITANNIA.NS...
Processing HINDALCO.NS...
Processing NESTLEIND.NS...
Processing TATAPOWER.NS...
Processing UPL.NS...
Processing VEDL.NS...
Processing COALINDIA.NS...
           Ticker  Trades  Total Profit  CAGR (%)  Successful Trades
37  ADANIGREEN.NS       2    1889567.60     53.89                  1
12  BAJFINANCE.NS       5   14142595.05     37.96                  2
25   EICHERMOT.NS       3    4912240.22     28.92                  2
39   BRITANNIA.NS       8    3159478.78     25.37                  5
13     HCLTECH.NS       5    2031227.70     21.96                  4
16       TITAN.NS       5    1964081.64     21.71                  4
38  APOLLOHOSP.NS       1    1817205.41     21.13                  1
19    DIVISLAB.NS       4    1762140.23     20.90                  2
29    JSWSTEEL.NS       6    1412446.41     19.28                  4
26    SHREECEM.NS       3    1323989.89     18.81                  1
5   HINDUNILVR.NS       2     954667.26     16.52                  2
36  TATACONSUM.NS       4     942833.31     16.43                  2
28  ULTRACEMCO.NS       3     928696.32     16.33                  1
41   NESTLEIND.NS       1     874708.55     15.92                  1
17   NESTLEIND.NS       1     874708.55     15.92                  1
11      MARUTI.NS       7     844785.97     15.69                  5
1          TCS.NS       2     795854.80     15.29                  2
6    KOTAKBANK.NS       5     795577.97     15.29                  3
4    ICICIBANK.NS       7     776661.50     15.13                  4
18   TATASTEEL.NS      12     738696.56     14.80                  3
22  BAJAJ-AUTO.NS       3     671142.87     14.18                  2
33        BPCL.NS       9     618346.17     13.65                  4
15  ADANIPORTS.NS       5     603563.88     13.50                  2
40    HINDALCO.NS      12     511386.67     12.47                  6
44        VEDL.NS      11     479287.10     12.08                  3
3     HDFCBANK.NS      13     425938.57     11.38                  4
35   POWERGRID.NS       9     405463.67     11.09                  3
20         M&M.NS       4     398737.50     10.99                  1
43         UPL.NS       9     386654.03     10.82                  3
9           LT.NS       7     344622.45     10.17                  3
24       CIPLA.NS       4     338381.89     10.07                  1
27  TATAMOTORS.NS       6     319954.19      9.76                  2
0     RELIANCE.NS       9     314001.15      9.66                  3
30    HDFCLIFE.NS       2      87258.15      8.69                  1
31     SBILIFE.NS       7      88363.59      8.63                  2
21     DRREDDY.NS       6     258136.22      8.63                  3
2         INFY.NS      13     160999.57      6.42                  4
42   TATAPOWER.NS      19     147374.92      6.05                  3
14       WIPRO.NS      15     130142.36      5.56                  7
7         SBIN.NS      21     107961.67      4.87                  6
23  HEROMOTOCO.NS      19      46483.06      2.51                  7
8     AXISBANK.NS      17      25758.25      1.50                  5
45   COALINDIA.NS      28      24022.83      1.49                  8
32  INDUSINDBK.NS      12       5544.17      0.35                  5
34        IOCL.NS       0          0.00      0.00                  0
10        HDFC.NS       0          0.00      0.00                  0

Total Portfolio Profit: ₹49841688.65
Portfolio CAGR: 49.69%
"""


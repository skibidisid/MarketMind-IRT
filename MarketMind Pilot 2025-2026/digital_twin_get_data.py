# Imports 
import pandas as pd
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import numpy as np

# Main method
def main ():
    
    # API Key
    fred_key = "9b177a5e71c8d0c14448d99c238cd052"
    Ticker = "SPY"

    # Getting close/volume for etf
    closevolume_df = get_etf_prices(Ticker)
    closevolume_df_selective = closevolume_df[["Close", "Volume"]] 

    # Volatility ETF prices
    vol_df = get_etf_prices("^VIX")
    vol_df = vol_df[["Close"]]
    vol_df.rename(columns={"Close": "VIX_Close"}, inplace=True)

    # Getting treasury yields
    treasury_yields = fred(fred_key, "DGS10")

    # Getting cpi data
    cpi_data = fred(fred_key, "CPIAUCSL")

    # Getting corporate bond data
    corporate_bond_data = fred(fred_key, "AAA")

    # Getting sentiment data - ONLY MONTHLY MAKE DAILY LATER
    consumer_sentiment = fred(fred_key, "UMCSENT")

    # Calculating dividend yield feature
    dividend_yield_feature = calculate_dividend_yield_feature(Ticker, "2014-01-01")

    # Getting technical indicators
    tech_df = get_indicators(closevolume_df)
    
    '''plt.figure(figsize = (12,6))
    plt.plot(tech_df["RSI14"], "r")
    plt.plot(tech_df["MACD"], "g")
    plt.plot(tech_df["Signal_Line"], "b")
    plt.plot(tech_df["MACD_Histogram"], "y")

    plt.show()
'''

    # Merging all dataframes
    final_df = merge_dataframes(closevolume_df_selective, treasury_yields, cpi_data, corporate_bond_data, consumer_sentiment, dividend_yield_feature, tech_df, vol_df)
    
    # Final filtering to start from 2015-01-01
    start_date_filter = '2015-01-01'
    final_df = final_df.loc[final_df.index >= start_date_filter]

    print("\n--- Final Cleaned DataFrame Head (Ready for Modeling) ---")
    print(final_df.head())

    final_df.to_csv("final_etf_data.csv")

# Prices & Volume method
def get_etf_prices (Ticker):

    # Start & end date
    start_date = "2014-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Downloading data
    df = yf.download(Ticker, start=start_date, end=end_date, progress=False)

    # Flatten collumns 
    if (isinstance(df.columns, pd.MultiIndex)):
        # Create a clean list of columns, joining the level 0 (Price) and level 1 (Ticker)
        df.columns = df.columns.get_level_values(0) 
        # Alternatively: df.columns = [col[0] for col in df.columns]
    
    # Test print
    print(df.head())

    # Return
    return df

# FRED data collection method
def fred (api_key, type_of_data):

    # Getting 10 year treasury yields
    fred_df = pdr.DataReader(type_of_data, "fred", start="2014-01-01", api_key=api_key)

    fred_df.index.name = "Date"

    # Test print
    print(fred_df.head())

    # Return
    return fred_df

def calculate_dividend_yield_feature(Ticker, start_date):

    # Creating ticker object to get dividends
    ticker_obj = yf.Ticker(Ticker)
    dividends = ticker_obj.dividends

    # Calculate Dividend Amount (Rolling 365-day sum)
    ttm_dividend_amount = dividends.resample('D').sum().rolling(window='365D').sum()
    ttm_dividend_amount = ttm_dividend_amount.dropna()
    ttm_dividend_amount.name = "TTM_Dividend_Amount"
    
    # Forward fill to allign and dropping NaNs
    ttm_df = ttm_dividend_amount.to_frame()
    ttm_df["TTM_Dividend_Amount"].fillna(method="ffill", inplace=True)

    # Timezone Cleanup FIRST (before reindex to match date_range)
    if isinstance(ttm_df.index, pd.DatetimeIndex) and ttm_df.index.tz is not None:
        ttm_df.index = ttm_df.index.tz_localize(None)
    print(ttm_df.tail())

    # Get the current end date of your main data fetch
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    # Use pd.to_datetime to ensure start and end are consistent datetime objects
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))

    # Re-index ttm_df to this full date range, which introduces NaNs at the end
    ttm_df = ttm_df.reindex(date_range)
    ttm_df.index.name = "Date"

    # Final FFILL to carry the last known TTM value to the end date
    ttm_df["TTM_Dividend_Amount"] = ttm_df["TTM_Dividend_Amount"].ffill()

    # Test print/return
    print(ttm_df.tail())
    return ttm_df

def get_indicators(df):

    # Copying dataframe
    data = df.copy()

    # RSI:
    # Change/gain/loss calculation
    change = data["Close"].diff()
    gain = change.where(change > 0, 0).rolling(window=14).mean()
    loss = -change.where(change < 0, 0).rolling(window=14).mean()

    # RSI calculation
    rs = gain / loss
    data["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    # Calculate the 12-period EMA
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()

    # Calculate the 26-period EMA
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
    data["MACD"] = data["EMA12"] - data["EMA26"]

    # Calculate the 9-period EMA of MACD (Signal Line)
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # MACD Histogram
    data["MACD_Histogram"] = data["MACD"] - data["Signal_Line"]

    # Dropping NaN values
    data = data.dropna()

    data = data[["RSI14", "MACD", "Signal_Line", "MACD_Histogram"]]

    print(data.head())

    return data

# Merging dataframes
def merge_dataframes(closevolume_df_selective, treasury_yields, cpi_data, corporate_bond_data, consumer_sentiment, dividend_yield_feature, tech_df, vol_df):

    final_df = closevolume_df_selective.copy()

    final_df = final_df.join(treasury_yields, how = "left")
    final_df = final_df.join(cpi_data, how = "left")
    final_df = final_df.join(corporate_bond_data, how = "left")
    final_df = final_df.join(consumer_sentiment, how = "left")
    final_df = final_df.join(dividend_yield_feature, how = "left")
    final_df = final_df.join(tech_df, how = "left")
    final_df = final_df.join(vol_df, how = "left")

    final_df["UMCSENT"].fillna(method='ffill', inplace=True)
    final_df["DGS10"].fillna(method='ffill', inplace=True)
    final_df["CPIAUCSL"].fillna(method='ffill', inplace=True)
    final_df["AAA"].fillna(method='ffill', inplace=True)

    # Momentum: Is the price accelerating?
    final_df['Momentum_5d'] = final_df['Close'].pct_change(5)
    final_df['Momentum_20d'] = final_df['Close'].pct_change(20)

    # Volatility: How risky is the current move?
    final_df['Volatility_20d'] = final_df['Close'].pct_change().rolling(20).std()
    
    # Relative Strength Index Change (Detecting trend reversals)
    final_df['RSI_Velocity'] = final_df['RSI14'].diff()

    # Target: Predicting Log Returns (Better for Ensembles than Pct Change)
    final_df['Target_Returns'] = np.log(final_df['Close'] / final_df['Close'].shift(1)).shift(-1)

    return final_df

# Call main method
main()
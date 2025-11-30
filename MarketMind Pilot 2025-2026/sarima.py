import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pmd

# Main method
def main():
    # Setting ticker
    ticker = "VDE"

    # Creating csv file name
    csvFile = ticker + "_data.csv"

    # Getting data and putting into csv file
    get_data(ticker, csvFile)

    # Reading the data in the csv file
    df = read_csv(csvFile)

    # Grabbing just the close data
    closeData = getCloseData(df)

    # Running SARIMA forecasting
    run_sarima(closeData, ticker)

# Method to get data
def get_data(ticker, csvFile):
    # Downloading data as a data frame and converting into a csv
    data = yf.download(ticker, start="2005-01-01", end="2025-11-19")
    data.to_csv(csvFile)

# Reading csv into DataFrame
def read_csv(csvFile):
    if os.path.exists(csvFile):
        df = pd.read_csv(csvFile, parse_dates=["Date"], index_col="Date")
        return df
    else:
        print("File " + csvFile + " does not exist.")
        return None

# Extracting just the close data
def getCloseData(df):
    if df is not None:
        closeData = df["Close"].dropna()
        return closeData
    else:
        return []

# Running SARIMA forecasting
def run_sarima(close, ticker):
    # Plot the raw closing price
    plt.plot(close)
    plt.title(ticker + " Close Price")
    plt.show()

    # ACF plot
    sm.graphics.tsa.plot_acf(close, lags=40)
    plt.show()

    # PACF plot
    sm.graphics.tsa.plot_pacf(close, lags=40)
    plt.show()

    # Decompose into trend, seasonality, residuals
    decomposition = seasonal_decompose(close, model="additive", period=12)
    decomposition.plot()
    plt.show()

    # ADF test to check stationarity
    adf = adfuller(close)
    print("ADF Statistic:", adf[0])
    print("p-value:", adf[1])

    # Auto ARIMA selects p,d,q and P,D,Q automatically
    model = pmd.auto_arima(
        close,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        m=12,                    # monthly-ish seasonal period
        seasonal=True,
        test="adf",
        trace=True,
        error_action="ignore",
        suppress_warnings=True
    )

    # Extract best model parameters
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order

    # Fit SARIMA model
    sarima = sm.tsa.statespace.SARIMAX(
        close,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = sarima.fit()

    # Display model summary
    print(results.summary())

    # Forecast next 30 days
    forecast_steps = 30
    forecast = results.get_forecast(forecast_steps)
    forecast_values = forecast.predicted_mean
    conf = forecast.conf_int()

    print("\nNext 30-Day Forecast:")
    print(forecast_values)

    # Plot actual vs forecast
    plt.plot(close[-200:], label="Actual")
    plt.plot(forecast_values, label="Forecast")
    plt.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], alpha=0.2)
    plt.legend()
    plt.title(ticker + " SARIMA Forecast")
    plt.show()

# Calling main
main()

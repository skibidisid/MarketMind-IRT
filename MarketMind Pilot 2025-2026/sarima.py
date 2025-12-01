import yfinance as yf
import pandas as pd
import os
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


# Main method
def main():

    # Ticker to analyze
    ticker = "VDE"

    # CSV file
    csvFile = ticker + "_data.csv"

    # Getting price data
    get_data(ticker, csvFile)

    # Reading file
    csvFileData = read_csv(csvFile)

    # Extract close-only list
    closeData = getCloseData(csvFileData)

    # Convert to pandas series for SARIMA
    closeSeries = pd.Series(closeData)

    # Fit SARIMA + forecast + plot
    runSARIMA(closeSeries, ticker)

# Download price data into CSV
def get_data(ticker, csvFile):

    data = yf.download(ticker, start="2000-11-19", end="2025-11-19")
    data.to_csv(csvFile)

# Read CSV file
def read_csv(csvFile):

    if os.path.exists(csvFile):
        fo = open(csvFile, "r")
        csvFileData = fo.read().split("\n")
    else:
        print("File " + csvFile + " does not exist.")

    return csvFileData

# Pull only close prices
def getCloseData(csvFileData):

    closeData = []

    for i in range(3, len(csvFileData) - 1):

        tempList = csvFileData[i].split(",")

        # still using index 1 to match your format
        closeData.append(float(tempList[1]))

    return closeData

# SARIMA forecasting + plot
def runSARIMA(closeSeries, ticker):

    # SARIMA order
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # Fit model
    model = SARIMAX(closeSeries, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Forecast 30 days ahead
    forecast = results.forecast(30)

    # Print results
    print("\n=== SARIMA MODEL RESULTS FOR", ticker, "===")
    print("Last Close Price:", closeSeries.iloc[-1])
    print("\nNext 30-Day Forecast (first 5 shown):")
    print(forecast[:5])
    print("\nModel AIC:", results.aic)
    print("Model BIC:", results.bic)
    print("Model HQIC:", results.hqic)

    # ---- Plot ---- #
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(closeSeries, label="Historical Data")

    # Plot forecast (aligned after last index)
    forecast_index = range(len(closeSeries), len(closeSeries) + len(forecast))
    plt.plot(forecast_index, forecast, label="SARIMA 30-Day Forecast")

    # Visual marker at boundary
    plt.axvline(x=len(closeSeries)-1, color="gray", linestyle="--", alpha=0.6)

    # Labels & title
    plt.title(f"SARIMA Forecast for {ticker}")
    plt.xlabel("Time (Index)")
    plt.ylabel("Price")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # use last 30 observations as test
    n_test = 30
    train = closeSeries[:-n_test]
    test = closeSeries[-n_test:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # fit on training data only
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # forecast exactly len(test) steps ahead
    forecast = results.forecast(steps=n_test)

    # MAE between forecast and actual test data
    mae = mean_absolute_error(test, forecast)
    print("MAE on last 30 days:", mae)

    # Plot actual vs predicted on test window
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual (Test)", color="blue")
    plt.plot(forecast.index, forecast, label="Predicted", color="red")
    plt.axvline(x=test.index[0], color="gray", linestyle="--", alpha=0.6)
    plt.title(f"SARIMA Actual vs Predicted for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()



# Call main
main()

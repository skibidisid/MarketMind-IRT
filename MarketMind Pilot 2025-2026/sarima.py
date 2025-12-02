import yfinance as yf # yfinance - to download stock price data
import pandas as pd # pandas - for handling data in tables (DataFrames) or series
import os # os functions like checking if files exist
import numpy as np # numpy - numerical operations on arrays
import math # math - standard math functions (not used much here)
import warnings # warnings - to control warnings in python
import matplotlib.pyplot as plt # matplotlib - library used to plot data in graphs

# just hides the warnings in terminal so they dont clutter output// trivial warnings
warnings.filterwarnings("ignore")

# SARIMAX builds seasonal ARIMA models for forecasting tiem series data
from statsmodels.tsa.statespace.sarimax import SARIMAX

# the mean absolute error computes the average of the difference/residuals between actual and predicted values
from sklearn.metrics import mean_absolute_error


# Main method
def main():

    # Ticker to analyze
    ticker = "VDE"

    # CSV file to save downloaded data
    ## a CSV file is a comma separated value file; a spreadsheet basically
    csvFile = ticker + "_data.csv"

    # Getting price data and saves to CSV
    get_data(ticker, csvFile)

    # Reading file
    csvFileData = read_csv(csvFile)

    # Extract only the closing prices of the stocks
    closeData = getCloseData(csvFileData)

    # Convert to pandas series; required for SARIMAX
    closeSeries = pd.Series(closeData)

    # Fit SARIMA + forecast + plot
    runSARIMA(closeSeries, ticker)

# Download price data from yahoo finance into CSV
def get_data(ticker, csvFile):

    data = yf.download(ticker, start="2000-11-19", end="2025-11-19")
    data.to_csv(csvFile)

# Read CSV file
def read_csv(csvFile):
    # open the file and read all content as a single string. split it into lines and return a list of lines
    if os.path.exists(csvFile):
        fo = open(csvFile, "r")
        csvFileData = fo.read().split("\n")
    else:
        print("File " + csvFile + " does not exist.")

    # return the list of lines
    return csvFileData

# Pull only close prices
def getCloseData(csvFileData):

    # initialize empty list to store the closing prices
    closeData = []

    # loop through the CSV lines (start from line 3 to skip headers)
    for i in range(3, len(csvFileData) - 1):
        # split each line by commas to get individual columns
        tempList = csvFileData[i].split(",")

        # column 1 is the closing price; convert it to float and append to closeData.
        closeData.append(float(tempList[1]))

    # return the list of closing prices
    return closeData

# SARIMA forecasting + plot
def runSARIMA(closeSeries, ticker):

    # SARIMA order
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # ARIMA parameters are p,d,q
    # p is the autoregressive order. "how much does the past price affect todays price?"
        # if p = 2, todays price depends on the last 2 days prices.
    # d is the degree of differencing.
        #it makes the series stationary so it removes trends or drifts. so, if prices keep going up,
        # d = 1 takes the difference from yesterday so the model looks at day to day change instead of raw price
    # q is the moving average order. "How much do past errors affect todays predictions?"
        # if the model predicted yesterday wrong by $2, and q = 1, that error will partially adjust todays prediction.

    # in order to use SARIMA, we use P, D, Q, and S. S is 12 because there are 12 months in a year

    # Fit model, disp=false suppresses all of the information output that the model generates
    model = SARIMAX(closeSeries, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # predict the next 30 days
    forecast = results.forecast(30)

    # Print results
    print("\n=== SARIMA MODEL RESULTS FOR", ticker, "===")

    # iloc allowes us to get the most recent closing price
    print("Last Close Price:", closeSeries.iloc[-1])
    print("\nNext 30-Day Forecast (first 5 shown):")

    # first 5 elements of forecast
    print(forecast[:5])
    print("\nModel AIC:", results.aic)
    # AIC measures how well the model fits, also penalizes too many parameters
     # lower AIC = better model that isnt overly complex
    print("Model BIC:", results.bic)
    # BIC penalizes complexity
     # harsher version of AIC that helps determine whether or not model is simple enough
     # lower BIC = better
    print("Model HQIC:", results.hqic)
    # HQIC penalizes model complexity less than BIC but more than AIC
     # more of a holistic approach to rating a model
     # lower is better
    

    # matplotlib default code just to make the graph appear to our preferences
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

    # train = everything except last 30 data points
    # test = last 30 data points
    
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # fit on training data only to forecast exactly 30 data points to compare with test (actual)
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
    plt.title("SARIMA Actual vs Predicted for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    # vertical line marks start of test set
# Call main
main()

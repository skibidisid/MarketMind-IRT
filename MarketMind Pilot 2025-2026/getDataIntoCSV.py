import yfinance as yf
import pandas as pd
import os
import numpy as np
import math

# Main method
def main ():

    # Setting ticker
    ticker = "VDE"

    # Creating csv file name
    csvFile = ticker + "_data.csv"

    # Getting data and putting into csv file
    get_data(ticker, csvFile)

    # Reading the data in the csv file
    csvFileData = read_csv(csvFile)

    # Grabbing just the close data
    closeData = getCloseData(csvFileData)

    # Calculating strandard deviation
    calcVolatility(closeData, ticker)

# Method to get data
def get_data (ticker, csvFile):

    # Downloading data as a data frame and converting into a csv
    data = yf.download(ticker, start="2000-11-19", end="2025-11-19")
    data.to_csv(csvFile)

# Reading csv to add to a list
def read_csv (csvFile):

    # If file exists
    if (os.path.exists(csvFile)):
    
        # Open and read
        fo = open(csvFile, "r") 
        csvFileData = fo.read().split("\n")

    else:

        # Say it doesn't exist
        print("File " + csvFile + " does not exist.")

    # Returning the file data
    return csvFileData

# Extracting just the close data
def getCloseData (csvFileData):

    # Creating blank list
    closeData = []

    # For i in range (3, to len csvFileData - 1)
    for i in range (3, len(csvFileData) - 1):

        # Splitting by , and creating tempList
        tempList = csvFileData[i].split(",") 

        # Appending the close data
        closeData.append(float(tempList[1]))

    # Returning the close data
    return closeData

# Calculating volatility
def calcVolatility (closeData, ticker):

    # Creating returns list
    returns = []

    # For each item in the list
    for i in range(1, len(closeData)):

        # Standard Deviation formula
        daily_return = (closeData[i] - closeData[i - 1]) / closeData[i - 1]

        # Appending the value to a list
        returns.append(daily_return)
    
    # Calculating total std deviation
    dailyVolatility = np.std(returns)
    
    # Day percent change
    dayChange = dailyVolatility * 100

    # Annual percent change
    annualStdDeviation = dailyVolatility * math.sqrt(252) * 100

    # Outputting daily/annual standard deviation
    print("Daily Standard Deviation of " + ticker + ": " + format(dayChange, ".2f") + "%")
    print("Annual Standard Deviation of " + ticker + ": " + format(annualStdDeviation, ".2f") + "%")

# Calling main
main()
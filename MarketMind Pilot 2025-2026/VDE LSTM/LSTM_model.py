# Imports
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Date to start getting data from
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Main method
def main():

    # Loading data for ticker VDE using the method above
    data = load_data('VDE')
    df = data
    df.head()

    df = df.drop(['Date'], axis = 1)  # Only drop Date, keep Close
    df.head()

    # Visualizing the close data on a chart (matplotlib)
    plt.title("Close Price Visualization")
    plt.plot(df['Close'])
    plt.show()
    df

    # 100-point rolling mean of the Close price (100 day moving average)
    ma100 = df.Close.rolling(100).mean()
    ma100

    # Creating the figure for the moving average
    plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    plt.plot(ma100, 'r')
    plt.title('Graph Of Moving Averages Of 100 Days')

    # 200-point rolling mean of the Close price (200 day moving average)
    ma200 = df.Close.rolling(200).mean()
    ma200

    # Creating the figure for the moving average
    plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.title('Comparision Of 100 Days And 200 Days Moving Averages')

    # Shape of dataframe (rows/columns)
    df.shape

    # Takes 70% of the rows and puts it into train; will train on this part of the data
    train = pd.DataFrame(data[0:int(len(data)*0.70)])

    # Takes remaining 30% of the rows and puts it into test; will be tested for accuracy on this new data
    test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])

    # Shapes are then printed
    print(train.shape)
    print(test.shape)

    # Displays first 5 rows
    train.head()
    test.head()

    # Creating the scaler (fits between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0,1))

    # Takes close prices
    train_close = train.iloc[:, 4:5].values
    test_close = test.iloc[:, 4:5].values

    # Fitting the data using the scaler
    data_training_array = scaler.fit_transform(train_close)
    data_training_array
    joblib.dump(scaler, 'scaler.save')


    # Create empty lists to store input sequences (X) and target values (y) for LSTM training
    x_train = []
    y_train = []

    # Slide a window of 100 past Close prices to predict the next Close price
    for i in range(100, data_training_array.shape[0]):

        # X: 100 previous timesteps as input sequence
        x_train.append(data_training_array[i-100: i])

        # y: target = Close price at timestep i (next value after window)
        y_train.append(data_training_array[i, 0])

    # Convert lists to arrays; shape becomes (samples, 100 timesteps, 1 feature)
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Verify shape: should be (n_samples, 100, 1) for LSTM input
    x_train.shape

    # Building lstm
    model = Sequential()

    # First LSTM layer: 50 units, ReLU activation, returns sequences for stacking
    model.add(LSTM(units=50, activation='relu', return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))  # (100 timesteps, 1 feature)
    model.add(Dropout(0.2))  # 20% dropout to prevent overfitting

    # Second LSTM layer: 60 units
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    # Third LSTM layer: 80 units  
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    # Final LSTM layer: 120 units (no return_sequences, outputs final prediction)
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer: 1 unit for single price prediction (regression)
    model.add(Dense(units=1))

    # Display model architecture summary
    model.summary()

    # Training models using checkpoints
    # Save best model weights during training based on training loss
    checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',     # Save path
        monitor='loss',           # Track training loss
        mode='min',               # Save when loss decreases
        save_best_only=True,      # Only keep best model
        verbose=1                 # Print save notifications
    )

    # Compile: Adam optimizer, MSE loss (regression), track MAE metric
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])

    # Train for 50 epochs, save best model automatically
    model.fit(x_train, y_train, epochs=50, callbacks=[checkpoint])

    # Save final trained model
    model.save('keras_model.h5')

    # Preparing test data

    # Check test data shape
    test_close.shape

    # Get last 100 days from training as starting point for predictions
    past_100_days = pd.DataFrame(train_close[-100:])

    # Convert test Close prices to DataFrame
    test_df = pd.DataFrame(test_close)

    # Combine: last 100 train days + all test days (for continuous prediction)
    final_df = pd.concat([past_100_days, test_df], ignore_index=True)

    # Scale entire test dataset using same scaler parameters
    input_data = scaler.fit_transform(final_df)  

    # CREATE TEST SEQUENCES (same format as training)
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])      
        y_test.append(input_data[i, 0])          

    x_test, y_test = np.array(x_test), np.array(y_test)

    print(x_test.shape)  
    print(y_test.shape)

    # Generate predictions on test set
    y_pred = model.predict(x_test)

    # Inverse scale predictions and actual values back to original price scale
    scale_factor = 1 / scaler.scale_[0]  # Extract from fitted scaler
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Plot actual vs predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label="Original Price")
    plt.plot(y_pred, 'r', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Evaluating MAE (error)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean absolute error on test set: ", mae)

# Downloading price data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Calling main
main ()

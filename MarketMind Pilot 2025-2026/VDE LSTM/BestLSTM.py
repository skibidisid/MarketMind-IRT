# SAVED BEST LSTM, USE THIS FOR ACTUAL PREDICTION 

# Libraries
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import joblib
import yfinance as yf


# Load the best model you trained
model = load_model('best_lstm_model.h5')

# Load the scaler you saved
scaler = joblib.load('scaler.save')


# Get fresh data
new_data = yf.download('VDE', start='2020-01-01', end='2025-11-24')
close_prices = new_data['Close'].values.reshape(-1, 1)

# Scale it using YOUR saved scaler (MUST use same scaler!)
scaled_data = scaler.transform(close_prices)

# Create sequences (last 100 days predict next day)
x_test_new = []
for i in range(100, len(scaled_data)):
    x_test_new.append(scaled_data[i-100:i])
x_test_new = np.array(x_test_new)

# Get predictions from your model
y_pred = model.predict(x_test_new)

print("Predictions shape:", y_pred.shape)
print("First 5 predictions (scaled):", y_pred[:5])

# Convert scaled predictions back to actual dollar prices
y_pred_actual = scaler.inverse_transform(y_pred)

y_test_actual = close_prices[100:]

print("Predicted prices:", y_pred_actual[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, 'b', label='Actual Price')
plt.plot(y_pred_actual, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('LSTM Predictions vs Actual')
plt.show()

# ========================================
# PREDICT TOMORROW (ONE DAY INTO FUTURE)
# ========================================

# Get the LAST 100 days from your scaled data
last_100_days = scaled_data[-100:]

# Reshape for prediction (model expects shape: (1, 100, 1))
x_input = last_100_days.reshape(1, 100, 1)

# Make prediction for tomorrow
tomorrow_pred_scaled = model.predict(x_input, verbose=0)

# Convert back to actual price
tomorrow_pred_actual = scaler.inverse_transform(tomorrow_pred_scaled)

print("\n" + "="*50)
print("TOMORROW'S PREDICTION")
print("="*50)
print(f"Today's close (2025-11-24): ${close_prices[-1][0]:.2f}")
print(f"Tomorrow's predicted price: ${tomorrow_pred_actual[0][0]:.2f}")
print(f"Expected change: ${tomorrow_pred_actual[0][0] - close_prices[-1][0]:.2f}")
print(f"Expected % change: {((tomorrow_pred_actual[0][0] / close_prices[-1][0]) - 1) * 100:.2f}%")
print("="*50)


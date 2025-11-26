#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/034adarsh/Stock-Price-Prediction-Using-LSTM/blob/main/LSTM_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Import all the required libraries
# 
# ---
# 
# 

# In[1]:


import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


# # Define start day to fetch the dataset from the yahoo finance library
# 
# ---
# 
# 

# In[2]:


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Define a function to load the dataset

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# In[3]:


data = load_data('VDE')
df=data
df.head()


df = df.drop(['Date'], axis = 1)  # Only drop Date, keep Close
df.head()

# In[5]:
plt.title("Close Price Visualization")
plt.plot(df['Close'])
plt.show()



# In[6]:


df


# # Plotting moving averages of 100 day
# 
# ---
# 
# 

# In[7]:


ma100 = df.Close.rolling(100).mean()
ma100


# In[8]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.title('Graph Of Moving Averages Of 100 Days')


# # Defining 200 days moving averages and plotting comparision graph with 100 days moving averages
# 
# ---
# 
# 

# In[9]:


ma200 = df.Close.rolling(200).mean()
ma200


# In[10]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.title('Comparision Of 100 Days And 200 Days Moving Averages')


# In[12]:


df.shape


# # Spliting the dataset into training (70%) and testing (30%) set

# In[25]:


# Splitting data into training and testing

train = pd.DataFrame(data[0:int(len(data)*0.70)])
test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])

print(train.shape)
print(test.shape)


# In[26]:


train.head()


# In[27]:


test.head()


# # Using MinMax scaler for normalization of the dataset
# 
# ---
# 
# 

# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))




# In[44]:


train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values


# In[29]:


data_training_array = scaler.fit_transform(train_close)
data_training_array

import joblib
joblib.dump(scaler, 'scaler.save')

# In[36]:


x_train = []
y_train = [] 

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) 


# In[37]:


x_train.shape


# # ML Model (LSTM)
# 
# ---
# 
# 

# In[38]:


from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


# In[39]:


model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
              ,input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


# In[40]:


model.summary()


# # Training the model
# 
# ---
# 
# 

# In[74]:

checkpoint = ModelCheckpoint(
    'best_lstm_model.h5',     # Filepath to save the model
    monitor='loss',           # Or use 'val_loss' if you have validation_data
    mode='min',               # Save only if the loss decreases
    save_best_only=True,      # Only save the best model
    verbose=1                 # Print message when saving
)


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MAE'])
model.fit(x_train, y_train, epochs=50, callbacks=[checkpoint])
  # Remove validation_data for now


# In[42]:


model.save('keras_model.h5')


# In[ ]:


test_close.shape
test_close


# In[54]:


past_100_days = pd.DataFrame(train_close[-100:])


# In[55]:


test_df = pd.DataFrame(test_close)


# **Defining the final dataset for testing by including last 100 coloums of the training dataset to get the prediction from the 1st column of the testing dataset.**
# 
# ---
# 

# In[56]:


final_df = pd.concat([past_100_days, test_df], ignore_index=True)


# In[58]:


final_df.head()


# In[ ]:


input_data = scaler.fit_transform(final_df)
input_data


# In[60]:


input_data.shape


# # Testing the model
# 
# ---
# 
# 

# In[62]:


x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i])
   y_test.append(input_data[i, 0])


# In[64]:


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# # Making prediction and plotting the graph of predicted vs actual values
# 
# ---
# 
# 

# In[65]:


# Making predictions

y_pred = model.predict(x_test)


# In[66]:


y_pred.shape


# In[67]:


y_test


# In[ ]:


y_pred


# In[69]:


scaler.scale_


# In[70]:


scale_factor = 1/0.00985902
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor


# In[71]:


plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# # Model evaluation

# In[76]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error on test set: ", mae)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st

# Fetch data from Yahoo Finance
def load_data(ticker):
    data = yf.download(ticker, start='2010-01-01', end='2023-06-01')
    return data

# Preprocess the data
def preprocess_data(data):
    dataset = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    return X_train, y_train, X_test, y_test, scaler, scaled_data

# Create dataset function
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Build the ANN model
def build_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(100,)))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plotting function
def plot_predictions(scaled_data, train_predict, test_predict, scaler, time_step):
    train_predict_plot = np.empty_like(scaled_data, dtype=np.float32)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict)+time_step, :] = scaler.inverse_transform(train_predict)

    test_predict_plot = np.empty_like(scaled_data, dtype=np.float32)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1, :] = scaler.inverse_transform(test_predict)

    plt.figure(figsize=(14, 5))
    plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
    plt.plot(train_predict_plot, label='Training Prediction')
    plt.plot(test_predict_plot, label='Testing Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Streamlit app
st.title('Stock Price Prediction using ANN')
ticker = st.text_input('Enter Stock Ticker', 'UNVR.JK')

if st.button('Predict'):
    data = load_data(ticker)
    X_train, y_train, X_test, y_test, scaler, scaled_data = preprocess_data(data)
    
    model = build_model()
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    plot_predictions(scaled_data, train_predict, test_predict, scaler, 100)
    
    train_rmse = np.sqrt(np.mean(((train_predict - y_train) ** 2)))
    test_rmse = np.sqrt(np.mean(((test_predict - y_test) ** 2)))
    
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')

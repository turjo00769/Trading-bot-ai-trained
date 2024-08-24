import os
import pandas as pd
import numpy as np
import talib
import ta
from finta import TA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure GPU is used if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Function to add technical indicators using TA-Lib and TA
def add_features(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])
    df['stochastic_k'], df['stochastic_d'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Function to scale data and create sequences for LSTM
def prepare_data(df, target_column='close', sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, df.columns.get_loc(target_column)])
    
    return np.array(X), np.array(y), scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and save the best model
def process_and_train_with_error_learning(input_directory, sequence_length=60, num_iterations=100, chunk_size=10000):
    best_model = None
    best_accuracy = float('-inf')
    best_iteration = -1

    for asset in os.listdir(input_directory):
        asset_path = os.path.join(input_directory, asset)
        if not os.path.isdir(asset_path):
            continue

        for year in os.listdir(asset_path):
            year_path = os.path.join(asset_path, year)
            if not os.path.isdir(year_path):
                continue

            for file in os.listdir(year_path):
                file_path = os.path.join(year_path, file)
                if not file_path.endswith('.csv'):
                    continue

                df_chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
                for df in df_chunk_iter:
                    add_features(df)

                    X, y, scaler = prepare_data(df)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    for i in range(num_iterations):
                        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

                        accuracy = model.evaluate(X_test, y_test, verbose=0)
                        print(f"Iteration {i+1} - Model Accuracy: {accuracy}")

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_iteration = i+1

    best_model.save(f'/content/drive/MyDrive/Quotex/trained_models/best_model_iteration_{best_iteration}.h5')
    print(f"Best Model Saved with Accuracy: {best_accuracy} on Iteration {best_iteration}")

# Main Execution
input_directory = '/content/drive/MyDrive/Quotex/oanda'  # Update this path accordingly
process_and_train_with_error_learning(input_directory)

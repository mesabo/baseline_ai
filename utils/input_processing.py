#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/28/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
import numpy as np
import pandas as pd
def create_time_series(dataset, look_back, forecast_period):
    """
    Create time series data for LSTM input and forecasting.

    Args:
        dataset (np.array): Normalized dataset.
        look_back (int): Number of previous days to use as input.
        forecast_period (int): Number of future days to predict.

    Returns:
        tuple: Input features (X) and target labels (Y).
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), :])  # Look-back window
        Y.append(dataset[(i + look_back):(i + look_back + forecast_period), 0])  # Forecast target
    return np.array(X), np.array(Y)

from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col, look_back, forecast_period):
    """
    Preprocess the data by splitting into train/validation/test sets and normalizing features.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Name of the target column.
        look_back (int): Number of previous days to use as input.
        forecast_period (int): Number of future days to predict.

    Returns:
        tuple: Train, validation, and test sets for LSTM.
    """
    df = df.dropna()  # Handle missing values
    train, validate, test = split_dataset(df, train_years=2, validate_years=1, test_years=1)

    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.drop(columns=["Date"]))
    validate_scaled = scaler.transform(validate.drop(columns=["Date"]))
    test_scaled = scaler.transform(test.drop(columns=["Date"]))

    # Create time series
    X_train, y_train = create_time_series(train_scaled, look_back, forecast_period)
    X_val, y_val = create_time_series(validate_scaled, look_back, forecast_period)
    X_test, y_test = create_time_series(test_scaled, look_back, forecast_period)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def split_dataset(data, train_years=2, validate_years=1, test_years=1):
    """
    Splits the dataset into training, validation, and testing sets based on the number of years.

    Parameters:
        data (pd.DataFrame): The dataset with a 'Date' column.
        train_years (int): Number of years for the training set.
        validate_years (int): Number of years for the validation set.
        test_years (int): Number of years for the test set.

    Returns:
        tuple: Three DataFrames (train, validate, test).
    """
    # Ensure the Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Define the split dates
    train_end_date = data['Date'].min() + pd.DateOffset(years=train_years)
    validate_end_date = train_end_date + pd.DateOffset(years=validate_years)
    test_end_date = validate_end_date + pd.DateOffset(years=test_years)

    # Create subsets
    train = data[data['Date'] < train_end_date]
    validate = data[(data['Date'] >= train_end_date) & (data['Date'] < validate_end_date)]
    test = data[(data['Date'] >= validate_end_date) & (data['Date'] < test_end_date)]

    return train, validate, test
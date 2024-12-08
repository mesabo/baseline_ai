#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/08/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col):
    """
    Preprocess the data by handling missing values, splitting into train/validation/test sets,
    and normalizing the features.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Name of the target column.

    Returns:
        tuple: Train, validation, and test splits as (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    df = df.dropna()  # Drop rows with missing values

    X_droped = df.drop(columns=[target_col])
    X_np = X_droped.select_dtypes(include=np.number) # selected only columns with number values
    X = X_np.values # keep only data values without header

    y = df[target_col].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from argument_parser import parse_arguments

# Load CSV data
def load_csv_data(filepath):
    """
    Load CSV data from the specified file path.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df

# Preprocess the data
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
    X = df.drop(columns=[target_col]).select_dtypes(include=np.number).values
    y = df[target_col].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class SimpleRegression(nn.Module):
    """
    A simple fully connected regression model.

    Args:
        input_size (int): The number of input features.
    """
    def __init__(self, input_size):
        super(SimpleRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the SimpleRegression model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNConv1(nn.Module):
    """
    A 1D convolutional neural network with a single convolutional layer.

    Args:
        input_size (int): The size of the input sequence.
    """
    def __init__(self, input_size):
        super(CNNConv1, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * (input_size // 2), 64)
        self.act2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the CNNConv1 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act2(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNConv2(nn.Module):
    """
    A 1D convolutional neural network with two convolutional layers.

    Args:
        input_size (int): The size of the input sequence.
    """
    def __init__(self, input_size):
        super(CNNConv2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the CNNConv2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
# Train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

# Visualize loss
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Trend')
    plt.legend()
    plt.show()

# Plot true vs predicted values
def plot_real_predictions(true_values, predicted_values):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True Values", marker="o", linestyle="-", linewidth=2)
    plt.plot(predicted_values, label="Predicted Values", marker="x", linestyle="--", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Yield (kg)")
    plt.title("True vs Predicted Values for Last 20 Rows")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

# CLI-enabled main function
def main():
    """
    Main function to run the selected regression model.
    Parses command-line arguments for model type, optimizer type, batch size, and epochs.
    """
    args = parse_arguments()

    # Load and preprocess data
    filepath = "../data/farm_fishing_time_series_data.csv"
    df = load_csv_data(filepath)
    target_col = "Yield (kg)"
    df = df.drop("Date", axis=1)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(df, target_col)

    # Reshape for CNNs
    if args.model_type != "SimpleRegression":
        X_train = X_train[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

    # Prepare DataLoader
    train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in zip(X_train, y_train)]
    val_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in zip(X_val, y_val)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Select model
    input_size = X_train.shape[1] if args.model_type == "SimpleRegression" else X_train.shape[2]
    if args.model_type == "SimpleRegression":
        model = SimpleRegression(input_size)
    elif args.model_type == "CNNConv1":
        model = CNNConv1(input_size)
    elif args.model_type == "CNNConv2":
        model = CNNConv2(input_size)

    # Configure optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train and evaluate
    criterion = nn.MSELoss()
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, args.epochs)
    plot_loss(train_losses, val_losses)

    # Test evaluation and plotting
    test_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in zip(X_test, y_test)]
    test_X, test_y = zip(*test_data)
    test_X = torch.stack(test_X)
    test_y = torch.tensor(test_y)
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).squeeze().numpy()
    mse = mean_squared_error(test_y, predictions)
    print(f"Test Mean Squared Error: {mse:.4f}")
    plot_real_predictions(test_y[-20:].numpy(), predictions[-20:])

if __name__ == "__main__":
    main()
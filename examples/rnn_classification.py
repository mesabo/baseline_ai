#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/15/2024
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

# Argument parser for CLI
def parse_arguments():
    parser = argparse.ArgumentParser(description="Time-Series Modeling with RNNs.")
    parser.add_argument("--model_type", choices=["LSTM", "GRU", "SimpleRegression"], default="LSTM", help="Type of model to train.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer type.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    return parser.parse_args()

# Load CSV data
def load_csv_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df

# Create sequences for RNN input
def create_sequences(data, target, sequence_length):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(target[i + sequence_length])
    return np.array(sequences), np.array(targets)

# Preprocess the data
def preprocess_data(df, target_col, sequence_length):
    df = df.dropna()
    X = df.drop(columns=[target_col]).select_dtypes(include=np.number).values
    y = df[target_col].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = create_sequences(X_train, y_train, sequence_length)
    X_val, y_val = create_sequences(X_val, y_val, sequence_length)
    X_test, y_test = create_sequences(X_test, y_test, sequence_length)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.fc(hn[-1])

# Train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
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
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
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
def MyClassification():
    args = parse_arguments()
    filepath = "../data/farm_fishing_time_series_data.csv"
    df = load_csv_data(filepath)
    target_col = "Yield (kg)"
    sequence_length = 10  # Sequence length for RNNs
    df = df.drop("Date", axis=1)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(df, target_col, sequence_length)
    train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in zip(X_train, y_train)]
    val_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in zip(X_val, y_val)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    input_size = X_train.shape[2]
    if args.model_type == "LSTM":
        model = LSTMModel(input_size, hidden_size=64, num_layers=2, output_size=1)
    elif args.model_type == "GRU":
        model = GRUModel(input_size, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) if args.optimizer == "adam" else optim.SGD(model.parameters(), lr=0.001)
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, args.epochs)
    plot_loss(train_losses, val_losses)
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
    MyClassification()

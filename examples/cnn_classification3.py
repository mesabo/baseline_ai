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

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Load CSV data
def load_csv_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df


# Preprocess the data
def preprocess_data(df, target_col):
    # Drop rows with missing values (if any)
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=[target_col]).select_dtypes(include=np.number).values
    y = df[target_col].values

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Model definition for regression
class SimpleRegressor(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# Train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50):
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


# Main function
def main():
    filepath = "../data/farm_fishing_time_series_data.csv"
    df = load_csv_data(filepath)

    target_col = "Yield (kg)"  # Update this with the correct target column
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(df, target_col)

    # Convert data to PyTorch tensors
    train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in
                  zip(X_train, y_train)]
    val_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in
                zip(X_val, y_val)]
    test_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in
                 zip(X_test, y_test)]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)

    # Initialize model, loss, and optimizer
    input_size = X_train.shape[1]
    model = SimpleRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader)

    # Plot loss
    plot_loss(train_losses, val_losses)

    # Evaluate the model on the test set
    test_X, test_y = zip(*test_data)
    test_X = torch.stack(test_X)
    test_y = torch.tensor(test_y)
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).squeeze()
    mse = mean_squared_error(test_y, predictions)
    print(f"Test Mean Squared Error: {mse:.4f}")

    # Predict on 5 new samples
    real_data = np.random.rand(5, X_train.shape[1])
    real_data = StandardScaler().fit_transform(real_data)  # Standardize real data
    real_predictions = model(torch.tensor(real_data, dtype=torch.float32)).squeeze().detach().numpy()
    print("Predictions on real data:", real_predictions)


if __name__ == "__main__":
    main()

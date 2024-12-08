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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error

from utils.load_csv_data import load_csv_data
from utils.preprocess_data import preprocess_data
from utils.visualize_result import plot_loss, plot_real_predictions
from models.cnns.simple_regression import SimpleRegression
from models.cnns.cnn_conv1 import CNNConv1
from models.cnns.cnn_conv2 import CNNConv2


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


def MyClassification(args):
    """
    Main function to run the selected regression model.
    Parses command-line arguments for model type, optimizer type, batch size, and epochs.
    """

    # Load and preprocess data
    filepath = "data/farm_fishing_time_series_data.csv"

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

    # Select model [[rows, 1, cols],[rows, 1, cols],[rows, 1, cols],...]
    input_size = X_train.shape[1] if args.model_type == "SimpleRegression" else X_train.shape[2]

    if args.model_type == "CNNConv1":
        model = CNNConv1(input_size)
    elif args.model_type == "CNNConv2":
        model = CNNConv2(input_size)
    else:
        model = SimpleRegression(input_size)


    # Configure optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    save_image_path = f"output/{args.model_type}_{args.optimizer}_{args.batch_size}_{args.epochs}"

    # Train and evaluate
    criterion = nn.MSELoss()
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, args.epochs)
    plot_loss(save_image_path, train_losses, val_losses)

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
    plot_real_predictions(save_image_path, test_y[:50].numpy(), predictions[:50])

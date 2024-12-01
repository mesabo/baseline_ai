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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Function to create a synthetic dataset
def create_dataset(num_samples=100, num_features=5):
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)
    return X, y


# Function to preprocess the dataset
def preprocess_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Model definition
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# Function for training the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(axis=1) == y_batch).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(correct / len(train_loader.dataset))

        # Validation phase
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                correct += (outputs.argmax(axis=1) == y_batch).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / len(val_loader.dataset))

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}, Train Acc = {train_accuracies[-1]:.4f}, Val Acc = {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


# Function for visualizing loss and accuracy
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label="Train Accuracy")
    plt.plot(range(epochs), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()

    plt.show()


# Function for making predictions
def make_predictions(model, X_real):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_real, dtype=torch.float32)).argmax(axis=1)
    return predictions


# Main function
def main():
    # Create and preprocess dataset
    X, y = create_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(X, y)

    # Convert data to PyTorch tensors
    train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)) for x, y in
                  zip(X_train, y_train)]
    val_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)) for x, y in zip(X_val, y_val)]
    test_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)) for x, y in
                 zip(X_test, y_test)]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)

    # Initialize model, loss, and optimizer
    input_size = X.shape[1]
    model = SimpleClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, criterion, optimizer, train_loader,
                                                                             val_loader)

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # Test the model
    test_X, test_y = zip(*test_data)
    test_X = torch.stack(test_X)
    test_y = torch.tensor(test_y)
    test_predictions = make_predictions(model, test_X)
    test_accuracy = accuracy_score(test_y, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predict on 5 rows of real data
    real_data = np.random.rand(5, X.shape[1])
    real_data = StandardScaler().fit_transform(real_data)  # Standardize real data
    real_predictions = make_predictions(model, real_data)
    print("Real Data Predictions:", real_predictions.numpy())


if __name__ == "__main__":
    main()

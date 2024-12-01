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


# CNN model definition for regression
class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(32 * (input_size // 4), 64)  # Flattened feature size
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
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


# Plot true vs predicted values for the last 20 rows as lines
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


# Update main function to include line plot
def main():
    filepath = "../data/farm_fishing_time_series_data.csv"
    df = load_csv_data(filepath)

    target_col = "Yield (kg)"  # Update this with the correct target column
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(df, target_col)

    # Reshape inputs for CNN (1D convolution expects input shape [batch_size, channels, sequence_length])
    X_train = X_train[:, np.newaxis, :]  # Add channel dimension
    X_val = X_val[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

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
    input_size = X_train.shape[2]
    model = CNNRegressor(input_size)
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
        predictions = model(test_X).squeeze().numpy()
    mse = mean_squared_error(test_y, predictions)
    print(f"Test Mean Squared Error: {mse:.4f}")

    # Predict on the last 20 rows of the test set
    real_data = test_X[-20:]  # Last 20 rows
    real_y = test_y[-20:]  # Corresponding true values
    with torch.no_grad():
        real_predictions = model(real_data).squeeze().numpy()

    # Print predictions
    for i in range(len(real_y)):
        print(f"True: {real_y[i]:.4f}, Predicted: {real_predictions[i]:.4f}")

    # Plot predictions as line plot
    plot_real_predictions(real_y.numpy(), real_predictions)


if __name__ == "__main__":
    main()

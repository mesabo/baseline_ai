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

import os

import torch
import torch.nn as nn

from models.rnns.lstm_forecasting import LSTMForecasting, LSTM2Forecasting, LSTM2CNNForecasting
from utils.input_processing import preprocess_data
from utils.load_csv_data import load_csv_data
from utils.output_model import ModelHandler
from utils.visualize_result import plot_loss, plot_multi_steps_predictions


class RunRNNModel:
    """
    Class to manage RNN-based model training, validation, and evaluation.

    Args:
        args (argparse.Namespace): Parsed arguments from the command line.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dynamically load the model class based on args.model_type
        self.model_class = {
            "LSTM": LSTMForecasting,
            "LSTM2": LSTM2Forecasting,
            "LSTM2CNN": LSTM2CNNForecasting,
            # "GRU": GRUForecasting,
            # "BiLSTM": BiLSTMForecasting,
            # "BiGRU": BiGRUForecasting,
        }.get(args.model_type, None)

        if not self.model_class:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        # Validate paths
        if not os.path.isdir(self.args.output_dir):
            raise ValueError(f"Output directory does not exist: {self.args.output_dir}")
        if not os.path.isfile(self.args.dataset_path):
            raise ValueError(f"Dataset file not found: {self.args.dataset_path}")

        # Output directory for models and logs
        self.output_dir = f"{self.args.output_dir}/{args.model_type}_{args.optimizer}_{args.batch_size}_{args.epochs}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the ModelHandler for saving/loading models
        self.model_handler = ModelHandler()

    def train_rnn(self, model, criterion, optimizer, train_loader, val_loader):
        train_losses, val_losses = [], []
        os.makedirs(os.path.join(self.output_dir, "model"), exist_ok=True)  # Ensure the directory exists

        for epoch in range(self.args.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch.float())
                loss = criterion(outputs, y_batch.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch.float())
                    loss = criterion(outputs, y_batch.float())
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model after each epoch
            model_save_path = os.path.join(self.output_dir, "model", f"{self.args.model_type}_epoch_{epoch + 1}.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Create directories if needed
            self.model_handler.save_model(model, model_save_path)

        return train_losses, val_losses

    def run(self):
        print(f"Running {self.args.model_type} Forecasting Model...")

        # Load dataset
        df = load_csv_data(self.args.dataset_path)
        look_back = self.args.lookback_days
        forecast_period = self.args.forecast_days
        target_col = self.args.target_column
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = preprocess_data(
            df, target_col, look_back, forecast_period
        )

        # Define model parameters
        input_size = X_train.shape[2]
        lstm_hidden_size = 64
        lstm_num_layers = 2
        bidirectional = False  # Change as needed
        cnn_input_size = lstm_hidden_size * (2 if bidirectional else 1)

        # Instantiate the model
        model = self.model_class(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            cnn_input_size=cnn_input_size,
            output_size=forecast_period,
            dropout=0.1,  # Optional dropout
            bidirectional=bidirectional
        ).to(self.device)

        model_save_path = os.path.join(self.output_dir, "model", f"{self.args.model_type}_latest.pth")
        plot_save_path = os.path.join(self.output_dir, "images", f"{self.args.model_type}")

        if self.args.mode == "test":
            if os.path.exists(model_save_path):
                print(f"Loading model from {model_save_path}")
                self.model_handler.load_model(model, model_save_path)
            else:
                raise FileNotFoundError(f"No trained model found at {model_save_path}")
        else:
            print("Starting training...")
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=self.args.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                batch_size=self.args.batch_size, shuffle=False
            )
            train_losses, val_losses = self.train_rnn(model, criterion, optimizer, train_loader, val_loader)
            plot_loss(plot_save_path, train_losses, val_losses, self.args.mode)
            self.model_handler.save_model(model, model_save_path)

        # Test
        print("Testing model...")
        model.eval()
        with torch.no_grad():
            test_predictions = model(torch.tensor(X_test).float().to(self.device)).cpu().numpy()
        plot_multi_steps_predictions(plot_save_path, y_test[:20], test_predictions[:20], self.args.mode)
        print("Testing complete.")
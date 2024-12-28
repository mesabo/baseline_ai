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

import torch
import torch.nn as nn


class LSTMForecasting(nn.Module):
    """
    LSTM-based model for multi-step forecasting.

    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of hidden units in the LSTM.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of steps to forecast.
        dropout (float, optional): Dropout rate between LSTM layers. Default is 0.0.
        bidirectional (bool, optional): If True, use bidirectional LSTMs. Default is False.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=False):
        super(LSTMForecasting, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout only applies if num_layers > 1
            bidirectional=bidirectional
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, look_back, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_period).
        """
        # LSTM output: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Apply Layer Normalization to the LSTM outputs (optional, improves training stability)
        lstm_out = self.layer_norm(lstm_out)

        # Select the last hidden state for each sequence
        last_hidden_state = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)

        # Fully connected layer to predict the forecast period
        output = self.fc(last_hidden_state)  # Shape: (batch_size, forecast_period)

        return output


class LSTM2Forecasting(nn.Module):
    """
    LSTM-based model for multi-step forecasting with additional dense layers.

    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of hidden units in the LSTM.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of steps to forecast.
        dropout (float, optional): Dropout rate between LSTM layers. Default is 0.0.
        bidirectional (bool, optional): If True, use bidirectional LSTMs. Default is False.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=False):
        super(LSTM2Forecasting, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Fully connected layers with activation functions
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        # Final output layer
        self.fc3 = nn.Linear(32, output_size)

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, look_back, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_period).
        """
        # LSTM output: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Apply Layer Normalization to the LSTM outputs (optional, improves training stability)
        lstm_out = self.layer_norm(lstm_out)

        # Select the last hidden state for each sequence
        last_hidden_state = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)

        # Pass through dense layers with activation
        dense_out = self.fc1(last_hidden_state)
        dense_out = self.relu1(dense_out)
        dense_out = self.fc2(dense_out)
        dense_out = self.relu2(dense_out)

        # Final output layer
        output = self.fc3(dense_out)  # Shape: (batch_size, forecast_period)

        return output

class LSTM2CNNForecasting(nn.Module):
    """
    Hybrid model combining LSTM and CNN for multi-step forecasting.

    Args:
        input_size (int): Number of features in the input.
        lstm_hidden_size (int): Number of hidden units in the LSTM.
        lstm_num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of steps to forecast.
        cnn_input_size (int): The size of the input sequence for CNN.
        dropout (float, optional): Dropout rate for LSTM. Default is 0.0.
        bidirectional (bool, optional): If True, use bidirectional LSTMs. Default is False.
    """

    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        lstm_num_layers,
        cnn_input_size,
        output_size,
        dropout=0.0,
        bidirectional=False,
    ):
        super(LSTM2CNNForecasting, self).__init__()

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.lstm_layer_norm = nn.LayerNorm(lstm_hidden_size * (2 if bidirectional else 1))

        # CNN Component
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        # Fully connected layers after CNN
        flattened_size = 16 * (cnn_input_size // 2)  # Output size after CNN pooling
        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass of the hybrid LSTM + CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, look_back, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # LSTM: Process the sequential data
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, look_back, lstm_hidden_size * num_directions)
        lstm_out = self.lstm_layer_norm(lstm_out)

        # Select the last hidden state for each sequence
        last_hidden_state = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size * num_directions)

        # CNN: Expand dimensions for 1D convolution
        cnn_input = last_hidden_state.unsqueeze(1)  # Shape: (batch_size, 1, lstm_hidden_size * num_directions)
        cnn_out = self.conv1(cnn_input)
        cnn_out = self.act1(cnn_out)
        cnn_out = self.pool1(cnn_out)

        # Flatten and pass through fully connected layers
        cnn_out_flattened = cnn_out.view(cnn_out.size(0), -1)  # Shape: (batch_size, flattened_size)
        dense_out = self.fc1(cnn_out_flattened)
        dense_out = self.relu1(dense_out)
        output = self.fc2(dense_out)  # Final output

        return output
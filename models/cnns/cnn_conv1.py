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
        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)
        return x

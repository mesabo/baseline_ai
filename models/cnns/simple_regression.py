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
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

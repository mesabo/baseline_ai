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


class ModelHandler:
    """
    Utility class for saving and loading PyTorch models.
    """

    @staticmethod
    def save_model(model, path):
        """
        Save the model weights to a file.

        Args:
            model (torch.nn.Module): Model to save.
            path (str): Path to save the model weights.
        """
        # Ensure the directory for the file exists
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), path)
        print(f"Model weights saved to {path}")

    @staticmethod
    def load_model(model, path):
        """
        Load the model weights from a file.

        Args:
            model (torch.nn.Module): Model to load weights into.
            path (str): Path to the saved model weights.
        """
        model.load_state_dict(torch.load(path))
        print(f"Model weights loaded from {path}")
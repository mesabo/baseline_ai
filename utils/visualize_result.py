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
import matplotlib.pyplot as plt

# Visualize loss
def plot_loss(save_image_path, train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Trend')
    plt.legend()
    if save_image_path:
        plt.savefig(f"{save_image_path}_loss.png")
    else:
        plt.show()
    plt.clf()

# Plot true vs predicted values
def plot_real_predictions(save_image_path, true_values, predicted_values):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True Values", marker="o", linestyle="-", linewidth=2)
    plt.plot(predicted_values, label="Predicted Values", marker="x", linestyle="--", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Yield (kg)")
    plt.title("True vs Predicted Values for Last 20 Rows")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    if save_image_path:
        plt.savefig(f"{save_image_path}_prediction.png")
    else:
        plt.show()
    plt.clf()

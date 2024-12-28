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
import os
import numpy as np
import matplotlib.pyplot as plt

# Visualize loss
def plot_loss(save_image_path, train_losses, val_losses,run_mode="train_val"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Trend')
    plt.legend()
    if save_image_path:
        # Ensure the output directory exists
        directory = os.path.dirname(save_image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{run_mode}_loss.png")
    else:
        plt.show()
    plt.clf()

# Plot true vs predicted values
def plot_real_predictions(save_image_path, true_values, predicted_values, run_mode="train_val"):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True Values", marker="o", linestyle="-", linewidth=2)
    plt.plot(predicted_values, label="Predicted Values", marker="x", linestyle="--", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Yield (kg)")
    plt.title("True vs Predicted Values for Last 20 Rows")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    if save_image_path:
        print(save_image_path)
        # Ensure the output directory exists
        directory = os.path.dirname(save_image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{run_mode}_prediction.png")
    else:
        plt.show()
    plt.clf()

# Plot true vs predicted values for multi-step predictions
def plot_multi_steps_predictions(save_image_path, true_values, predicted_values, run_mode="train_val"):
    """
    Plot true vs predicted values for each step in multi-step predictions.

    Args:
        save_image_path (str): Path to save the plots.
        true_values (np.array): Array of true values (batch_size, forecast_steps).
        predicted_values (np.array): Array of predicted values (batch_size, forecast_steps).
        run_mode (str): Mode of the run (e.g., "train_val", "test").
    """
    num_steps = predicted_values.shape[1]  # Number of forecast steps
    for step in range(num_steps):
        plt.figure(figsize=(12, 6))
        plt.plot(true_values[:, step], label=f"True Values (Step {step + 1})", marker="o", linestyle="-", linewidth=2)
        plt.plot(predicted_values[:, step], label=f"Predicted Values (Step {step + 1})", marker="x", linestyle="--", linewidth=2)
        plt.xlabel("Sample Index")
        plt.ylabel("Yield (kg)")
        plt.title(f"True vs Predicted Values - Step {step + 1}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        if save_image_path:
            # Ensure the output directory exists
            directory = os.path.dirname(save_image_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"{directory}/{run_mode}_prediction_step_{step + 1}.png")
        else:
            plt.show()
        plt.clf()

def plot_step_dependent_predictions(save_image_path, true_values, predicted_values, run_mode="train_val"):
    """
    Plot multi-step true vs predicted values where predictions depend on earlier steps.

    Args:
        save_image_path (str): Path to save the plot.
        true_values (np.array): Array of true values (batch_size, forecast_steps).
        predicted_values (np.array): Array of predicted values (batch_size, forecast_steps).
        run_mode (str): Mode of the run (e.g., "train_val", "test").
    """
    plt.figure(figsize=(14, 8))
    num_samples = min(10, true_values.shape[0])  # Plot only the first 10 samples
    num_steps = true_values.shape[1]  # Number of forecast steps

    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))  # Unique colors for samples

    for sample_idx in range(num_samples):
        # Plot the true trajectory
        plt.plot(
            range(num_steps),
            true_values[sample_idx],
            label=f"True (Sample {sample_idx + 1})",
            color=colors[sample_idx],
            marker="o",
            linestyle="-",
            linewidth=2,
        )

        # Plot the predicted trajectory
        plt.plot(
            range(num_steps),
            predicted_values[sample_idx],
            label=f"Predicted (Sample {sample_idx + 1})",
            color=colors[sample_idx],
            marker="x",
            linestyle="--",
            linewidth=2,
        )

    plt.xlabel("Forecast Step")
    plt.ylabel("Yield (kg)")
    plt.title("Step-Dependent True vs Predicted Values")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Legend adjustments
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize="small", bbox_to_anchor=(1.05, 1))

    # Save or show the plot
    if save_image_path:
        directory = os.path.dirname(save_image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{run_mode}_step_dependent_predictions.png", bbox_inches="tight")
    else:
        plt.show()
    plt.clf()
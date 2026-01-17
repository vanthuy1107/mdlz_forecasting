"""Visualization utilities."""
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Union


def plot_difference(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    save_path: str = "../outputs/prediction_vs_truth.png",
    show: bool = True
):
    """
    Plot comparison of true vs predicted values.
    
    Args:
        y_true: True values (real scale).
        y_pred: Predicted values (real scale).
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    plt.figure(figsize=(20, 6))
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    times = range(len(y_true))
    
    plt.plot(times, y_true, label="True Outbound", marker="o")
    plt.plot(times, y_pred, label="Predicted Outbound", marker="x")
    
    plt.title("Outbound Prediction Daily")
    plt.xlabel("Day")
    plt.ylabel("Outbound Quantity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if needed
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curve(
    train_losses: Union[list, np.ndarray],
    val_losses: Union[list, np.ndarray],
    save_path: str = "../outputs/learning_curve.png",
    show: bool = True
):
    """
    Plot learning curve (train loss vs validation loss).
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='x', label='Validation Loss')
    
    plt.title("Learning Curve (Train vs Validation Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if needed
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


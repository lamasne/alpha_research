from src.config import TrainingConfig
from src.utils import create_new_filename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from pathlib import Path


def add_config_to_plot(fig, run_config: TrainingConfig):
    """ Add a comment at the bottom of the plot with config params """
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # [left, bottom, right, top]
    footer_text = " | ".join(f"{k}: {v}" for k, v in run_config.__dict__.items())
    fig.text(
        0.02, 0.02, f"Training config: {footer_text}",
        fontsize=8,
        fontfamily='monospace',
        ha='left',
        wrap=True,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )


class TrainingTracker:
    def __init__(self, config: TrainingConfig, tracking_path, base_name: str = "train_conv"):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.lrs = []
        self.train_config = config

        # Saving preparation
        tracking_path.mkdir(parents=True, exist_ok=True)
        self.save_path = create_new_filename(tracking_path, base_name, "png")
    
    def add_epoch(self, epoch, train_loss, val_loss, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lrs.append(lr)
    
    def to_dataframe(self):
        return pl.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'lr': self.lrs
        })
    
    def plot_losses(self, show=True):
            """
            Simple plot of train vs validation MSE through epochs
            """

            # Plot curves
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot train loss on left y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train MSE', color=color)
            ax1.plot(self.epochs, self.train_losses, color=color, linewidth=2, label='Train MSE')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for validation loss
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Validation MSE', color=color)
            ax2.plot(self.epochs, self.val_losses, color=color, linewidth=2, 
                    linestyle='--', label='Validation MSE')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Apply log scale to both axes if needed
            for ax, losses in [(ax1, self.train_losses), 
                                    (ax2, self.val_losses)]:
                if len(losses) > 1 and all(l > 0 for l in losses):  # All positive
                    ratio = max(losses) / min(losses)
                    if ratio > 100:
                        ax.set_yscale('log')           

            ax1.set_xticks(np.arange(0, len(self.epochs) + 1, 5))

            plt.title('Evolution of Training and Validation Losses')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            add_config_to_plot(fig, self.train_config)
            
            # Save the plot
            plt.savefig(self.save_path, dpi=100, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()


def plot_grid_search_results(x_label, y_label, data: pd.DataFrame, config: TrainingConfig, tracking_path: Path):
    vars = data.columns
    pivot_table = data.pivot(columns=vars[0], index=vars[1], values=vars[2])

    # Create the heatmap
    fig = plt.figure(figsize=(10, 8))

    # Use imshow for heatmap
    plt.imshow(pivot_table.values, aspect='auto', cmap='jet', 
            # norm=LogNorm(vmin=pivot_table.values.min(), 
            #                 vmax=pivot_table.values.max())
    )

    # Add labels and title
    plt.title('Results of Grid Scan', fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Set custom ticks
    x_ticks = np.arange(len(pivot_table.index))
    y_ticks = np.arange(len(pivot_table.columns))

    plt.xticks(y_ticks, pivot_table.columns)
    plt.yticks(x_ticks, pivot_table.index)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('MSE Loss', fontsize=12)

    # Add text annotations for each cell
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            loss_value = pivot_table.iloc[i, j]
            if not np.isnan(loss_value):
                plt.text(j, i, f'{loss_value:.2e}', 
                        ha='center', va='center', 
                        color='white' if loss_value < pivot_table.values.mean() else 'black',
                        fontsize=9)

    # Add config to plot
    add_config_to_plot(fig, config)

    # Save the plot
    save_path = create_new_filename(tracking_path, "grid_search_results", "png")
    plt.savefig(save_path)


def plot_val_predictions(y_test, y_pred, config, tracking_path, title="Validation Predictions"):
    """
    Scatter plot + Time series plot of predictions vs true values.
    """
    y_test = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, s=5, alpha=0.4)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)  # perfect-pred line
    plt.xlabel("True Next-Day Extrinsic (next bid - next intrinsic)")
    plt.ylabel("Predicted Next-Day Extrinsic")
    plt.title("NN predictions")
    plt.grid(True, alpha=0.3)
    add_config_to_plot(plt.gcf(), config)
    save_path = create_new_filename(tracking_path, "scatter_pred_vs_true", "png")
    plt.savefig(save_path)

    # Take first 200 points for clarity
    n_plot = min(200, len(y_test))
    idx = np.arange(n_plot)
    plt.figure(figsize=(14, 5))
    plt.plot(idx, y_test[:n_plot], 'b-', label='True', alpha=0.7, linewidth=1)
    plt.plot(idx, y_pred[:n_plot], 'r-', label='Predicted', alpha=0.7, linewidth=1)
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    plt.text(0.02, 0.98, f'MSE: {mse:.4f}\nRMSE: ${rmse:.4f}', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Sample Index')
    plt.ylabel('Next-day extrinsic value ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    add_config_to_plot(plt.gcf(), config)
    
    # Save the plot
    save_path = create_new_filename(tracking_path, "pred_vs_true_by_sample", "png")
    plt.savefig(save_path)
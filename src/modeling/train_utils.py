from src.config import tracking_dir, TrainingConfig
from src.utils import add_config_to_plot, create_new_filename
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


class TrainingTracker:
    def __init__(self, config: TrainingConfig, base_name: str = "train_conv"):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.lrs = []
        self.train_config = config

        # Saving preparation
        tracking_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = create_new_filename(tracking_dir, base_name, "png")
    
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


def plot_grid_search_results(data: pd.DataFrame, config: TrainingConfig):
    vars = data.columns
    pivot_table = data.pivot(index=vars[0], columns=vars[1], values=vars[2])
    # pivot_table = data.pivot(index=vars[0], on=vars[1], values=vars[2])

    # Create the heatmap
    fig = plt.figure(figsize=(10, 8))

    # Use imshow for heatmap
    plt.imshow(pivot_table.values, aspect='auto', cmap='viridis', 
            norm=LogNorm(vmin=pivot_table.values.min(), 
                            vmax=pivot_table.values.max()))

    # Add labels and title
    plt.title('Loss as Function of Learning Rate and Log Transform', fontsize=14, pad=20)
    plt.xlabel('Log Transform', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)

    # Set custom ticks
    lr_ticks = np.arange(len(pivot_table.index))
    patience_ticks = np.arange(len(pivot_table.columns))

    plt.xticks(patience_ticks, pivot_table.columns)
    plt.yticks(lr_ticks, [f'{lr:.0e}' for lr in pivot_table.index])

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Loss (log scale)', fontsize=12)

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
    save_path = create_new_filename(tracking_dir, "grid_search_results", "png")
    plt.savefig(save_path)


def plot_val_predictions(y_val, y_pred, config, title="Validation Predictions"):
    """
    Scatter plot + Time series plot of predictions vs true values.
    """

    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_val, y_pred, s=5, alpha=0.4)
    lo = min(y_val.min(), y_pred.min())
    hi = max(y_val.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)  # perfect-pred line
    plt.xlabel("True extrinsic (next-day bid - intrinsic)")
    plt.ylabel("Predicted extrinsic")
    plt.title("NN predictions on validation set")
    plt.grid(True, alpha=0.3)
    add_config_to_plot(plt.gcf(), config)
    save_path = create_new_filename(tracking_dir, "scatter_pred_vs_true", "png")
    plt.savefig(save_path)

    # Take first 200 points for clarity
    n_plot = min(200, len(y_val))
    idx = np.arange(n_plot)
    plt.figure(figsize=(14, 5))
    plt.plot(idx, y_val[:n_plot], 'b-', label='True', alpha=0.7, linewidth=1)
    plt.plot(idx, y_pred[:n_plot], 'r-', label='Predicted', alpha=0.7, linewidth=1)
    mse = np.mean((y_pred - y_val) ** 2)
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
    save_path = create_new_filename(tracking_dir, "pred_vs_true_by_sample", "png")
    plt.savefig(save_path)
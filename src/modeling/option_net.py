from src.config import full_logger, TrainingConfig, outputs_dir
from src.utils import timeit, create_new_filename, create_new_dir
from src.modeling.backtest import run_backtest
from src.data_prep.preprocessing import prepare_data, build_dataloaders
from .train_utils import plot_grid_search_results, plot_val_predictions, TrainingTracker
import itertools
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple
from datetime import datetime
from pathlib import Path

def hyperparam_grid_search(is_run_all_backtests=True):
    """
    Grid search for hyperparameters tuning
    """
    tracking_path = create_new_dir(outputs_dir / "train_tracking", f"{datetime.now().strftime('%Y-%m-%d')}_experiment")

    # Define variables
    lrs = [5e-6, 1e-5]
    is_target_logs = [False]
    is_standardizes = [True]

    ## Grid search
    all_combinations = list(itertools.product(lrs, is_target_logs, is_standardizes))
    nb_combinations = len(all_combinations)
    results = []

    for i, (lr, is_target_log, is_standardize) in enumerate(all_combinations, 1):
        full_logger.info(f"[Hyper-parameters grid scan {i}/{nb_combinations} ({(i/nb_combinations)*100:.0f}%)]: lr={lr}, is_target_log={is_target_log}, log={is_target_log}")
        grid_point = {
            'lr': lr,
            'is_target_log': is_target_log,
            'is_standardize': is_standardize
        }
        config = {**grid_point}

        mse = run(TrainingConfig(**config), tracking_path, is_run_backtest=is_run_all_backtests)
        results.append({**grid_point,'mse': mse})

        full_logger.blank_line()

    full_logger.info(f"Grid search complete: {nb_combinations}/{nb_combinations} configs")

    # Choose 2 dimensions to plot heatmap of loss
    sel_dims = ["is_target_log", "is_standardize"]
    res_df = pd.DataFrame(results)
    plot_grid_search_results(*sel_dims, res_df[sel_dims + ["mse"]], TrainingConfig(**config), tracking_path)


def run(config: TrainingConfig, tracking_path: Path, is_run_backtest=False):
    # Report params
    params_message = ", ".join([f"{k}={v}" for k, v in asdict(config).items()])
    full_logger.info("Running OptionNet model with: " + params_message)

    # Prepare data
    X, y, df = prepare_data(config, is_reporting=False)

    # Split into train and val sets
    fold_loaders, X_test, y_test, test_range = build_dataloaders(X, y, df)
    
    # Train
    model = train_kfold_cv(fold_loaders, X.shape[1], config, tracking_path)

    # Save model
    save_model_and_config(model, config, tracking_path)

    # Predict validation dataset
    y_test, y_pred, mse = predict_option_net(model, X_test, y_test, config)
    plot_val_predictions(y_test, y_pred, config, tracking_path)
    
    # Back testing
    if is_run_backtest:
        run_backtest(df[test_range[0]:test_range[1]+1], y_test, y_pred)

    return mse


def predict_option_net(
    model: torch.nn.Module, 
    X_test: torch.Tensor, 
    y_test: torch.Tensor, 
    config: TrainingConfig
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Calculate predictions on test/validation set.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features as PyTorch tensor
        y_test: Test targets as PyTorch tensor  
        config: Training configuration
    
    Returns:
        y_true: Ground truth as torch.Tensor
        y_pred: Predictions as torch.Tensor
        mse: Mean squared error as float
    """
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Handle log-transform
    if config.is_target_log:
        y_test = torch.expm1(y_test)
        y_pred = torch.expm1(y_pred)
    
    # Calculate metrics
    mse = torch.nn.functional.mse_loss(y_pred, y_test)
    rmse = torch.sqrt(mse)
    mae = torch.nn.functional.l1_loss(y_pred, y_test)
    
    # Log results
    full_logger.info("Test Results:")
    full_logger.info(f"  MSE:  {mse.item():.6f}")
    full_logger.info(f"  RMSE: {rmse.item():.6f}")
    full_logger.info(f"  MAE:  {mae.item():.6f}")
    if config.is_target_log:
        full_logger.info("  (Metrics after inverse log-transform)")
    
    return y_test, y_pred, mse.item()


def save_model_and_config(model, config: TrainingConfig, tracking_path: Path):
    """Save model into pth file and config into json file"""
    model_name = create_new_filename(tracking_path, "model", "pth")
    model_path = tracking_path / model_name
    torch.save(model.state_dict(), model_path)
    # Save config with same basename, .json
    config_path = model_path.with_suffix(".json")
    config.save(config_path)
    full_logger.info(f"Model saved to: {model_path}")
    full_logger.info(f"Config saved to: {config_path}")


@timeit
def train_kfold_cv(fold_loaders, input_dim, config: TrainingConfig, tracking_path: Path):
    """
    Train model using k-fold time series cross-validation.
    
    Args:
        fold_loaders: List of (train_loader, val_loader) for each fold
        config: Training configuration
    
    Returns:
        cv_results: List of results for each fold
        best_model: Model with best validation performance
    """
    cv_results = []
    best_overall_model = None
    best_overall_loss = float('inf')
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        full_logger.info(f"Training Fold {fold_idx + 1}/{len(fold_loaders)}")
        
        # Train on this fold
        fold_result = train_option_net(
            train_loader, 
            val_loader, 
            input_dim, 
            config,
            tracking_path
        )
    
        full_logger.info(f"Training complete. Best val MSE: {fold_result['best_val_loss']:.4e} (epoch {fold_result['best_epoch']})")
        
        cv_results.append(fold_result)
        
        # Track best overall model
        if fold_result['best_val_loss'] < best_overall_loss:
            best_overall_loss = fold_result['best_val_loss']
            best_overall_model = fold_result["model"]
            best_fold_idx = fold_idx + 1
    
    # Report summary
    full_logger.info("--- CROSS-VALIDATION SUMMARY ---")
    
    val_losses = [r['best_val_loss'] for r in cv_results]
    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    
    for i, result in enumerate(cv_results):
        full_logger.info(
            f"Fold {i+1}: Best val MSE = {result['best_val_loss']:.3f} "
            f"(epoch {result['best_epoch']})"
        )
    
    full_logger.info(f"Average validation MSE: {avg_val_loss:.3f} ± {std_val_loss:.3f}")
    full_logger.info(f"Stability Index (for hyperparameters): {avg_val_loss/std_val_loss:.3f}")
    full_logger.info(f"Figure of Merit (avg_val_loss*std_val_loss): {avg_val_loss*std_val_loss:.3f}")
    full_logger.info(f"Best model from fold {best_fold_idx} with MSE = {best_overall_loss:.3f}")
    
    return best_overall_model


def train_option_net(train_loader, val_loader, input_dim, config: TrainingConfig, tracking_path: Path):
    """
    Train model for a single fold in cross-validation.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Number of input features
        config: Training configuration    
    Returns:
        results: Dict with training metrics and metadata
    """
    # Build model
    model = OptionNet(input_dim).to(config.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    if config.is_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs,
            eta_min=config.lr/50
        )
    
    criterion = nn.MSELoss()

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    best_epoch = 0
    tracker = TrainingTracker(config, tracking_path)

    # Training loop
    for epoch in tqdm(range(1, config.epochs + 1), desc="Training", unit="epoch"):
        # ----- TRAINING PHASE -----
        model.train()
        train_loss = 0.0
        n_train = 0
        
        for xb, yb in train_loader:
            # Forward pass
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            batch_size = yb.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size
        
        avg_train_loss = train_loss / n_train if n_train > 0 else 0

        # ----- VALIDATION PHASE -----
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                # Forward pass
                pred = model(xb)
                loss = criterion(pred, yb)
                
                # Accumulate loss
                batch_size = yb.size(0)
                val_loss += loss.item() * batch_size
                n_val += batch_size
        
        avg_val_loss = val_loss / n_val if n_val > 0 else 0
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update training history
        tracker.add_epoch(epoch, avg_train_loss, avg_val_loss, current_lr)
        if epoch % 5 == 0:
            tracker.plot_losses(show=False)

        # ----- EARLY STOPPING CHECK -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch}")
                break
        
        # ----- UPDATE LEARNING RATE -----
        if config.is_lr_scheduler:
            scheduler.step()
        
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Prepare results
    results = {
        'model': model,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_losses': tracker.train_losses,
        'val_losses': tracker.val_losses,
        'learning_rates': tracker.lrs,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'stopped_early': patience_counter >= config.patience
    }
    
    return results


class OptionNet(nn.Module):
    """
    OptionNet is a neural network model for option pricing.
    Architecture: 
    - 2 hidden layers with ReLU activations and Dropout
    - 1 output layer for regression
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,) output


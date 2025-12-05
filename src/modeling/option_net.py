from src.config import ROOT, full_logger, TrainingConfig
from src.utils import convert_to_polars_date, timeit
from .train_utils import plot_grid_search_results, plot_val_predictions, TrainingTracker
import itertools
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np
import pickle
import json
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


def hyperparam_grid_search():
    # Define run params
    base_config = {
        'opt_type': 'puts',
        'is_standardize': False,
        'is_lr_scheduler': True,
        'DET_range': (2, 30),
        'K_dist_pct_max': 5,
        # 'date_range': ("2010-01-01", "2018-12-31")
    }
    lrs = [1e-5, 1e-4]
    patiences = [10]
    log_options = [True, False]

    ## Grid search
    all_combinations = list(itertools.product(patiences, lrs, log_options))
    nb_combinations = len(all_combinations)
    results = []

    for i, (patience, lr, use_log) in enumerate(all_combinations, 1):
        full_logger.info(f"[Hyper-parameters grid scan {i}/{nb_combinations} ({(i/nb_combinations)*100:.0f}%)]: lr={lr}, patience={patience}, log={use_log}")
        grid_point = {
            'lr': lr,
            'patience': patience,
            'is_target_log': use_log
        }
        config = {**base_config,**grid_point}

        _, _, mse = run(TrainingConfig(**config))
        results.append({**grid_point,'mse': mse})        
        
        full_logger.blank_line()

    full_logger.info(f"Grid search complete: {nb_combinations}/{nb_combinations} configs")

    # Choose 2 dimensions to plot heatmap of loss
    sel_dims = ["lr", "is_target_log"]
    res_df = pd.DataFrame(results)
    plot_grid_search_results(res_df[sel_dims + ["mse"]], TrainingConfig(**config))


def run(config: TrainingConfig):
    params_message = "--- Running OptionNet model with " + \
                     ", ".join([f"{k}={v}" for k, v in asdict(config).items()]) + " ---"
    full_logger.info(params_message)
    X, y, feature_cols, dates = prepare_data(config, is_reporting=False)
    model = train_option_net(X, y, feature_cols, config)
    y_val, y_pred, mse = predict_option_net(model, X, y, config)
    plot_val_predictions(y_val, y_pred, config)
    return y_val, y_pred, mse


def predict_option_net(model, X, y, config: TrainingConfig):
    """
    Calculate predictions on validation set. And reverse y's log transform if needed.
    """
    n = len(X)
    n_train = int(config.train_frac * n)
    X_true = torch.tensor(X[n_train:], dtype=torch.float32, device=config.device)
    y_val = y[n_train:]
    model.eval()
    with torch.no_grad():
        y_pred = model(X_true).cpu().numpy()

    # Inverse log-transform if needed
    if config.is_target_log:
        y_val = np.expm1(y_val)
        y_pred = np.expm1(y_pred)
    
    # Report metrics
    mse = np.mean((y_pred - y_val) ** 2)
    full_logger.info(f"Validation MSE {'after log transform inversion' if config.is_target_log else ''}: {mse:.4f}")

    return y_val, y_pred, mse

@timeit
def train_option_net(X, y, feature_cols, config: TrainingConfig):
       
    def check_input_dim(X, feature_cols):
        # Determine input dimension + sanity check
        input_dim = X.shape[1]
        if input_dim != len(feature_cols):
            raise ValueError(f"Input dimension {input_dim} does not match number of feature columns {len(feature_cols)}")
        return input_dim
    
    # Build model, optimizer, criterion
    input_dim = check_input_dim(X, feature_cols)
    model = OptionNet(input_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    if config.is_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs,       # Match your 50 epochs
            eta_min=config.lr/50    # End with very small LR
        )    
    criterion = nn.MSELoss()

    # Split data into train and val sets + split into batches
    train_loader, val_loader = build_dataloaders(X, y, batch_size=256, train_frac=config.train_frac, device=config.device) # shuffling is fine because non-sequential model (e.g. not RNN)

    ## Training loop
    # Early stopping params
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()  # Initialize with current
    # train history
    tracker = TrainingTracker(config)

    # full_logger.info(f"Training...")
    for epoch in tqdm(range(1, config.epochs + 1), desc="Training", unit="epoch"):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            train_loss += loss.item() * bs
            n_train += bs

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                bs = yb.size(0)
                val_loss += loss.item() * bs
                n_val += bs
        val_loss /= n_val

        # Update training history
        current_lr = optimizer.param_groups[0]['lr']
        tracker.add_epoch(epoch, train_loss, val_loss, current_lr)
        if epoch % 5 == 0:
            tracker.plot_losses(show=False)
        # file_logger.info(f"Epoch {epoch:3d} | train MSE {train_loss:.4e} | "
        #              f"val MSE {val_loss:.4e} | LR: {current_lr:.2e}")
            
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Update best
            # tqdm.write(f"✓ New best val loss: {val_loss:.4e}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)  # Restore best
                epoch -= config.patience  # Roll back to best epoch
                break

        # Update LR
        if config.is_lr_scheduler:
            scheduler.step()  

    full_logger.info(f"Training complete. Best val MSE: {best_val_loss:.4e} (epoch {epoch})")

    return model


def build_dataloaders(X, y, batch_size=256, train_frac=0.8, device="cpu"):
    n = len(X)
    n_train = int(train_frac * n)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32, device=device)
    X_val   = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    y_val   = torch.tensor(y[n_train:], dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


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
    


#####################
### PREPROCESSING ###
#####################

@timeit
def prepare_data(
    config: TrainingConfig,
    # opt_type = "puts", # calls or puts
    # DET_range = [2,30],
    # K_dist_max = 0.1,
    # date_range = None,
    # is_standardize=False,
    # is_target_log=False,
    opt_dir = ROOT / "resources/data/dataset1", 
    rf_path  = ROOT / "resources/data/GS1_data.json",
    rv_garch_path = ROOT / "resources/data/rv_garch_h2_train2.pkl",
    is_reporting=True,
):
    """
    Prepare dataset for training option_net: feature engineering + input/output extraction
    Moreover, to facilitate model accuracy, I try and focus on one regime by filtering inputs 
    (i.e. I don't want my model to learn to predict target value in contexts for which I think the market behaves differently)
    """
    # Load data
    df = pl.read_csv(opt_dir / f"{config.opt_type}.csv")

    # Convert UNIX timestamps to dates
    df = df.with_columns([
        pl.from_epoch(pl.col(f"{t}_UNIX"), time_unit="s")
        .cast(pl.Date)
        .alias(f"{t}_DATE")
        for t in ["QUOTE", "EXPIRE"]
    ])

    # Create unique contract ID
    df = df.with_columns([
        (pl.col("EXPIRE_DATE").cast(pl.Utf8) + "_" + pl.col("STRIKE").cast(pl.Utf8))
            .alias("CONTRACT_ID")
            .cast(pl.Categorical)
    ])
    # Sort within each contract by quote date
    df = df.sort(["CONTRACT_ID", "QUOTE_DATE"])

    # Add features + next-day fields per contract
    df = df.with_columns([
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID"),
        (pl.col("ASK") - pl.col("BID")).alias("SPREAD"),
        pl.col("VOLUME").log1p().alias("LOG_VOLUME"),
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).log().alias("LOG_MONEYNESS"),

        # next-day within same CONTRACT_ID
        pl.col("QUOTE_DATE").shift(-1).over("CONTRACT_ID").alias("NEXT_QUOTE_DATE"), # next available quote date
        pl.col("BID").shift(-1).over("CONTRACT_ID").alias("NEXT_BID"),
        pl.col("ASK").shift(-1).over("CONTRACT_ID").alias("NEXT_ASK"),
        pl.col("UNDERLYING_LAST").shift(-1).over("CONTRACT_ID").alias("NEXT_UNDERLYING"),
    ])
    intrinsic = 1 if config.opt_type == "calls" else -1 * (pl.col("NEXT_UNDERLYING") - pl.col("STRIKE"))
    df = df.with_columns(
        (pl.col("NEXT_BID") - intrinsic).alias("NEXT_EXTRINSIC_BID")
    )
    # print(df[["CONTRACT_ID", "QUOTE_DATE", "NEXT_QUOTE_DATE"]])

    # Add risk-free-rate, garch predictions, and get next_trad_date
    df = add_rf(df, rf_path)
    df, anchor_dates = add_rv_garch(df, rv_garch_path, is_reporting=is_reporting)
    anchor_dates = sorted(anchor_dates)
    next_trad_date = {d : anchor_dates[i+1] for i, d in enumerate(anchor_dates[:-1])}

    # Filter out rows where next-trading-day values are not available
    len_bef = len(df)
    df = df.with_columns(
        pl.col("QUOTE_DATE")
        .map_elements(next_trad_date.get, return_dtype=pl.Date)
        .alias("EXPECTED_NEXT_TRADE_DATE")
    )
    # print(df[["QUOTE_DATE", "NEXT_QUOTE_DATE", "EXPECTED_NEXT_TRADE_DATE"]])
    df = df.filter(
        pl.col("NEXT_BID").is_not_null()
        & pl.col("NEXT_UNDERLYING").is_not_null()
        & (pl.col("NEXT_QUOTE_DATE") == pl.col("EXPECTED_NEXT_TRADE_DATE"))
    )
    len_aft = len(df)
    full_logger.info(f"Number of datapoints went from {len_bef:,} to {len_aft:,} (-{(len_bef-len_aft)/len_bef*100:.1f}%) after next-day filter")

    # Filter DTE 
    if config.DET_range is not None: 
        len_bef = len(df)
        df = df.filter(
            (pl.col("DTE") >= config.DET_range[0]) & (pl.col("DTE") <= config.DET_range[1])
        )
        len_aft = len(df)
        full_logger.info(f"Number of datapoints went from {len_bef:,} to {len_aft:,} (-{(len_bef-len_aft)/len_bef*100:.1f}%) after DTE filter")

    # Filter by strike distance
    if config.K_dist_pct_max is not None: 
        len_bef = len(df)
        df = df.filter(pl.col("STRIKE_DISTANCE_PCT").abs() <= config.K_dist_pct_max)
        len_aft = len(df)
        full_logger.info(f"Number of datapoints went from {len_bef:,} to {len_aft:,} (-{(len_bef-len_aft)/len_bef*100:.1f}%) after |K_dist| < {config.K_dist_pct_max:.1f}% filter")

    # Filter by QUOTE_DATE range
    if config.date_range is not None and len(config.date_range)==2:
        len_bef = len(df)
        df = df.filter(
            (pl.col("QUOTE_DATE") >= pl.lit(config.date_range[0]).str.strptime(pl.Date)) &
            (pl.col("QUOTE_DATE") <= pl.lit(config.date_range[1]).str.strptime(pl.Date))
        )
        len_aft = len(df)
        full_logger.info(f"Number of datapoints went from {len_bef:,} to {len_aft:,} (-{(len_bef-len_aft)/len_bef*100:.1f}%) after dates filter")

    # Get NN inputs and outputs + dates
    feature_cols = [
        "STRIKE",
        "UNDERLYING_LAST",
        "MID",
        "SPREAD",
        "DTE",
        "GARCH-1", "GARCH-2",
        "IV",
        "RF",
        "LOG_VOLUME",
        "LOG_MONEYNESS",
        'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO'
    ]
    target_col = "NEXT_EXTRINSIC_BID"
    dates = df.select("QUOTE_DATE").to_numpy().flatten()
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    # Scale features
    if config.is_standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Log-transform target
    if config.is_target_log:
        y = np.log1p(np.maximum(y, 0))  # log1p for numerical stability, clip at 0 to avoid log of negative

    return X, y, feature_cols, dates


@timeit
def add_rv_garch(df: pl.DataFrame, rv_garch_path: Path, horizon: int = 2, is_reporting: bool = True) -> pl.DataFrame:
    """
    Add GARCH(horizon) realized volatility forecasts to df.

    Assumes rv_garch_path is a pickle of:
        segments = [(anchor_t, ts, rvs), ...]
    where:
        - anchor_t: datetime of anchor quote (t)
        - ts:      array of forecast dates (t+1, t+2, ...)
        - rvs:     array of forecast vols for those dates, len == horizon
    """
    # Load GARCH RV predictions
    with open(rv_garch_path, "rb") as f:
        segments = pickle.load(f)

    if is_reporting:
        full_logger.info(f"Number of RV predictions: {len(segments)}")
    
    # Sanity: first/last segment have correct horizon
    if any([len(segments[j][i])!=horizon for i in [1,2] for j in [0,-1]]):
        raise Exception(f"RV input must be of horizon = {horizon}")

    # Sanity: NEXT_QUOTE_DATE matches between underlying and options data
    df = fix_date_mismatches(df, segments, is_reporting=is_reporting)

    ## Add GARCH-i cols to df
    anchor_dates = []
    garch_cols = [[] for _ in range(horizon)]    
    for anchor_t, _, rvs in segments:
        anchor_dates.append(convert_to_polars_date(anchor_t))
        for i in range(horizon):
            garch_cols[i].append(float(rvs[i]))
    garch_df = pl.DataFrame(
        {
            "QUOTE_DATE": anchor_dates,
            **{f"GARCH-{i+1}": garch_cols[i] for i in range(horizon)},
        }
    )
    # Ensure types match: QUOTE_DATE in both dfs are Date
    garch_df = garch_df.with_columns(pl.col("QUOTE_DATE").cast(pl.Date))
    df = df.with_columns(pl.col("QUOTE_DATE").cast(pl.Date))
    # Join GARCH forecasts to main df (all at once)
    df = df.join(garch_df, on="QUOTE_DATE", how="left")

    # Report and drop missing GARCH rows
    if is_reporting:
        col = "GARCH-1"
        non_null = df[col].is_not_null().sum()
        total = len(df)
        full_logger.info("\nNo RV predictions for GARCH training window:")
        full_logger.info(f"{total - non_null} out of {total} rows ({(1 - non_null/total)*100:.2f}%) "
            f"have null value for GARCH and will be removed.")
    df = df.drop_nulls(subset=[f"GARCH-{i+1}" for i in range(horizon)])

    return df, anchor_dates


def fix_date_mismatches(df, segments, is_reporting=True):
    """
    Trust underlying NEXT_QUOTE_DATE over options data NEXT_QUOTE_DATE + report mismatches
    """

    if is_reporting:
        full_logger.info("NEXT_QUOTE_DAY sanity check...")
    mismatches = []
    for anchor_t, ts, _ in segments:
        anchor_date = convert_to_polars_date(anchor_t)
        matching_rows = df.filter(pl.col("QUOTE_DATE").dt.date() == anchor_date)
        total_rows = matching_rows.height
        if total_rows and len(ts) > 0:
            ts_0 = convert_to_polars_date(ts[0])
            matches = matching_rows.filter(
                pl.col("NEXT_QUOTE_DATE").dt.date() == ts_0
            ).height
            if matches != total_rows:
                unique_dates = matching_rows["NEXT_QUOTE_DATE"].unique().sort()
                mismatches.append((anchor_date, ts_0, unique_dates))

    if len(mismatches) == 0:
        if is_reporting:
            full_logger.info("No mismatches found between underlying and options data NEXT_QUOTE_DATE.")
        return df
    
    else:
        if is_reporting:
            nb_segments = len(segments)
            mismatch_count = len(mismatches)
            full_logger.info(f"Mismatches found: {mismatch_count}/{nb_segments} ({mismatch_count/nb_segments*100:.3f}%)")
            for anchor_date, expected, found in mismatches[:5]:
                if len(found)>1:
                    full_logger.info(f"  For {anchor_date.strftime('%Y-%m-%d')}: conflict between next trading day within options data: {found[0]}, {found[1]} (with {len(found)} total options)")
                else:
                    full_logger.info(f"  For {anchor_date.strftime('%Y-%m-%d')}: next trading day expected by underlying (yf underlying): {expected}, expected by options data: {found[0]}")
        
            full_logger.info("Applying corrections to NEXT_QUOTE_DATE in options data to match underlying data...")
            
        # Create mapping from anchor_date to ts_0
        date_map = {anchor_date: ts_0 for anchor_date, ts_0, _ in mismatches}
        # Apply all corrections at once
        df = df.with_columns(
            pl.when(pl.col("QUOTE_DATE").dt.date().is_in(list(date_map.keys())))
            .then(pl.col("QUOTE_DATE").dt.date().replace(date_map))
            .otherwise(pl.col("NEXT_QUOTE_DATE"))
            .alias("NEXT_QUOTE_DATE")
        )

    return df


def add_rf(df, rf_path):
    """
    Add risk free rate to df
    """
    with open(rf_path, "r") as f:
        rfs = json.load(f)

    rf_df = (
        pl.DataFrame({
            "date_str": list(rfs.keys()),
            "rf": list(rfs.values()),
        })
        .with_columns([
            pl.col("date_str")
                .str.strptime(pl.Date, "%Y-%m-%d")
                .dt.strftime("%Y-%m")
                .alias("YM"),
        ])
        .select(["YM", "rf"])
    )

    df = df.with_columns([
        pl.col("QUOTE_DATE")
            .dt.strftime("%Y-%m")
            .alias("YM"),
    ])

    # Join and convert RF to decimal
    df = (
        df.join(rf_df, on=["YM"], how="left")
        .with_columns((pl.col("rf")).alias("RF"))
        .drop(["YM"])
    )

    return df
import pickle
import json
import polars as pl
from src.config import ROOT
from src.utils import convert_to_polars_date, timeit
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def run():
    X, y, feature_cols = prepare_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_option_net(X, y, feature_cols, epochs=50, lr=1e-3, device=device)
    return model


def train_option_net(X, y, feature_cols, epochs=50, lr=1e-3, device="cpu"):
    
    def check_input_dim(X, feature_cols):
        # Determine input dimension + sanity check
        input_dim = X.shape[1]
        if input_dim != len(feature_cols):
            raise ValueError(f"Input dimension {input_dim} does not match number of feature columns {len(feature_cols)}")
        return input_dim
    
    # Build model, optimizer, criterion
    input_dim = check_input_dim(X, feature_cols)
    model = OptionNet(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Split data into train and val sets + split into batches
    train_loader, val_loader = build_dataloaders(X, y, batch_size=256, train_frac=0.8, device=device) # shuffling is fine because non-sequential model (e.g. not RNN)

    for epoch in range(1, epochs + 1):
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

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {train_loss:.4e} | val MSE {val_loss:.4e}")

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
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,) output
    



#####################
### PREPROCESSING ###
#####################

@timeit
def prepare_data(
    opt_type = "puts", # calls or puts
    opt_dir = ROOT / "resources/data/dataset1", 
    file_prefix = "filt",
    rf_path  = ROOT / "resources/data/GS1_data.json",
    rv_garch_path = ROOT / "resources/data/rv_garch_h2_train2.pkl",
):
    """
    Prepare dataset for training option_net: feature engineering + input/output extraction
    """
    # Load data
    df = pl.read_csv(opt_dir / f"{file_prefix}_{opt_type}.csv")

    # Convert dates to Date type
    df = df.with_columns([
        pl.col("QUOTE_DATE").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("EXPIRE_DATE").str.strptime(pl.Date, "%Y-%m-%d"),
    ])

    # Sort within each contract by quote date
    df = df.sort(["CONTRACT_ID", "QUOTE_DATE"])

    # Add features + next-day fields per contract
    df = df.with_columns([
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).alias("MONEYNESS"),
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).log().alias("LOG_MONEYNESS"),
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID"),
        (pl.col("ASK") - pl.col("BID")).alias("SPREAD"),
        pl.col("VOLUME").log1p().alias("LOG_VOLUME"),

        # next-day within same CONTRACT_ID
        pl.col("BID").shift(-1).over("CONTRACT_ID").alias("NEXT_BID"),
        pl.col("UNDERLYING_LAST").shift(-1).over("CONTRACT_ID").alias("NEXT_UNDERLYING"),
        pl.col("QUOTE_DATE").shift(-1).over("CONTRACT_ID").alias("NEXT_QUOTE_DATE"),
    ])

    # print(df[["CONTRACT_ID", "QUOTE_DATE", "NEXT_QUOTE_DATE"]])

    # Filter out rows where next-day values are not available
    len_bef = len(df)
    df = df.filter(
        (
            pl.col("NEXT_BID").is_not_null()
            & pl.col("NEXT_UNDERLYING").is_not_null()
            & (
                pl.col("NEXT_QUOTE_DATE")
                == pl.col("QUOTE_DATE") + pl.duration(days=1)
            )
        )
    )
    len_aft = len(df)
    print(f"len(df) went from {len_bef} to {len_aft} (-{(len_bef-len_aft)/len_bef*100:.2f}%) after next-day filter")

    # Add advanced features 
    df = add_rf(df, rf_path)
    df = add_rv_garch(df, rv_garch_path)

    # Compute next-day intrinsic and target extrinsic value at bid
    if opt_type == "calls":
        df = df.with_columns([
            pl.max_horizontal(pl.lit(0), pl.col("NEXT_UNDERLYING") - pl.col("STRIKE"))
            .alias("NEXT_INTRINSIC"),
        ])
    elif opt_type == "puts":
        df = df.with_columns([
            pl.max_horizontal(pl.lit(0), pl.col("STRIKE") - pl.col("NEXT_UNDERLYING"))
            .alias("NEXT_INTRINSIC"),
        ])

    df = df.with_columns([
        (pl.col("NEXT_BID") - pl.col("NEXT_INTRINSIC"))
        .alias("TARGET_EXTRINSIC_BID"),
    ])
    
    # print(df.select(["CONTRACT_ID", "DTE", "UNDERLYING_LAST", "NEXT_UNDERLYING"]).head(10))
    
    feature_cols = [
        "STRIKE",
        "UNDERLYING_LAST",
        "MID",
        "IV",
        "RF",
        "GARCH-1",
        "GARCH-2",
        "DTE",
        "SPREAD",
        "LOG_VOLUME",
        "LOG_MONEYNESS",
    ]

    target_col = "TARGET_EXTRINSIC_BID"

    # Get NN inputs and outputs
    df = df.select(feature_cols + [target_col])
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    return X, y, feature_cols


def add_rv_garch(df, rv_garch_path, horizon=2):
    """
    Add GARCH realized volatility prediction (previously calculated) to df
    """    
    # Load garch rv predictions
    with open(rv_garch_path, "rb") as f:
        segments = pickle.load(f)

    print(f"Number of RV predictions: {len(segments)}")
    
    # Check first and last segments are the right sizes
    if any([len(segments[j][i])!=horizon for i in [1,2] for j in [0,-1]]):
        raise Exception(f"RV input must be of horizon = {horizon}")

    # Sanity check NEXT_QUOTE_DATE
    df = fix_date_mismatches(df, segments)

    # Add GARCH-i cols and fill them
    for i in range(horizon):
        df = df.with_columns(pl.lit(None).alias(f"GARCH-{i+1}"))
    for anchor_t, _, rvs in segments:
        anchor_date = convert_to_polars_date(anchor_t)
        exprs = []
        for i, rv in enumerate(rvs):
            expr = pl.when(pl.col("QUOTE_DATE").dt.date() == anchor_date) \
                .then(rv) \
                .otherwise(pl.col(f"GARCH-{i+1}")) \
                .alias(f"GARCH-{i+1}")
            exprs.append(expr)

        df = df.with_columns(exprs)

    # Report null GARCH
    col = "GARCH-1"
    non_null = df[col].is_not_null().sum()
    total = len(df)
    print("\nNo RV predictions for GARCH training window:")
    print(f"{total - non_null} out of {total} {(1 - non_null/total)*100:.2f}% of rows have null value for GARCH and will be removed")

    # Eliminate NA GARCH rows
    garch_cols = [f"GARCH-{i}" for i in range(1, horizon+1)]
    df = df.drop_nulls(subset=garch_cols)

    return df


def fix_date_mismatches(df, segments):
    """
    Trust underlying NEXT_QUOTE_DATE over options data NEXT_QUOTE_DATE + report mismatches
    """

    print("\nNEXT_QUOTE_DAY sanity check")
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
        print("No mismatches found between underlying and options NEXT_QUOTE_DATE")
        return df
    
    else:
        nb_segments = len(segments)
        mismatch_count = len(mismatches)
        print(f"Mismatches found: {mismatch_count}/{nb_segments} ({mismatch_count/nb_segments*100:.3f}%)")
        for anchor_date, expected, found in mismatches[:5]:
            if len(found)>1:
                print(f"  For {anchor_date.strftime('%Y-%m-%d')}: conflict between next trading day within options data: {found[0]}, {found[1]} (with {len(found)} total options)")
            else:
                print(f"  For {anchor_date.strftime('%Y-%m-%d')}: next trading day expected by underlying (yf underlying): {expected}, expected by options data: {found[0]}")
        
        # Apply all corrections at once
        print("Applying corrections to NEXT_QUOTE_DATE in options data to match underlying data...")
        # Create mapping from anchor_date to ts_0
        date_map = {anchor_date: ts_0 for anchor_date, ts_0, _ in mismatches}
        
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
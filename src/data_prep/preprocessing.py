import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import os
import json
import pickle
import numpy as np
import scipy.stats as si
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from scipy.optimize import brentq
from src.config import full_logger, TrainingConfig, dataset_dir, dataset_filename, inputs_dir, outputs_dir
from src.utils import timeit, convert_to_polars_date
from src.data_prep.data_aquisition import get_SPY_EOD_df, get_data_from_fred


########################
### DATA PREPARATION ###
########################

rf_path  = inputs_dir / "GS1_data.json"


def build_dataloaders(
    X:torch.Tensor, 
    y:torch.Tensor, 
    df:pl.DataFrame, 
    batch_size:int=256, 
    is_shuffle_training:bool=True
) -> (list, torch.Tensor, torch.Tensor, tuple):
    """
    - Split data into: 3 (training + validation) walk-forward expanding-window folds + test set
    - Split into batches 
    - Optional: Shuffle training data (which is fine since we work with a non-sequential model, e.g. not RNN)
    Args:
    - is_shuffle_training: whether to shuffle training data
    """
    
    dates = df["QUOTE_DATE"].to_numpy()

    # Ensure chronological order
    if not np.all(np.diff(dates) >= 0):
        raise ValueError("Dates are not in chronological order")

    # Define boundaries of each fold + test set
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    year_boundaries = {
        2018: np.where(years <= 2018)[0][-1],
        2019: np.where(years <= 2019)[0][-1],
        2020: np.where(years <= 2020)[0][-1],
        2021: np.where(years <= 2021)[0][-1],
        2022: np.where(years <= 2022)[0][-1]
    }
    folds = [
        # (train_start, train_end, val_start, val_end)
        (0, year_boundaries[2018], year_boundaries[2018]+1, year_boundaries[2019]),
        (0, year_boundaries[2019], year_boundaries[2019]+1, year_boundaries[2020]),
        (0, year_boundaries[2020], year_boundaries[2020]+1, year_boundaries[2021]),
    ]
    test_range = (year_boundaries[2021]+1, year_boundaries[2022])
    full_logger.info(f"Test set: {years[test_range[0]]}-{years[test_range[1]]}")

    # Build loaders for each fold
    fold_loaders = []

    full_logger.info("Spliting Dataset into multiple folds (train + val) and test set...")

    for i, (train_start, train_end, val_start, val_end) in enumerate(folds):
        full_logger.info(f"Fold {i+1} | Training: {years[train_start]}-{years[train_end]}, Validation: {years[val_start]}-{years[val_end]}")
        
        X_train = X[train_start:train_end+1]
        y_train = y[train_start:train_end+1]
        X_val = X[val_start:val_end+1]
        y_val = y[val_start:val_end+1]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=is_shuffle_training
        )
        
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    # Test set
    X_test = X[test_range[0]:test_range[1]+1]
    y_test = y[test_range[0]:test_range[1]+1]
    
    
    return fold_loaders, X_test, y_test, test_range

@timeit
def prepare_data(
    config: TrainingConfig,
    opt_dir = dataset_dir,
    rv_garch_path = outputs_dir / "rv_garch_h2_train2.pkl",
    is_reporting=True,
) -> (torch.Tensor, torch.Tensor, pl.DataFrame):
    """
    Prepare dataset for training option_net: feature engineering + input/output extraction
    Moreover, to facilitate model accuracy, I try and focus on one regime by filtering inputs 
    (i.e. I don't want my model to learn to predict target value in contexts for which I think 
    the market behaves differently, but outputs that don't respect the criteria stay in)
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
    # Sort within each contract by quote date (mini time-series)
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
    df = df.with_columns([
        (pl.col("BID") - intrinsic).alias("EXTRINSIC_BID"),
        (pl.col("ASK") - intrinsic).alias("EXTRINSIC_ASK"),
        (pl.col("NEXT_BID") - intrinsic).alias("NEXT_EXTRINSIC_BID")
    ])
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
    if config.DTE_range is not None: 
        len_bef = len(df)
        df = df.filter(
            (pl.col("DTE") >= config.DTE_range[0]) & (pl.col("DTE") <= config.DTE_range[1])
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


    # Sort data timely for walk-forward training
    df = df.sort("QUOTE_DATE")

    X, y = get_inputs_targets(df, config)

    return X, y, df


def get_inputs_targets(df, config: TrainingConfig):
    """Get NN inputs and outputs + dates"""

    feature_cols = [
        "EXTRINSIC_BID",
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
        "DELTA", "GAMMA", "VEGA", "THETA", "RHO"
    ]
    target_col = "NEXT_EXTRINSIC_BID"

    X = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32, device=config.device)
    y = torch.tensor(df[target_col].to_numpy(), dtype=torch.float32, device=config.device)
    
    if config.is_standardize:
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + 1e-8
        X = (X - X_mean) / X_std
    
    if config.is_target_log:
        y = torch.log1p(torch.clamp(y, min=0))

    assert X.shape[1] == len(feature_cols), "X's first dimension should match feature_cols"

    return X, y


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


######################
# Data preprocessing #
######################

@timeit
def format_opt_data(
    dir: Path = dataset_dir,
    input_filename: str = dataset_filename,
    is_vol_analysis: bool = False,
):
    """
    Preprocess raw options data from Kaggle.
    - Split into calls / puts and normalize column names
    - Keep only columns of interest
    - Filter by volume quantile
    """

    spy_file = dir / input_filename
    df = pl.read_csv(spy_file)
    full_logger.info(f"Initial number of options data points: {df.height:,}")

    # Format col names and keep only cols of interest
    new_cols = [c.strip("[] ").strip() for c in df.columns]
    df = df.rename(dict(zip(df.columns, new_cols)))
    df = df.rename({"QUOTE_UNIXTIME": "QUOTE_UNIX"})
    features_dupl1 = ["VOLUME", "BID", "ASK", "IV", "DELTA", "GAMMA", "VEGA", "THETA", "RHO",]
    features_dupl2 = [f"{t}_{feat}" for feat in features_dupl1 for t in ["C","P"]]
    features_selected = features_dupl2 + [
        "QUOTE_UNIX", "EXPIRE_UNIX", "DTE",
        "UNDERLYING_LAST", "STRIKE", "STRIKE_DISTANCE_PCT",
    ]
    df = df.select(features_selected)
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in features_dupl2])
    df = df.with_columns(
        (pl.col("STRIKE_DISTANCE_PCT") * 100).alias("STRIKE_DISTANCE_PCT")
    )

    # --- Optional: volumme analysis ---
    if is_vol_analysis:
        strike_distance_volume_analysis(df.to_pandas())


    vol_thresh = TrainingConfig().volume_pctl_thresh
    for t in ["C","P"]:
        q = df.select(pl.col(f"{t}_VOLUME").quantile(vol_thresh)).item()
        full_logger.info(f"Volume threshold for {"calls" if t == "C" else "puts"}: {q} ({vol_thresh:.2%} percentile)")
        type_df = (
            df
            .filter(pl.col(f"{t}_VOLUME") > q)
            .select([
                "QUOTE_UNIX", "EXPIRE_UNIX", "DTE",
                "UNDERLYING_LAST", "STRIKE", "STRIKE_DISTANCE_PCT",
                *[
                    pl.col(f"{t}_{feat}").alias(feat)
                    for feat in features_dupl1
                ],
            ])
        )
        if t == "C":
            df_calls = type_df
        else:
            df_puts = type_df

    full_logger.info(f"Number of puts data points after spliting and volume filtering: {df_puts.height:,}")
    full_logger.info(f"Number of calls data points after spliting and volume filtering: {df_calls.height:,}")

    # Save
    df_calls.write_csv(dir / "calls.csv")
    df_puts.write_csv(dir / "puts.csv")

    return df_calls, df_puts



def strike_distance_volume_analysis(df):
    """
    Analyze strike distance - volume relationship
    """
    # 1) bins
    bins = np.linspace(df["STRIKE_DISTANCE_PCT"].min(),
                    df["STRIKE_DISTANCE_PCT"].max(), 20)
    n_bins = len(bins) - 1

    # 2) assign obs to bins (1..n_bins)
    df["sd_bin"] = np.digitize(df["STRIKE_DISTANCE_PCT"], bins)
    df["sd_bin"] = df["sd_bin"].clip(1, n_bins)

    # 3) sum volume per bin and reindex to all bins
    volume_per_bin = df.groupby("sd_bin")["C_VOLUME"].sum()
    volume_per_bin = volume_per_bin.reindex(np.arange(1, n_bins + 1), fill_value=0)

    # 4) x positions = bin centers (length n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure()
    plt.bar(bin_centers,
            volume_per_bin.values,
            width=(bins[1] - bins[0]),
            edgecolor='black')
    plt.xlabel("Strike distance (%)")
    plt.ylabel("Total traded volume")
    plt.yscale("log")
    plt.title("Traded volume per strike distance percentage")
    plt.tight_layout()
    plt.show()


#################
# Sanity checks #
#################
@timeit
def IV_sanity_check(df, opt_type='call', nb_rows=1000, trading_days_per_year=365.0, FRED_API_KEY=None):
    """
    Calculate implied volatility from options dataframe
    """
    # Sample rows
    df = df.to_pandas()
    df = df.sample(n=nb_rows, random_state=0).copy()

    # Acquire risk-free rates
    if not os.path.exists(rf_path):
        rfs = get_data_from_fred(FRED_API_KEY, "GS1", output_dir=inputs_dir)
    else:   
        with open(rf_path, 'r') as file:
            rfs = json.load(file)
    
    # Map quote dates to risk-free rates, and convert to decimal
    rf_series = pd.Series(rfs)
    rf_series.index = pd.to_datetime(rf_series.index).to_period("M")
    df["RF"] = (
        pd.to_datetime(df["QUOTE_UNIX"], unit="s")
        .dt.to_period("M")
        .map(rf_series)
        / 100
    )
    # print(
    #     df.assign(
    #         QUOTE_YM = pd.to_datetime(df["QUOTE_UNIX"], unit="s").dt.strftime("%Y-%m")
    #     )[["QUOTE_YM", "QUOTE_UNIX", "RF"]]
    # )
    def iv_row(row):
        C = (row["BID"] + row["ASK"]) / 2.0
        S = row["UNDERLYING_LAST"]
        K = row["STRIKE"]
        T = row["DTE"] / trading_days_per_year
        rf = row["RF"]

        return calculate_IV(C, S, K, T, rf, opt_type)

    df["MY_IV"] = df.apply(iv_row, axis=1)

    # Error metrics on the sliced df
    df["IV_ERR"] = df["MY_IV"] - df["IV"]
    mean_err = df["IV_ERR"].mean()
    mae = df["IV_ERR"].abs().mean()

    iv_mean = df["IV"].mean()
    mean_err_pct = 100 * mean_err / iv_mean
    mae_pct = 100 * mae / iv_mean

    print(df[["IV", "MY_IV"]].head(5).to_string(
        float_format="{:.2e}".format,
        index=False
    ))
    print(f"Mean error     : {mean_err:.2e} ({mean_err_pct:.2f}%)")
    print(f"Mean abs error : {mae:.2e} ({mae_pct:.2f}%)")

    return df

def calculate_IV(C_market, S, K, T, r, opt_type) -> float:
    """Calculate the implied volatility of a European option using the Black-Scholes model."""
    def objective(sigma):
        return compute_BS_price(S, K, T, r, sigma, opt_type) - C_market
    try:
        return brentq(objective, 1e-6, 5.0, maxiter=1000)
    except ValueError:
        return np.nan  # or handle as needed


def compute_BS_price(S, K, T, r, sigma, opt_type):
    """
    Calculate the Black-Scholes option price.
    inputs:
        S: Current stock price.
        K: Strike price.
        T: Time to expiration (in years).
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying stock (annualized).
        opt_type: Type of option ('call' or 'put'). 
    outputs:
        option_price: Theoretical price of the option.
    """
    if T <= 0:
        raise ValueError("Time to expiration must be greater than zero.")
    if sigma <= 0:
        raise ValueError("Volatility must be greater than zero.")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike price must be greater than zero.")
    if opt_type not in ['call', 'put']:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))    
    d2 = d1 - sigma * np.sqrt(T)

    if opt_type == 'call':
        option_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif opt_type == 'put':
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")

    return option_price

@timeit
def underlying_sanity_check(
    dataset1_path = f"{inputs_dir}/dataset1/SPY Options 2010-2023 EOD.csv",
    dataset2_path = f"{inputs_dir}/dataset2/spy-daily-eod-options-quotes-2020-2022.csv",
    yf_path = f"{inputs_dir}/yf/yf_spy_prices_2020_2022.csv",
):
    """ Compare Kaggle datasets with YFinance data """
    ## Load Kaggle dataset1
    df1 = pd.read_csv(dataset1_path)
    # Format col names and keep only cols of interest
    df1.columns = df1.columns.str.strip("[] ").str.strip()
    # Filter df1's date range
    start = int(pd.Timestamp("2020-01-01").timestamp())
    end   = int(pd.Timestamp("2021-12-31").timestamp())
    df1 = df1[(df1["QUOTE_UNIXTIME"] >= start) & (df1["QUOTE_UNIXTIME"] <= end)]
    # df1.sort_values('QUOTE_UNIXTIME', inplace=True)
    # Format dates
    df1['QUOTE_DATE'] = pd.to_datetime(df1['QUOTE_UNIXTIME'], unit='s').dt.normalize()

    ## Load Kaggle dataset2
    df2 = pd.read_csv(dataset2_path)
    df2.columns = df2.columns.str.strip("[] ").str.strip()
    df2 = df2[(df2["QUOTE_UNIXTIME"] >= start) & (df2["QUOTE_UNIXTIME"] <= end)]
    df2['QUOTE_DATE'] = pd.to_datetime(df2['QUOTE_UNIXTIME'], unit='s').dt.normalize()

    ## Load YFinance data
    if not os.path.exists(yf_path):
        df_yf = get_SPY_EOD_df()
    else:
        df_yf = pd.read_csv(
            yf_path,
            parse_dates=["Date"]
        )
    # Format dates
    df_yf["Date"] = pd.to_datetime(df_yf["Date"], utc=True)
    df_yf["Date"] = df_yf["Date"].dt.tz_convert(None).dt.normalize()    
    df_yf.sort_values('Date', inplace=True)
    # Filter df_yf for the same date range
    start_dt = pd.to_datetime(start, unit='s')
    end_dt   = pd.to_datetime(end,   unit='s')
    df_yf = df_yf[(df_yf["Date"] >= start_dt) & (df_yf["Date"] <= end_dt)]

    # Compare plots
    plt.scatter(df_yf['Date'], df_yf['Close'], s=8, marker='o',
                label='YFinance Close', color='blue', alpha=0.6)
    # plt.scatter(df_yf['Date'], jitter(df_yf['Adj Close']), s=8, marker='x',
    #             label='YFinance Adj Close', color='red', alpha=0.6)
    plt.scatter(df1['QUOTE_DATE'], df1['UNDERLYING_LAST'], s=8, marker='^',
                label='Dataset1', color='orange', alpha=0.6)
    plt.scatter(df2['QUOTE_DATE'], df2['UNDERLYING_LAST'], s=8, marker='s',
                label='Dataset2', color='green', alpha=0.6)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('SPY Close Prices Comparison (2020-2021)')
    plt.legend()
    plt.show()


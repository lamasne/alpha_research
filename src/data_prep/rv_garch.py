import pickle
from src.config import inputs_dir, outputs_dir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model
from src.utils import timeit

@timeit
def predict_rolling_garch(
    file_path = inputs_dir / "yf/yf_spy_prices_2010_2022.csv", 
    horizon = 2, 
    train_size=252*2,
    is_plot=True,
):    
    """
    Returns {horizon}-day forecast of conditional volatility for each time in file {file_path} 
    based on garch model with a rolling window training dataset of size {train_size}
    """

    # Load SPY EOD prices and format dates
    df = pd.read_csv(file_path)
    df = df[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"].dt.tz_convert(None).dt.normalize()
    df.sort_values("Date", inplace=True)

    # Compute returns and realized volatilities
    df["Return"] = 100 * df["Close"].pct_change()
    df = df.dropna(subset=["Return"])
    returns   = df["Return"]
    ret_dates = df["Date"]

    if train_size >= len(ret_dates):
        raise Exception("train_size must be smaller than the dataset")

    # ---------- Forecast ---------
    forecast_segments = []

    # Forecast cond vol at different time points in test data
    for t in range(train_size, len(ret_dates)-horizon):  

        # Fit GARCH(1,1) model to rolling training data - up to the current test day (no look-ahead bias) 
        rol_train_ret = returns[t-train_size:t]
        model = arch_model(rol_train_ret, vol="GARCH", p=1, q=1)
        res = model.fit(disp="off")

        # Get conditional volatility forecast for horizon
        fc = res.forecast(horizon=horizon)
        fc_vols = np.sqrt(fc.variance.values[-1])
        fc_dates = ret_dates[t:t + horizon].to_numpy()  # Dates for forecast
        forecast_segments.append((t-1, fc_dates, fc_vols))

    # Plot forecast_segments
    def plot_GARCH(w=5):
        rv = returns.rolling(window=w).std()
        cv = arch_model(returns, vol="GARCH", p=1, q=1).fit(disp="off").conditional_volatility

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ret_dates, rv, label=f"Realized {w}-day rolling volatility", linewidth=1, color="green")
        ax.plot(ret_dates, cv, label="Cond. volatility (CV) ", linewidth=1, color="blue", linestyle="--")

        step = 5
        for i, (anchor_idx, fc_dates, fc_vols) in enumerate(forecast_segments[::step]):
            anchor_date = ret_dates[anchor_idx]
            anchor_cv   = cv[anchor_idx]
            plot_dates = [anchor_date] + list(fc_dates)
            plot_vols = [anchor_cv] + list(fc_vols)
            ax.plot(
                plot_dates,
                plot_vols,
                marker="*",
                color="red",
                markersize=3,
                label=f"CV anchor + {horizon}-day GARCH" if i == 0 else None,
            )
            
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Volatility")
        ax.grid(True)

    if is_plot:
        plot_GARCH()

    # Save forecast_segments with pickle
    for i, (anchor_idx, fc_dates, fc_vols) in enumerate(forecast_segments):
        forecast_segments[i] = (ret_dates.iloc[anchor_idx], fc_dates, fc_vols)    
    out_path = outputs_dir / f"rv_garch_h{horizon}_train{train_size//252}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(forecast_segments, f)

    plt.show()

    return forecast_segments


@timeit
def rolling_garch_study(
    file_path = inputs_dir / "yf/yf_spy_prices_2020_2022.csv",
    horizon: int = 5,
    rolling_windows = (5, 10, 30),
):
    """
    Fit a rolling-window GARCH(1,1) model on EOD SPY returns (2020–H1 2021) and
    predict next-day volatility. Then plot 'horizon'-day rolling realized
    volatility on the test period (H2 2021) with the 1-day forecast.
    """

    print(f"Starting GARCH rolling forecast study with horizon={horizon} days for different rolling_windows={rolling_windows} days")


    # Load SPY EOD prices and format dates
    df = pd.read_csv(file_path)[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"].dt.tz_convert(None).dt.normalize()
    df.sort_values("Date", inplace=True)

    # Compute returns and realized volatilities
    df["Return"] = 100 * df["Close"].pct_change()
    df = df.dropna(subset=["Return"])
    returns   = df["Return"]
    ret_dates = df["Date"]

    print(f"full range in time is {ret_dates.min().strftime('%Y-%m-%d')} to {ret_dates.max().strftime('%Y-%m-%d')}")

    # real_vols = [returns.rolling(window=rolling_window).std().shift(-rolling_window//2) for rolling_window in rolling_windows]
    real_vols = [returns.rolling(window=w).std() for w in rolling_windows]


    # ---------- Train / test split ----------
    init_train_mask = (ret_dates >= "2020-01-01") & (ret_dates <= "2022-06-30")
    test_mask = (ret_dates >= "2022-07-01") & (ret_dates <= "2022-12-31")
    init_train_ret   = returns[init_train_mask]
    test_ret    = returns[test_mask]

    if init_train_ret.empty or test_ret.empty:
        raise ValueError("Empty train or test segment. Check date ranges and CSV content.")

    # ---------- Forecast ---------
    forecast_segments = []

    # Forecast cond vol at different time points in test data
    nb_steps = 4 # number of rolling forecasts to make
    for t in range(0, len(test_ret) - horizon, len(test_ret)//nb_steps):  

        # Fit GARCH(1,1) model to rolling training data - up to the current test day (no look-ahead bias) 
        rol_train_ret = pd.concat([init_train_ret[t:], test_ret.iloc[:t]])
        if rol_train_ret.empty:
            continue
        model = arch_model(rol_train_ret, vol="GARCH", p=1, q=1)
        res = model.fit(disp="off")

        # Get conditional volatility on training data
        train_vol = res.conditional_volatility

        # Get conditional volatility forecast for horizon
        fc = res.forecast(horizon=horizon)
        fc_vols = np.sqrt(fc.variance.values[-1])
        
        # Combine: train + forecast
        combined_vols = np.concatenate([train_vol, fc_vols])
        
        # Adjust dates to include t-1
        train_dates = ret_dates[rol_train_ret.index]  # Dates for training period
        fc_dates = ret_dates[test_mask].iloc[t:t + horizon].to_numpy()  # Dates for forecast
        combined_dates = np.concatenate([train_dates, fc_dates])
        
        forecast_segments.append((combined_dates, combined_vols))

    # ---------- Plot full returns ----------
    real_vols_cent = [returns.rolling(window=rolling_window).std().shift(-rolling_window//2) for rolling_window in rolling_windows]

    for i, rvs in enumerate([real_vols, real_vols_cent]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Top plot: Returns and realized vol
        ax1.plot(ret_dates, returns, label="Daily returns", linewidth=1)
        ax1.plot(ret_dates, rvs[0], label=f"Realized {rolling_windows[0]}-day rolling volatility", linewidth=1)
        ax1.set_title(f"SPY Daily Returns and {"CENTERED" if i==1 else ""} Realized Volatility")
        ax1.set_ylabel("Return (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Absolute returns and realized vol  
        ax2.plot(ret_dates, returns.abs(), label="Absolute returns", linewidth=1, color='orange')
        for rv, w in zip(rvs, rolling_windows):
            ax2.plot(ret_dates, rv, label=f"{"Centered realized" if i==1 else "Realized"} {w}-day rolling volatility", linewidth=1)
        ax2.set_title(f"Absolute returns for different {"CENTERED" if i==1 else ""} RV rolling windows")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

    # ---------- Plot realized vs forecast vol on test ----------
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ret_dates, real_vols[0], label=f"Realized {rolling_windows[0]}-day rolling volatility", linewidth=1)

    # Plot all forecasts with the same label (only first one gets labeled to avoid duplicates)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, max(1, len(forecast_segments))))
    for i, (dates, vols) in enumerate(forecast_segments):
        if i == 0:
            ax.plot(dates[:-horizon], vols[:-horizon], color=colors[i], alpha=0.3, marker='o', markersize=4,
                    label=f"GARCH(1,1) cond vol: training data")
            ax.plot(dates[-horizon-1:], vols[-horizon-1:], color="red", alpha=0.6, marker='*', markersize=4,
                    label=f"GARCH(1,1) {horizon}-day cond vol: forecast")
        else:
            ax.plot(dates[:-horizon], vols[:-horizon], color=colors[i], alpha=0.3, marker='o', markersize=4)
            ax.plot(dates[-horizon-1:], vols[-horizon-1:], color="red", alpha=0.6, marker='*', markersize=4)
    
    ax.legend()
    ax.set_title(f"GARCH(1,1) {horizon}-day forecasts vs realized volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (% daily)")
    fig2.tight_layout()

    plt.show()
    
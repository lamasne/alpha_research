import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model



def neural_net_model():
    pass



def predict_1day_volatility(file_path="data/yf/yf_spy_prices_2020_2022.csv", horizon = 5, rolling_windows=[5, 10, 30]):
    """
    Fit a GARCH(1,1) model on EOD SPY returns (2020–H1 2021) and
    predict next-day volatility. Then plot 'horizon'-day rolling realized
    volatility on the test period (H2 2021) with the 1-day forecast.
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
    real_vols = [returns.rolling(window=rolling_window).std().shift(-rolling_window//2) for rolling_window in rolling_windows]


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
        train_vol = np.sqrt(res.conditional_volatility)

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top plot: Returns and realized vol
    ax1.plot(ret_dates, returns, label="Daily returns", linewidth=1)
    ax1.plot(ret_dates, real_vols[0], label=f"Realized {rolling_windows[0]}-day rolling volatility", linewidth=1)
    ax1.set_title("SPY Daily Returns and Realized Volatility")
    ax1.set_ylabel("Return (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Absolute returns and realized vol  
    ax2.plot(ret_dates, returns.abs(), label="Absolute returns", linewidth=1, color='orange')
    for real_vol, rolling_window in zip(real_vols, rolling_windows):
        ax2.plot(ret_dates, real_vol, label=f"Realized {rolling_window}-day rolling volatility", linewidth=1)
    ax2.set_title("Absolute Returns vs Realized Volatility")
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
            ax.plot(dates[-horizon-1:], vols[-horizon-1:], color="red", alpha=0.6, marker='o', markersize=4,
                    label=f"GARCH(1,1) {horizon}-day cond vol: forecast")
        else:
            ax.plot(dates[:-horizon], vols[:-horizon], color=colors[i], alpha=0.3, marker='o', markersize=4)
            ax.plot(dates[-horizon-1:], vols[-horizon-1:], color="red", alpha=0.6, marker='o', markersize=4)
    
    ax.legend()
    ax.set_title(f"GARCH(1,1) {horizon}-day forecasts vs realized volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (% daily)")
    fig2.tight_layout()

    plt.show()
    
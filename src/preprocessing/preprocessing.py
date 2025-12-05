import matplotlib.pyplot as plt
import traceback
from curl_cffi import requests as curl_requests
from pathlib import Path
import time
import random
import pandas as pd
import polars as pl
import yfinance as yf
import os
from fredapi import Fred
import json
import numpy as np
import scipy.stats as si

from .yfinance_cookie_patch import patch_yfdata_cookie_basic
from scipy.optimize import brentq
from src.config import ROOT, full_logger, TrainingConfig
from src.utils import timeit


######################
# Data preprocessing #
######################

@timeit
def format_opt_data(
    dir: Path = ROOT / "resources/data/dataset1",
    input_filename: str = "SPY Options 2010-2023 EOD.csv",
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
    rf_path = ROOT/"resources/data/GS1_data.json"
    if not os.path.exists(rf_path):
        rfs = get_data_from_fred(FRED_API_KEY, "GS1", output_dir=ROOT/"resources/data")
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
    objective = lambda sigma: compute_BS_price(S, K, T, r, sigma, opt_type) - C_market
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
def underlying_sanity_check(dir=ROOT/"resources/data"):
    """ Compare Kaggle datasets with YFinance data """
    ## Load Kaggle dataset1
    df1 = pd.read_csv(f"{dir}/dataset1/SPY Options 2010-2023 EOD.csv")

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
    df2 = pd.read_csv(f"{dir}/dataset2/spy-daily-eod-options-quotes-2020-2022.csv")
    df2.columns = df2.columns.str.strip("[] ").str.strip()
    df2 = df2[(df2["QUOTE_UNIXTIME"] >= start) & (df2["QUOTE_UNIXTIME"] <= end)]
    df2['QUOTE_DATE'] = pd.to_datetime(df2['QUOTE_UNIXTIME'], unit='s').dt.normalize()

    ## Load YFinance data
    yf_input_path = f"{dir}/yf/yf_spy_prices_2020_2022.csv"
    if not os.path.exists(yf_input_path):
        df_yf = get_SPY_EOD_df()
    else:
        df_yf = pd.read_csv(
            yf_input_path,
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

    # Compare plose
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


#########################
# SPY Options Scraper  #
#########################

patch_yfdata_cookie_basic()


def get_SPY_EOD_df():
    session = curl_requests.Session(impersonate="chrome")
    tkr = yf.Ticker("SPY", session=session)
    # spy = tkr.history(start="2020-01-01", end="2023-01-01")
    # spy.to_csv(ROOT/"resources/data/yf/yf_spy_prices_2020_2022.csv")
    # return spy

    y1 = "2020"
    y2 = "2023"

    spy = tkr.history(
        start= f"{y1}-01-01",
        end= f"{y2}-01-01",
        auto_adjust=False,
        actions=True
    )

    spy.to_csv(ROOT/f"resources/data/yf/yf_spy_prices_{y1}_{int(y2)-1}.csv")
    return spy


def get_SPY_options_df(output_dir=ROOT/"resources/data", nb_expiries=5, nb_attempts=3):
    # one server connection to reduce cryptographic exchanges and be more human-like
    ticker = "SPY"
    session = curl_requests.Session(impersonate="chrome")

    # Fetch data concurrently as a list of tuples (calls, puts)
    print(f"Fetching options data for {ticker}...")
    start_time = time.perf_counter()

    try:
        tkr = yf.Ticker(ticker, session=session)
        price = tkr.fast_info["last_price"]
        if price is None:
            h = tkr.history(
                period="2d", interval="1d", auto_adjust=True, progress=False
            )
            price = float(h["Close"].iloc[-1])
        print(f"Current price for {ticker} is ${price:.2f}.")

        expiries = tkr.options or []

        if not expiries:
            print(f"No options data for {ticker}.")
            return [], []

        # keep only the nearest since short-term strategy
        expiries = expiries[:nb_expiries]

        # Fetch options data for each expiration date
        calls_list, puts_list = [], []
        for exp in expiries:
            chain = None
            # retries specifically for Yahoo throttling
            for attempt in range(nb_attempts):
                try:
                    chain = tkr.option_chain(exp)  # uses shared session
                    break
                except yf.shared._exceptions.YFRateLimitError:
                    time.sleep(2.0 * (2**attempt))
                except Exception as e:
                    print(f"{ticker} {exp}: option_chain error ({e}).")
                    break
            if chain is None:
                continue  # next expiry
            if (chain.calls is None or chain.calls.empty) and (
                chain.puts is None or chain.puts.empty
            ):
                continue
            try:
                exp_ts = pd.Timestamp(exp).tz_localize("UTC")
                c = chain.calls.copy()
                p = chain.puts.copy()
                for elem in [c, p]:
                    elem["expiration"] = exp_ts
                    elem["price"] = price
                    elem["ticker"] = ticker
                    if "lastTradeDate" in elem:
                        elem["lastTradeDate"] = pd.to_datetime(
                            elem["lastTradeDate"], utc=True, errors="coerce"
                        )
                calls_list.append(c)
                puts_list.append(p)

            except Exception as e:
                print(f"{ticker} {exp}: formatting process error ({e}).")
                continue

            time.sleep(0.4 + random.random() * 0.6)

    except Exception as e:
        print(
            f"Error fetching options data for {ticker}: {e}\n{traceback.format_exc()}"
        )
        return [], []

    finally:
        session.close()

    if not calls_list or not puts_list:
        print("No data fetched.")
        return pd.DataFrame(), pd.DataFrame()

    else:
        elapsed = time.perf_counter() - start_time
        print(f" Data fetched successfully in {elapsed:.2f} seconds.")

    df_calls = pd.concat(calls_list, ignore_index=True)
    df_puts = pd.concat(puts_list, ignore_index=True)

    now = pd.Timestamp.now(tz="UTC")
    for df in (df_calls, df_puts):
        df["remainingDays"] = (df["expiration"] - now).dt.days
        if "lastTradeDate" in df:
            df["duration"] = (df["expiration"] - df["lastTradeDate"]).dt.days
        else:
            df["duration"] = pd.NA

    # Save to JSON
    if not df_calls.empty and not df_puts.empty:
        os.makedirs(output_dir, exist_ok=True)
        df_calls.to_json(
            os.path.join(output_dir, "calls.json"), orient="records", lines=True
        )
        df_puts.to_json(
            os.path.join(output_dir, "puts.json"), orient="records", lines=True
        )
        print("Data saved to JSON files")

    return df_calls, df_puts


###########
# OTHERS  #
###########

# def get_close_finnhub(ticker):
#     """
#     Fetches daily closing prices for a single ticker using Finnhub API.
#     Returns a Pandas Series with dates as index and closing prices as values.
#     """

#     start_ts = int((datetime.today() - timedelta(days=31)).timestamp())
#     end_ts = int((datetime.today() - timedelta(days=1)).timestamp())

#     client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
#     res = client.stock_candles(ticker, 'D', start_ts, end_ts)

#     if res['s'] != 'ok':
#         print(f"No data for {ticker}")
#         return pd.Series(dtype=float)

#     df = pd.DataFrame({'Close': res['c']}, index=pd.to_datetime(res['t'], unit='s'))
#     print(df.head())  # Print first rows for debugging

#     df['Close'].plot(title=f"{ticker} Close Prices", ylabel="Price (USD)", xlabel="Date")
#     plt.show()

#     return df['Close']


def get_data_from_fred(FRED_API_KEY, label, output_dir=ROOT/"resources/data", is_save=True):
    """
    Fetches the data from FRED using the provided API key and data label.
    Saves the data to a JSON file.
    Example, label=GS1 --> saves 1-Year Treasury Constant Maturity Rate (GS1)
    """
    fred = Fred(api_key=FRED_API_KEY)

    rfs = fred.get_series(label)

    # Convert to dictionary with date keys as strings
    data_dict = {str(date.date()): float(value) if value is not None else None
                for date, value in rfs.items()}
    
    # Save to JSON
    file_path = os.path.join(output_dir, f'{label}_data.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print(f"{label} data saved to {file_path}")
    
    return 1


def get_sp500_tickers(output_dir=ROOT/"resources/data"):
    """
    Fetches the S&P 500 tickers from Wikipedia and saves them to a JSON file.
    """
    # Fetch S&P 500 tickers from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    tickers = table['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]  # Replace '.' with '-' for Yahoo Finance compatibility
    print("S&P 500 tickers fetched successfully.")

    # Save tickers to JSON file
    file_path = os.path.join(output_dir, 'tickers.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(tickers, f)
    print(f"S&P 500 tickers saved to {file_path}")

    return 1
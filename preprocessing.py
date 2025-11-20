import matplotlib.pyplot as plt
import traceback
from curl_cffi import requests as curl_requests
from yfinance_cookie_patch import patch_yfdata_cookie_basic
import time
import random
import pandas as pd
import yfinance as yf
import os
from fredapi import Fred
import json
import numpy as np
import scipy.stats as si
from scipy.optimize import brentq


######################
# Data preprocessing #
######################

def format_opt_df(dir="data/dataset1", input_filename="SPY Options 2010-2023 EOD.csv"):
    """
    Format raw options data from Kaggle
    """
    spy_file = f"{dir}/{input_filename}"
    df = pd.read_csv(spy_file)
    
    # Format col names and keep only cols of interest
    df.columns = df.columns.str.strip("[] ").str.strip()

    df.rename(columns={"QUOTE_UNIXTIME": "QUOTE_UNIX"}, inplace=True)
    features_of_interest = [
        'QUOTE_UNIX', 'EXPIRE_UNIX', 'DTE', 
        'UNDERLYING_LAST', 'STRIKE', 'STRIKE_DISTANCE_PCT',
        'C_VOLUME', 'C_BID', 'C_ASK', 'C_IV',
        'P_VOLUME', 'P_BID', 'P_ASK', 'P_IV'
    ]
    df = df[features_of_interest]

    # Convert volumes, bid and ask columns to numeric
    for col in ['C_VOLUME', 'C_BID', 'C_ASK', 'C_IV', 'P_VOLUME', 'P_BID', 'P_ASK', 'P_IV']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter via volume threshold, split into calls and puts, and format col names for consistency
    # # histogram of C_VOLUME (identified as negative binomial)
    # fig = px.density_heatmap(df, nbinsx=20, nbinsy=20, x="C_VOLUME", y="STRIKE_DISTANCE_PCT", marginal_x="histogram", marginal_y="histogram")
    df_calls = (
        df.loc[df["C_VOLUME"] > df["C_VOLUME"].quantile(0.95)]
        .drop(columns=["P_VOLUME", "P_BID", "P_ASK", "P_IV"])
        .rename(columns={"C_VOLUME": "VOLUME", "C_BID": "BID", "C_ASK": "ASK", "C_IV": "IV"})
        .copy()
    )
    df_puts = (
        df.loc[df["P_VOLUME"] > df["P_VOLUME"].quantile(0.95)]
        .drop(columns=["C_VOLUME", "C_BID", "C_ASK", "C_IV"])
        .rename(columns={"P_VOLUME": "VOLUME", "P_BID": "BID", "P_ASK": "ASK", "P_IV": "IV"})
        .copy()
    )

    # Save
    df_calls.to_csv(f"{dir}/calls.csv", index=False)
    df_puts.to_csv(f"{dir}/puts.csv", index=False)

    return df_calls, df_puts


def filt_opt_df(dir="data/dataset1", filename="calls.csv", start_date="2020-01-01", end_date="2021-12-31"):
    """
    Prepare training set
    """
    input_path = f"{dir}/{filename}"
    df = pd.read_csv(input_path)

    # Select DTE between 2 and 10 days
    df = df[(df["DTE"] >= 2) & (df["DTE"] <= 10)]

    # Select strike distance pct between -5% and 5%
    df = df[abs(df["STRIKE_DISTANCE_PCT"]) <= 0.05]

    # Select data that is between 2020 and 2022 in unix
    start = int(pd.Timestamp(start_date).timestamp())
    end   = int(pd.Timestamp(end_date).timestamp())
    df = df[(df["QUOTE_UNIX"] >= start) & (df["QUOTE_UNIX"] <= end)]

    # Sort by EXPIRE_UNIX
    df.sort_values("QUOTE_UNIX", inplace=True)

    # Format dates
    for feature in ["QUOTE", "EXPIRE"]:
        df[feature + "_DATE"] = (
            pd.to_datetime(df[feature + "_UNIX"], unit="s")
            .dt.normalize()
        )
        df.drop(feature + "_UNIX", axis=1, inplace=True)

    # Reorder df
    df = df[['QUOTE_DATE', 'EXPIRE_DATE', 'DTE', 'UNDERLYING_LAST', 'STRIKE', 'STRIKE_DISTANCE_PCT','VOLUME', 'BID', 'ASK', 'IV']]

    df.to_csv(f"{dir}/filt_{filename}", index=False)

    return df


#################
# Sanity checks #
#################

def IV_sanity_check(df, opt_type='call', nb_rows=20, trading_days_per_year=365.0, FRED_API_KEY=None):
    """
    Calculate implied volatility from options dataframe
    """
    df = df.head(nb_rows).copy()

    # Acquire risk-free rates
    rf_path = 'data/GS1_data.json'
    if not os.path.exists(rf_path):
        rfs = get_data_from_fred(FRED_API_KEY, "GS1", output_dir='data')
    else:   
        with open(rf_path, 'r') as file:
            rfs = json.load(file)
    
    # Map quote dates to risk-free rates, and convert to decimal
    rf_series = pd.Series(rfs)
    rf_series.index = pd.to_datetime(rf_series.index).to_period("M")
    df["RF"] = df["QUOTE_DATE"].dt.to_period("M").map(rf_series) / 100 

    def iv_row(row):
        C = (row["BID"] + row["ASK"]) / 2.0
        S = row["UNDERLYING_LAST"]
        K = row["STRIKE"]
        T = row["DTE"] / trading_days_per_year
        rf = row["RF"]

        return calculate_IV(C, S, K, T, rf, opt_type)

    df["MY_IV"] = df.apply(iv_row, axis=1)

    print(df[["IV", "MY_IV"]])

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


def underlying_sanity_check(dir="data"):
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
    yf_input_path = f"{dir}/yf/yf_spy_prices_2020_2022_no_adj.csv"
    if not os.path.exists(yf_input_path):
        df_yf = get_SPY_EOD_df()
    else:
        df_yf = pd.read_csv(
            "data/yf/yf_spy_prices_2020_2022_no_adj.csv",
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
    # spy.to_csv("data/yf/yf_spy_prices_2020_2022.csv")
    # return spy

    spy = tkr.history(
        start="2020-01-01",
        end="2023-01-01",
        auto_adjust=False,
        actions=True
    )

    spy.to_csv("data/yf/yf_spy_prices_2020_2022_no_adj.csv")
    return spy


def get_SPY_options_df(output_dir="data", nb_expiries=5, nb_attempts=3):
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


def get_data_from_fred(FRED_API_KEY, label, output_dir='data', is_save=True):
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


def get_sp500_tickers(output_dir='data'):
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
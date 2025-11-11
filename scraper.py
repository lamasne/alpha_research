from functools import partial
from curl_cffi import requests
import time, random

import finnhub
import pandas as pd
from fredapi import Fred
import json
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import itertools
import os
import time
import matplotlib.pyplot as plt
import pickle


def get_adj_closed_prices(ticker = "AAPL"):
    """
    Fetch adjusted daily closing prices for a single ticker.
    """
    # close_prices = get_close_finnhub(ticker)
    close_prices = get_close_yfinance(ticker)

    return close_prices


def get_close_finnhub(ticker):
    """
    Fetches daily closing prices for a single ticker using Finnhub API.
    Returns a Pandas Series with dates as index and closing prices as values.
    """
    start_ts = int((datetime.today() - timedelta(days=31)).timestamp())
    end_ts = int((datetime.today() - timedelta(days=1)).timestamp())

    client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
    res = client.stock_candles(ticker, 'D', start_ts, end_ts)

    if res['s'] != 'ok':
        print(f"No data for {ticker}")
        return pd.Series(dtype=float)

    df = pd.DataFrame({'Close': res['c']}, index=pd.to_datetime(res['t'], unit='s'))
    print(df.head())  # Print first rows for debugging

    df['Close'].plot(title=f"{ticker} Close Prices", ylabel="Price (USD)", xlabel="Date")
    plt.show()

    return df['Close']


def get_close_yfinance(ticker):
    """
    Fetches daily adjusted closing prices for a single ticker using Yahoo Finance API.
    Returns a Pandas Series with dates as index and closing prices as values.
    """
    try:
        data = yf.download(ticker, period="1mo", interval='1d', auto_adjust=True)
    except Exception as e:
        print(f"Download failed: {e}")
        return pd.Series(dtype=float)

    if data.empty:
        print(f"No data for {ticker}. Skipping...")
        return pd.Series(dtype=float)

    if "Adj Close" in data.columns:
        data = data['Adj Close']
    else:
        print(f"No 'Adj Close' data for {ticker}. Skipping...")
        return pd.Series(dtype=float)

    data = data.dropna()
    print(f"Fetched {len(data)} data points for {ticker}.")

    data.plot(title=f"{ticker} Adjusted Close Prices", ylabel="Price (USD)", xlabel="Date")
    plt.show()

    return data


def data_acquisition(is_scrape, output_dir, FRED_API_KEY, max_n_tickers=None):
    """
    This function will scrape options data for "max_n_tickers" first tickers of S&P 500 and retrieve risk-free rates,
    """
    print("Starting data acquisition...")
    # Get options data for all tickers of SP500
    if is_scrape:
        with open(f'{output_dir}/tickers.json', 'r') as file:
            tickers = json.load(file)
        if max_n_tickers is not None:
            tickers = tickers[:max_n_tickers]
        df_calls, df_puts = get_all_options_data(tickers, output_dir, is_debugging=True)
    else:
        df_calls = pd.read_json(f'{output_dir}/calls.json', orient='records', lines=True)
        df_puts = pd.read_json(f'{output_dir}/puts.json', orient='records', lines=True)

    # Get risk free rate as last value given by FRED API
    if is_scrape:
        risk_free_rates = get_data_from_fred(FRED_API_KEY, "GS1", output_dir)
    with open(f'{output_dir}/GS1_data.json', 'r') as file:
        risk_free_rates = json.load(file)
    # plot_time_series("GS1", "1-Year Treasury Yield (GS1) Over Time")
    risk_free_rate = risk_free_rates[list(risk_free_rates.keys())[-1]]
    risk_free_rate = risk_free_rate / 100  # Convert percentage to decimal

    print("Data acquisition complete.")
    return df_calls, df_puts, risk_free_rate


def get_all_options_data(tickers, output_dir='data', max_workers = 1, is_debugging = False):
    """
    Fetches options data for a list of tickers using Yahoo Finance API.
    Uses multithreading to speed up the process.
    Returns: (DataFrame of calls, DataFrame of puts)
    """
    if is_debugging and os.path.exists(os.path.join(output_dir,"calls.pkl")) and os.path.exists(os.path.join(output_dir,"puts.pkl")):
        with open(os.path.join(output_dir,"calls.pkl"), "rb") as f:
            calls = pickle.load(f)
        with open(os.path.join(output_dir,"puts.pkl"), "rb") as f:
            puts = pickle.load(f)
        return calls, puts

    else:
        # one server connection to reduce cryptographic exchanges and be more human-like
        session = requests.Session(impersonate="chrome")

        # Fetch data concurrently as a list of tuples (calls, puts)
        print(f"Fetching data for {len(tickers)} tickers...")
        start_time = time.perf_counter()
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                worker = partial(get_options_data, session=session)
                results = list(ex.map(worker, tickers))
        finally:
            session.close()

        if not results or all(r==( [], [] ) for r in results):
            print("No data fetched.")
            return [], []
        else:
            elapsed = time.perf_counter() - start_time
            print(f" Data fetched successfully in {elapsed:.2f} seconds.")

        ## Unpack the results into calls and puts
        # If only one ticker, unpack it directly
        if len(results) == 1:    
            calls, puts = results[0]
        # If multiple tickers, combine their options data
        else:
            calls = list(itertools.chain.from_iterable(result[0] for result in results))
            puts = list(itertools.chain.from_iterable(result[1] for result in results))

        if is_debugging:
            # Save results with pickle
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir,"calls.pkl"), "wb") as f: pickle.dump(calls, f)
            with open(os.path.join(output_dir,"puts.pkl"),  "wb") as f: pickle.dump(puts,  f)

    df_calls = generate_options_df(calls)
    df_puts = generate_options_df(puts)

    # Save the data to JSON files
    print("Saving data...", end='')
    os.makedirs(output_dir, exist_ok=True)
    df_calls.to_json(os.path.join(output_dir, 'calls.json'), orient='records', lines=True, date_format='iso')
    df_puts.to_json(os.path.join(output_dir, 'puts.json'), orient='records', lines=True, date_format='iso')
    print(" Done.")

    return df_calls, df_puts


def get_options_data(ticker, session, nb_expiries = 5, nb_attempts = 3):
    """
    Fetches options data for a single ticker using Yahoo Finance API.
    Returns: (DataFrame of calls, DataFrame of puts)
    Each DataFrame contains a list of lists of dictionaries containing options data and grouped by expiration dates.
    """
    tkr = yf.Ticker(ticker, session=session)

    # Check if the ticker has options data
    try:
        expiries = tkr.options or []
    except Exception as e:
        print(f"Error fetching options data for {ticker} - ({e}).")
        return [], []

    if not expiries:
        print(f"No options data for {ticker}.")
        return [], []

    # get price of ticker
    try:
        try:
            price = tkr.fast_info["last_price"]
        except Exception:
            h = tkr.history(period="2d", interval="1d", auto_adjust=True, progress=False)
            price = float(h["Close"].iloc[-1])
    except Exception as e:
        print(f"{ticker}: failed to get last price ({e}).")
        return [], []

    # keep only the nearest since short-term strategy
    expiries = expiries[:nb_expiries]

    # Fetch options data for each expiration date
    calls, puts = [], []
    for exp in expiries:
        chain = None
        # retries specifically for Yahoo throttling
        for attempt in range(nb_attempts):
            try:
                chain = tkr.option_chain(exp)  # uses shared session
                # price = chain.underlying['regularMarketPrice']

                break
            except yf.shared._exceptions.YFRateLimitError:
                time.sleep(2.0 * (2 ** attempt))   # 2,4,8,16,32,64s
            except Exception as e:
                print(f"{ticker} {exp}: option_chain error ({e}).")
                break
        if chain is None:
            continue  # next expiry

        if (chain.calls is None or chain.calls.empty) and (chain.puts is None or chain.puts.empty):
            continue

        try:
            calls.append(_options_df_to_dict(chain.calls, exp, price, ticker))
            puts.append(_options_df_to_dict(chain.puts, exp, price, ticker))
        except Exception as e:
            print(f"{ticker} {exp}: post-process error ({e}).")
            continue

        _sleep_jitter()  # only after a successful fetch

    return calls, puts


def _sleep_jitter(base=0.4):
    """
    Sleep for a random amount of time between 0.4 and 1 second to avoid being blocked by Yahoo Finance API.
    """
    time.sleep(base + random.random()*0.6)


def _options_df_to_dict(df, exp_date, price, ticker):
    """
    Add expiration date and price to the DataFrame, remove timezone info, and convert it to a list of dictionaries.
    """
    if df is None or df.empty:
        return []
    exp_dt = pd.Timestamp(exp_date).tz_localize("UTC")
    df = df.assign(expiration=exp_dt, price=price, ticker=ticker)

    if "lastTradeDate" in df.columns:
        df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], utc=True)
    return df.to_dict(orient='records')


def generate_options_df(option_groups):
    """
    Generates a DataFrame gathering all options data, and calculates the remaining days to expiration and duration.
    """
    # Flatten the list of lists of dictionaries into a single list of dictionaries
    opts_flat = list(itertools.chain.from_iterable(option_groups))

    if not opts_flat:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(opts_flat)

    df["remainingDays"] = (df["expiration"] - pd.Timestamp.now(tz="UTC")).dt.days

    # Needed for BS pricing
    if "lastTradeDate" in df.columns:
        df["duration"] = (df["expiration"] - df["lastTradeDate"]).dt.days
    else:
        df["duration"] = pd.NA
    return df


def get_data_from_fred(FRED_API_KEY, label, output_dir='data'):
    """
    Fetches the data from FRED using the provided API key and data label.
    Saves the data to a JSON file.
    Example, label=GS1 --> saves 1-Year Treasury Constant Maturity Rate (GS1)
    """
    fred = Fred(api_key=FRED_API_KEY)

    risk_free_rate = fred.get_series(label)
    
    # Convert to dictionary with date keys as strings
    data_dict = {str(date.date()): float(value) if value is not None else None
                for date, value in risk_free_rate.items()}
    
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
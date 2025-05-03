import pandas as pd
from fredapi import Fred
import json
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import itertools
import os
import time


def get_all_options_data(tickers, output_dir='data'):
    """
    Fetches options data for a list of tickers using Yahoo Finance API.
    Uses multithreading to speed up the process.
    Returns: (DataFrame of calls, DataFrame of puts)
    """
    # Fetch data concurrently as a list of tuples (calls, puts)
    print(f"Fetching data for {len(tickers)} tickers...")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=100) as executor:
         results = list(executor.map(get_options_data, tickers))
    
    elapsed = time.perf_counter() - start_time

    if results is None:
        print("No data fetched.")
        return [], []
    else:
        print(f" Data fetched successfully in {elapsed:.2f} seconds.")
    
    # Unpack the results into calls and puts
    if len(results) == 1:
        # If only one ticker, unpack it directly
        calls, puts = results[0]
    else:
        # If multiple tickers, combine their options data
        calls = list(itertools.chain.from_iterable(result[0] for result in results))
        puts = list(itertools.chain.from_iterable(result[1] for result in results))

    df_calls = generate_options_df(calls)
    df_puts = generate_options_df(puts)

    # Save the data to JSON files
    print("Saving data...", end='')
    os.makedirs(output_dir, exist_ok=True)
    df_calls.to_json(os.path.join(output_dir, 'calls.json'), orient='records', lines=True, date_format='iso')
    df_puts.to_json(os.path.join(output_dir, 'puts.json'), orient='records', lines=True, date_format='iso')
    print(" Done.")

    return df_calls, df_puts


def get_options_data(ticker):
    """
    Fetches options data for a single ticker using Yahoo Finance API.
    Returns: (DataFrame of calls, DataFrame of puts)
    Each DataFrame contains a list of lists of dictionaries containing options data and grouped by expiration dates.
    """
    stock = yf.Ticker(ticker)

    # Check if the ticker has options data
    if not stock.options:
        print(f"No options data for {ticker}.")
        return [], []
        
    # Fetch options data for each expiration date
    calls, puts = [], []
    for exp in stock.options:
        opt = stock.option_chain(exp)
        price = opt.underlying['regularMarketPrice']
        calls.append(_options_df_to_dict(opt.calls, exp, price))
        puts.append(_options_df_to_dict(opt.puts, exp, price))
    return calls, puts



def _options_df_to_dict(df, exp_date, price):
    """
    Add expiration date and price to the DataFrame, remove timezone info, and convert it to a list of dictionaries.
    """
    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
    df = df.assign(expiration=exp_dt, price=price)
    df['lastTradeDate'] = df['lastTradeDate'].dt.tz_localize(None)
    return df.to_dict(orient='records')


def generate_options_df(option_groups):
    """
    Generates a DataFrame gathering all options data, and calculates the remaining days to expiration and duration.
    """
    # Flatten the list of lists of dictionaries into a single list of dictionaries
    opts_flat = list(itertools.chain.from_iterable(option_groups))
    df = pd.DataFrame.from_records(opts_flat)
    df['remainingDays'] = (df['expiration'] - datetime.now()).dt.days
    df['duration'] = (df['expiration'] - df['lastTradeDate']).dt.days
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
from src.config import inputs_dir, outputs_dir
import traceback
from curl_cffi import requests as curl_requests
import time
import random
import yfinance as yf
from fredapi import Fred
from .yfinance_cookie_patch import patch_yfdata_cookie_basic
import pandas as pd
import os
import json

#########################
# SPY Options Scraper  #
#########################

patch_yfdata_cookie_basic()


def get_SPY_EOD_df(y1:str = "2020", y2:str = "2023"):
    session = curl_requests.Session(impersonate="chrome")
    tkr = yf.Ticker("SPY", session=session)
    spy = tkr.history(
        start= f"{y1}-01-01",
        end= f"{y2}-01-01",
        auto_adjust=False,
        actions=True
    )
    spy.to_csv(inputs_dir / f"yf/yf_spy_prices_{y1}_{int(y2)-1}.csv")
    return spy


def get_SPY_options_df(nb_expiries=5, nb_attempts=3):
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
        os.makedirs(outputs_dir, exist_ok=True)
        df_calls.to_json(
            os.path.join(outputs_dir, "calls.json"), orient="records", lines=True
        )
        df_puts.to_json(
            os.path.join(outputs_dir, "puts.json"), orient="records", lines=True
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


def get_data_from_fred(FRED_API_KEY, label, is_save=True):
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
    if is_save:
        file_path = os.path.join(outputs_dir, f'{label}_data.json')
        os.makedirs(outputs_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        print(f"{label} data saved to {file_path}")
    
    return 1


def get_sp500_tickers():
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
    file_path = os.path.join(outputs_dir, 'tickers.json')
    os.makedirs(outputs_dir, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(tickers, f)
    print(f"S&P 500 tickers saved to {file_path}")

    return 1
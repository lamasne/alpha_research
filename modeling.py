import numpy as np
import scipy.stats as si
import pandas as pd
from scipy.optimize import brentq
from arch import arch_model


def predict_volatility(adj_close_prices):
    """
    Predicts the implied volatility of a list of stock tickers using GARCH model.
    Trains on daily adjusted close prices for 5 years by default.
    """
    if adj_close_prices.empty:
        raise ValueError("No adjusted close prices provided. Please provide a valid DataFrame.")
    
    # Compute daily returns as percentage changes
    returns = 100 * adj_close_prices.pct_change().dropna()

    # Fit a GARCH(1,1) model to the returns
    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit()

    # Forecast the volatility for the next day
    forecast = res.forecast(horizon=1) # horizon=1 means we want the next day's forecast
    iv = forecast.variance.values[-1, 0]**0.5  # implied volatility

    return iv

def check_bs_price(df, r):
    """
    Check if BS price, based on yfinance IV, match market price for each option.
    inputs:
        df: DataFrame containing all options data.
        r: Risk-free rate.
    outputs:
        DataFrame with BS price and market price for each option.
    """
    # Filter data
    df_filt = df.copy()
    # Useless columns
    excluded_cols = ['lastTradeDate', 'contractSize', 'currency', 'expiration', 'remainingDays']
    df_filt = df_filt.drop(columns=[col for col in excluded_cols if col in df_filt.columns])
    # Options with virtually zero IV or no bid/ask
    df_filt = df_filt[
        (df_filt['impliedVolatility'] > 0.01) &
        (df_filt['bid'] > 0.1) &
        (df_filt['ask'] > 0.1) &
        (df_filt['lastPrice'] > 0.1) &
        (df_filt['duration'] > 0)
    ]
    # Options with no NaN values
    df_filt = df_filt.dropna()

    # Get a sample of 10 options for testing
    df_test = df_filt.sample(n=10, random_state=42)
    print(df_test)
    
    bs_prices = []
    for _, option in df_test.iterrows():
        # Extract option data
        S = option['price']  # Current stock price
        K = option['strike']  # Strike price
        T = option['duration'] / 365  # Time to expiration from last trade (years)
        sigma = option['impliedVolatility']  # Implied volatility
        market_price = option['lastPrice']  # Last market price of option

        # Calculate Black-Scholes price
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type='call')
        bs_prices.append({
            'Contract_Symbol': option['contractSymbol'], 
            'BS_Price(IV)': bs_price, 
            'Market_Price': market_price, 
            'IV': sigma, 
            'BS_IV(Market_Price)': implied_volatility(market_price, S, K, T, r, option_type='call')
        })

    return pd.DataFrame(bs_prices)
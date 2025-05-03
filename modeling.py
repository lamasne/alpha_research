import numpy as np
import scipy.stats as si
import pandas as pd


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes call option price.
    inputs:
        S: Current stock price.
        K: Strike price.
        T: Time to expiration (in years).
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying stock (annualized).
    outputs:
        call_price: Theoretical price of the call option.
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))    
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate call option price using the Black-Scholes formula
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

def check_bs_price(df, r):
    """
    Check if BS price matches market price for each option.
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
        bs_price = black_scholes_call(S, K, T, r, sigma)
        bs_prices.append({'contractSymbol': option['contractSymbol'], 'BS_Price': bs_price, 'Market_Price': market_price})

    return pd.DataFrame(bs_prices)
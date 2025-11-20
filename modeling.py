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

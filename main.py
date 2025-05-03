import pandas as pd
import os
from dotenv import load_dotenv
from scraper import get_all_options_data, get_data_from_fred
import json
from modeling import check_bs_price

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")    
output_dir = "data"
is_scrape = False # Set to True to scrape data again

if __name__ == "__main__":   
    # Get options data for all tickers of SP500
    if is_scrape:
        with open(f'{output_dir}/tickers.json', 'r') as file:
            tickers = json.load(file)
        df_calls, df_puts = get_all_options_data(tickers, output_dir)
    else:
        df_calls = pd.read_json(f'{output_dir}/calls.json', orient='records', lines=True)

    # Get risk free rate as last value given by FRED API
    if is_scrape:
        risk_free_rates = get_data_from_fred(FRED_API_KEY, "GS1", output_dir)
    with open(f'{output_dir}/GS1_data.json', 'r') as file:
        risk_free_rates = json.load(file)
    # plot_time_series("GS1", "1-Year Treasury Yield (GS1) Over Time")
    risk_free_rate = risk_free_rates[list(risk_free_rates.keys())[-1]]
    risk_free_rate = risk_free_rate / 100  # Convert percentage to decimal


    # Check implementation of Black-Scholes model by comparing BS price with market price
    df_comparison = check_bs_price(df_calls, risk_free_rate) # first row for testing
    print(df_comparison)

    # Previously implemented in the scraper
    
    print("Data visualization complete.")
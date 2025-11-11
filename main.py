import os
from dotenv import load_dotenv
from scraper import get_adj_closed_prices, data_acquisition
from modeling import check_bs_price


load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")   
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") 
output_dir = "data"
is_scrape = True # Set to True to scrape data again
max_n_tickers=5

def iv_modeling(is_scrape, output_dir, FRED_API_KEY, max_n_tickers):
    df_calls, _, _ = data_acquisition(is_scrape, output_dir, FRED_API_KEY, max_n_tickers)
    print(df_calls.head())
    get_adj_closed_prices()


def bs_option_pricing(is_scrape, output_dir, FRED_API_KEY, max_n_tickers):
    df_calls, df_puts, risk_free_rate = data_acquisition(is_scrape, output_dir, FRED_API_KEY, max_n_tickers)
    df_comparison = check_bs_price(df_calls, risk_free_rate) # first row for testing
    print(df_comparison.head()) 

if __name__ == "__main__":   
    iv_modeling(is_scrape, output_dir, FRED_API_KEY, max_n_tickers)

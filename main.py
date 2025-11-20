import os
from time import time
from dotenv import load_dotenv
from preprocessing import format_opt_df, filt_opt_df, IV_sanity_check

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")   
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") 

start_date = "2020-01-01"
end_date = "2021-12-31"

dir = "data/dataset1"
input_filename = "SPY Options 2010-2023 EOD.csv"
# dir = "data/dataset2"
# input_filename = "spy-daily-eod-options-quotes-2020-2022.csv"

if __name__ == "__main__":

    # format_opt_df(dir, input_filename)
    df = filt_opt_df(dir=dir, filename="calls.csv", start_date=start_date, end_date=end_date)
    # df = IV_sanity_check(df, FRED_API_KEY=FRED_API_KEY)
    

    print("--------Done---------")
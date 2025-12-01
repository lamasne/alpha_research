import os
from dotenv import load_dotenv
from preprocessing import format_opt_data, filt_opt_df, IV_sanity_check, underlying_sanity_check
from modeling import predict_1day_volatility_test, RV_GARCH_prediction

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

    # df_calls, df_puts = format_opt_data(dir, input_filename)
    # df_filt_calls = filt_opt_df(dir=dir, filename="calls.csv", start_date=start_date, end_date=end_date)
    # df = IV_sanity_check(df_filt_calls, FRED_API_KEY=FRED_API_KEY)
    # underlying_sanity_check()
    # predict_1day_volatility_test()
    
    RV_GARCH_prediction()
    
    print("--------Done---------")
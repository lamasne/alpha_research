import os
from dotenv import load_dotenv
from src.data_prep.data_aquisition import get_SPY_EOD_df
from src.data_prep.rv_garch import predict_rolling_garch, rolling_garch_study
from src.data_prep.preprocessing import format_opt_data, IV_sanity_check, underlying_sanity_check
from src.config import full_logger, TrainingConfig
import src.modeling.option_net as option_net
import src.modeling.backtest as backtest
import matplotlib.pyplot as plt


load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")   
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") 

# start_date = "2020-01-01"
# end_date = "2021-12-31"
start_date = None
end_date = None 

if __name__ == "__main__":
    full_logger.blank_line()
    full_logger.info("--------Starting---------")
    # get_SPY_EOD_df()
    # df_calls, df_puts = format_opt_data(is_vol_analysis=False)
    # df_calls = IV_sanity_check(df_calls, opt_type="call", FRED_API_KEY=FRED_API_KEY)
    # underlying_sanity_check()
    # rolling_garch_study()
    # rv_garch_forecasts = predict_rolling_garch(is_plot=False)
    # grid_search_results = option_net.hyperparam_grid_search()
    backtest.standalone_backtest()

    plt.show()
    full_logger.info("--------Done---------")
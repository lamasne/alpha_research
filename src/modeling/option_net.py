import polars as pl
from src.config import ROOT

# Load data
df = pl.read_csv(ROOT / "resources/data/dataset1/calls.csv")


df = (
    df
    .with_columns([
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).alias("moneyness"),
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).log().alias("log_moneyness"),
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("opt_mid"),
        (pl.col("ASK") - pl.col("BID")).alias("opt_spread"),
        pl.col("VOLUME").log1p().alias("log_volume"),
    ])
)

# Add useful derived columns at time t
df = (
    df
    .with_columns([
        # Day index (integer days from epoch)
        (pl.col("QUOTE_UNIX") // 86400).alias("quote_day"),
        (pl.col("EXPIRE_UNIX") // 86400).alias("expire_day"),

        # Moneyness / log-moneyness
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).alias("moneyness"),
        (pl.col("UNDERLYING_LAST") / pl.col("STRIKE")).log().alias("log_moneyness"),

        # Mid, spread
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("opt_mid"),
        (pl.col("ASK") - pl.col("BID")).alias("opt_spread"),

        # Liquidity proxies
        pl.col("VOLUME").log1p().alias("log_volume"),
    ])
)

# If you have underlying bid/ask, RV forecasts, Greeks, RF, merge them here.
# Example expected column names:
# 'UNDERLYING_BID', 'UNDERLYING_ASK', 'RV_GARCH_1D', 'DELTA', 'RF'

# 3) Sort and create next-day columns per (strike, expiry)
df = (
    df
    .sort(["STRIKE", "EXPIRE_UNIX", "QUOTE_UNIX"])
    .with_columns([
        pl.col("BID").shift(-1).over(["STRIKE", "EXPIRE_UNIX"]).alias("next_bid"),
        pl.col("UNDERLYING_LAST").shift(-1).over(["STRIKE", "EXPIRE_UNIX"]).alias("next_underlying"),
        pl.col("quote_day").shift(-1).over(["STRIKE", "EXPIRE_UNIX"]).alias("next_quote_day"),
    ])
)

# 4) Keep only proper next-day observations
df = df.filter(
    (pl.col("next_bid").is_not_null()) &
    (pl.col("next_underlying").is_not_null()) &
    (pl.col("next_quote_day") == pl.col("quote_day") + 1)  # strict next calendar day; adapt if needed
)

# 5) Compute next-day intrinsic and target extrinsic value at bid
df = df.with_columns([
    (pl.max_horizontal(pl.lit(0), pl.col("next_underlying") - pl.col("STRIKE")))
        .alias("next_intrinsic"),
    (pl.col("next_bid") - pl.col("next_intrinsic")).alias("target_extrinsic_bid"),
])

feature_cols = [
    "IV",                 # implied vol at t
    "DTE",                # days to expiry
    "moneyness",
    "log_moneyness",
    "opt_mid",
    "opt_spread",
    "log_volume",
]
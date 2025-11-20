# Backtesting of a Custom ML-Driven Option Trading Strategy

## Motivation
The Black–Scholes framework assumes that if volatility is correctly estimated, a delta-hedged option position should earn the risk-free rate. Many volatility trading strategies are therefore based on forecasting future realized volatility more accurately than the market.

However, option prices are influenced by additional factors beyond volatility expectations. For example, a plausible mechanism is that increased market risk aversion leads to higher option prices. These effects are not captured by volatility-based models alone and may persist systematically.

This project investigates whether a data-driven model can learn such effects and use them to generate alpha.

## Objective & Hypothesis
**Hypothesis.** Option prices embed information beyond volatility alone. A neural network trained on additional variables (e.g. past option price dynamics, liquidity, risk-aversion indicators) can generate profitable trading signals.

Build and train such a model on historical SPY end-of-day options data, then backtest a simple long/short options strategy based on its predictions, including transaction costs.

## Data
- Historical SPY options end-of-day (EOD) data from Kaggle:
    - 2010–2023: main dataset https://www.kaggle.com/datasets/benjaminbtang/spy-options-2010-2023-eod
    - 2020–2022: secondary dataset for cross-validation https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022
- Underlying SPY prices retrieved via `yfinance` and compared to Kaggle values for consistency.
- Filtering criteria:
    - Days to expiration (DTE) between 2 and 10
    - Moneyness (strike distance) between -5% and +5%
    - Option volume above the 95th percentile to ensure liquidity

## Train / Validation / Test Split
- Data split chronologically to avoid look-ahead bias:
    - **Training:** 2010–2018
    - **Validation:** 2019–2020
    - **Test:** 2021–2023
- Validation period used for hyperparameter tuning and threshold selection.
- No shuffling applied to preserve temporal structure.
- All performance metrics reported exclusively on the test period.

## Methodology

### Model
Supervised neural network predicting next-day option mid-price change.

**Output**
- Predicted next-day option mid-price change

**Example input features** (factors that could affect option pricing by market participants): 
- Implied volatility (IV)
- Forecasted volatility (e.g. GARCH)
- Option mid-price
- Price of underlying asset
- Strike price and moneyness
- Days to expiration
- Risk-free rate
- Greeks (e.g. Delta)
- Volume / liquidity indicators
- Market sentiment (e.g. quantified via NLP on social media)

Note: since all Black–Scholes inputs are included among the features, the neural network should be able to reproduce the BS pricing function and potentially extend beyond it if additional effects are present.

### Trading Strategy
To isolate the effect of the model, a simple directional strategy is used:

```
signal = NN prediction of (next-day option price – current price)

if signal > threshold → buy option
if signal < -threshold → sell option
```

Transaction costs are incorporated in the backtest.

## Backtest Framework
To be defined:
- Execution assumptions (EOD pricing, slippage model)
- Performance metrics (Sharpe ratio, annualized return, drawdown, hit ratio)

## Results

### Data exploration
Trading activity is strongly concentrated around at-the-money strikes. As strike distance increases, traded volume decreases approximately exponentially (appearing nearly linear on the log-scale), as shown below.
<img src="plots/volume_to_strike_distance_SPY_calls_2010-2023.png" width="400">

To verify the correctness of the implied volatility computation, I compared the calculated IV (`MY_IV`) against the dataset-provided IV values over random samples:

```
IV (dataset) → IV (computed)
1.32e-01     → 1.31e-01
1.67e-01     → 1.75e-01
1.20e-01     → 1.47e-01
1.33e-01     → 1.40e-01
9.66e-02     → 1.03e-01
...          → ...
```

The custom IV computation shows consistent behavior, with a mean error of **7.01e-03 (5.24%)** and a mean absolute error of **1.07e-02 (7.99%)**, confirming reasonable agreement with the reference data for this sanity check.


### Model
**To be completed upon backtest execution.**

### Backtest
**To be completed upon backtest execution.**

## Conclusions & Future Work
**To be completed upon backtest execution.**



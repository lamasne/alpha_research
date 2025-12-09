from src.config import full_logger, TrainingConfig, outputs_dir
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def standalone_backtest(
    model_dir: Path=outputs_dir/"train_tracking/2025-12-08_experiment_3",
    model_name: str="model_0.pth"
):
    """
    config and model must match
    """
    # Local imports to avoid circular dependency with modeling.option_net
    from src.data_prep.preprocessing import prepare_data, build_dataloaders
    from src.modeling.option_net import predict_option_net, OptionNet

    config_path = model_dir/model_name.replace(".pth", ".json")
    config = TrainingConfig.load(config_path)

    X, y, df = prepare_data(config, is_reporting=False)
    _, X_test, y_test, test_range = build_dataloaders(X, y, df)

    model = OptionNet(input_dim=X.shape[1]).to(config.device).eval()
    model.load_state_dict(torch.load(model_dir/model_name, map_location=config.device, weights_only=True))
    
    next_E, next_E_pred, _ = predict_option_net(model, X_test, y_test, config)
    run_backtest(df[test_range[0]:test_range[1]+1],next_E,next_E_pred)


def run_backtest(
    df_test,
    next_E,
    next_E_pred,
    fee: float = 1., # $ per option transaction
    threshold: float = 0.5, # predicted net gain ($) from which it is worth trading
    init_capital: float = 1e5, # initial capital
    max_trade_capital_ratio: float = 0.7, # maximum capital to use for trading
) -> dict:
    """
    Run backtest using predicted extrinsic values.
    Conditions: 
    - SPY option transaction fee is 'fee'
    - SPY asset transaction fee is negligible
    - trades are executed when expected gain is above 'threshold'
    - initial capital is 'init_capital'
    - no more than half of the capital is used for trading

    
    Args:
        df_test: DataFrame with option data
        next_E: Actual next-day extrinsic at bid (E_{t+1}^{bid})
        next_E_pred: Predicted next-day extrinsic at bid (Ŷ_{t+1}^{bid})
        threshold: Minimum predicted net gain to trigger trade (in $)
    
    Returns:
        Dictionary of backtest statistics
    """
    if torch.is_tensor(next_E):
        next_E = next_E.cpu().numpy()
    if torch.is_tensor(next_E_pred):
        next_E_pred = next_E_pred.cpu().numpy()

    # Calculate current extrinsic at ask
    current_E_ask = df_test["ASK"].to_numpy() - (df_test["STRIKE"].to_numpy() - df_test["UNDERLYING_LAST"].to_numpy())

    # Trading signals
    dates = df_test["QUOTE_DATE"].to_numpy()
    trading_days = np.unique(dates)
    n_opportunities = len(next_E_pred)

    pos_costs = df_test["ASK"].to_numpy() + df_test["UNDERLYING_LAST"].to_numpy() + fee
    net_gain_pred = next_E_pred - current_E_ask - 2*fee
    net_gain_actual = next_E - current_E_ask - 2*fee

    trade_signals = np.zeros(n_opportunities, dtype=bool)
    capital = init_capital
    day_capital_history = [capital]
    cumulative_idx = 0 # Keep track of where each day starts in the full array
    pnl_days = [0.]

    for date in trading_days:
        capital += pnl_days[-1]
        pnl_days.append(0.)

        # Get indices for this date
        day_mask = dates == date
        day_indices = np.where(day_mask)[0]

        day_pos_costs = pos_costs[day_mask]
        day_net_gain_pred = net_gain_pred[day_mask]
        day_net_gain_actual = net_gain_actual[day_mask]

        if len(day_net_gain_pred) == 0:
            day_capital_history.append(capital)
            cumulative_idx += len(day_indices)
            continue

        # while we still have money available to trade under our conditions, buy SPY and put(SPY) of the best predicted gain
        sorted_indices = np.argsort(day_net_gain_pred)[::-1]  # [1, 3, 0, 2]
        ordered_pred_gain = day_net_gain_pred[sorted_indices]
        ordered_actual_gain = day_net_gain_actual[sorted_indices]
        ordered_pos_cost = day_pos_costs[sorted_indices]
        daily_cost = 0.
        current_idx = 0
        while (
            current_idx < ordered_pos_cost.shape[0] 
            and daily_cost + ordered_pos_cost[current_idx] <= max_trade_capital_ratio * capital 
            and current_idx < len(sorted_indices)
            and ordered_pred_gain[current_idx] > threshold
        ):
            # Execute trade
            original_idx = day_indices[sorted_indices[current_idx]]
            trade_signals[original_idx] = True
            daily_cost += ordered_pos_cost[current_idx]
            pnl_days[-1] += ordered_actual_gain[current_idx]
            current_idx += 1

        # Update capital
        capital += pnl_days[-1]
        day_capital_history.append(capital)

        cumulative_idx += len(day_indices)

    # Calculate statistics
    n_trades = trade_signals.sum()
    if n_trades == 0:
        full_logger.info("No trades for the selected parameters")
        return 0
    pnl_per_trade = net_gain_actual[trade_signals] # pnl per realized trade
    avg_pnl_per_trade = pnl_per_trade.mean()
    cumulative_pnl = np.cumsum(pnl_per_trade)
    total_pnl = pnl_per_trade.sum()
    hit_ratio = (pnl_per_trade > 0).mean()
    sharpe = calculate_sharpe(day_capital_history)
    equity = init_capital + cumulative_pnl
    max_drawdown_pct = calculate_max_drawdown_pct(equity)

    # Generate plots
    backtest_plots(
        net_gain_actual, 
        net_gain_pred, 
        equity, 
        trading_days, 
        day_capital_history,
        cumulative_pnl, 
        pnl_per_trade, 
        trade_signals, 
        max_nb_datapoints=int(1e3)
    )

    stats = {
        "threshold": threshold,
        "n_val_points": len(pnl_per_trade),
        "n_trades": int(n_trades),
        "trade_fraction": n_trades / len(pnl_per_trade),
        "total_pnl": float(total_pnl),
        "avg_pnl_per_trade": float(avg_pnl_per_trade),
        "avg_pnl_per_val_point": float(total_pnl / len(pnl_per_trade)),
        "hit_ratio": float(hit_ratio) if not np.isnan(hit_ratio) else np.nan,
        "approx_sharpe_per_trade": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "final_pnl": float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0,
        "max_DD_pct": float(max_drawdown_pct)
    }
    
    full_logger.info(
        f"Backtest (threshold=${threshold:.2f}): "
        f"{stats['n_trades']:,d} trades, "
        f"Total PnL=${stats['total_pnl']:,.2f} (+{total_pnl/init_capital:.2%}), "
        f"Avg/trade=${stats['avg_pnl_per_trade']:.3f}, "
        f"Hit rate={stats['hit_ratio']:.1%}, "
        f"approx_sharpe_per_trade={stats['approx_sharpe_per_trade']:.2f}, "
        f"Max-DD={stats['max_DD_pct']:.2%}")
    
    return stats


def backtest_plots(net_gain_actual, 
    net_gain_pred, 
    equity,
    trading_days, 
    day_capital_history,
    cumulative_pnl, 
    pnl_per_trade, 
    trade_signals, 
    max_nb_datapoints=200
):
    """
    Generate backtest visualization plots.
    """
    n = min(max_nb_datapoints, len(net_gain_pred))
    
    # Figure 1: Net Gain predictions and errors
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: Predicted vs Actual Net Gain
    ax1.plot(net_gain_pred[:n], 'b-', label='Predicted Net Gain', alpha=0.8, linewidth=1.5)
    ax1.plot(net_gain_actual[:n], 'r:', label='Actual Net Gain', alpha=0.8, linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Net Gain ($)')
    ax1.set_title('Trading Signal: Predicted vs Actual Net Gain')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Prediction Error
    error = net_gain_pred[:n] - net_gain_actual[:n]
    ax2.plot(error, 'g-', alpha=0.8, linewidth=1.5, label='Prediction Error')
    ax2.fill_between(range(n), 0, error, alpha=0.2, color='green')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Trade Sequence')
    ax2.set_ylabel('Error ($)')
    ax2.set_title('Model Prediction Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Figure 2: P&L performance
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Cumulative P&L
    ax3.plot(cumulative_pnl, 'b-', linewidth=2)
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.set_title('Strategy Performance')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add key metrics
    final_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
    max_drawdown_pct = calculate_max_drawdown_pct(equity)
    ax3.text(0.02, 0.98, f'Final P&L: ${final_pnl:.2f} ({final_pnl/equity[0]:,.2%})\nMax Drawdown: ${max_drawdown_pct:.2f}\nTrades: {trade_signals.sum():,d}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bottom: Individual trade P&L (first 100)
    n_show = min(100, len(pnl_per_trade))
    colors = ['green' if x > 0 else 'red' for x in pnl_per_trade[:n_show]]
    ax4.bar(range(n_show), pnl_per_trade[:n_show], color=colors, alpha=0.6, width=1.0)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Trade Sequence')
    ax4.set_ylabel('Individual Trade P&L ($)')
    ax4.set_title(f'Trade-by-Trade Results (first {n_show} trades)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()


    fig3, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(trading_days, day_capital_history[1:])
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Capital ($)')
    ax5.set_title('Daily Capital Evolution')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()



def calculate_max_drawdown_pct(equity: np.ndarray) -> float:
    """
    Maximum drawdown as a percentage drop from peak.
    Example: returns 0.25 for a 25% drawdown.
    """
    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)

    # avoid division by zero
    valid = running_max > 0
    rel_dd = np.zeros_like(equity, dtype=float)
    rel_dd[valid] = (running_max[valid] - equity[valid]) / running_max[valid]

    # idx = np.argmax(rel_dd); max_dd = rel_dd[idx]; cumul_at_max_dd = equity[idx]
    # print(idx, max_dd, cumul_at_max_dd)

    return float(np.max(rel_dd))


def calculate_sharpe(capital_history):
    capital_arr = np.array(capital_history, dtype=float)
    daily_returns = capital_arr[1:] / capital_arr[0:-1] - 1

    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
    else:
        sharpe = np.nan

    return sharpe
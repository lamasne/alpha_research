import json
import pandas as pd
import matplotlib.pyplot as plt
from src.config import ROOT, full_logger, TrainingConfig
import time
import datetime
import numpy as np
from functools import wraps


def add_config_to_plot(fig, run_config: TrainingConfig):

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # [left, bottom, right, top]

    footer_text = " | ".join(f"{k}: {v}" for k, v in run_config.__dict__.items())

    fig.text(
        0.02, 0.02, f"Training config: {footer_text}",
        fontsize=8,
        fontfamily='monospace',
        ha='left',
        wrap=True,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )



def create_new_filename(dir, base_name, extension):
    i = 0
    while True:
        new_path = dir / f"{base_name}_{i}.{extension}"
        if not new_path.exists():
            break
        i += 1
    return new_path


def convert_to_polars_date(date_val):
    """Convert any date type to Polars Date."""
    if isinstance(date_val, pd.Timestamp):
        return date_val.date()  # Returns datetime.date
    elif isinstance(date_val, np.datetime64):
        return pd.Timestamp(date_val).date()
    elif isinstance(date_val, str):
        return pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        return date_val.date()
    else:
        return date_val
        


def timeit(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)  # Preserves function metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()
        full_logger.info(f"Time taken by {func.__name__}: {end_time - start_time:.3f} seconds")
        return result
    return wrapper


def plot_time_series(label, title=None):

    if title is None:
        title = label + ' Time Series'

    # Load the JSON data
    with open(ROOT/'resources/data/' + label + '_data.json', 'r') as f:
        data_dict = json.load(f)

    # Convert the dictionary to a pandas DataFrame
    data = pd.Series(data_dict)

    # Plot the data
    plt.figure(figsize=(10, 6))
    data.plot(marker='o', linestyle='-')

    # Customize plot
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()
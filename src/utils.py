import json
import pandas as pd
import matplotlib.pyplot as plt
from src.config import ROOT
import time
import datetime
import numpy as np
from functools import wraps


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
        print(f"Time taken by {func.__name__}: {end_time - start_time:.3f} seconds")
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
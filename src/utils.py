import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
from functools import wraps
from pathlib import Path

def create_new_dir(dir, base_name):
    i = 0
    while True:
        new_path = dir / f"{base_name}_{i}"
        if not new_path.exists():
            break
        i += 1
        # Check if directory is empty
        if new_path.exists() and not any(new_path.iterdir()):
            break  # Directory exists but is empty        
    Path(new_path).mkdir(exist_ok=True)
    return new_path

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

    from src.config import full_logger

    @wraps(func)  # Preserves function metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Format time with appropriate units
        if elapsed < 60:
            time_str = f"{elapsed:.3f} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            time_str = f"{hours}h {minutes}m {seconds:.1f}s"
        
        full_logger.info(f"Time taken by {func.__name__}: {time_str}")        
        return result
    return wrapper


def plot_time_series(input_path, label, title=None):

    if title is None:
        title = label + ' Time Series'

    # Load the JSON data
    with open(input_path, 'r') as f:
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
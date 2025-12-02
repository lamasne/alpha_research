import json
import pandas as pd
import matplotlib.pyplot as plt
from config import ROOT


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
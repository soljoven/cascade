import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_predict_historic(historic, predicted, property_name, num_years, actual=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    if actual:
        prediction_label = 'Prediction'
        title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)
        ax.bar(historic.index,
               historic,
               label='Actual Occupancy (Not to Scale)',
               color='r')
        ax.hlines(.5,
                  historic.index[0],
                  historic.index[-1],
                  linestyles='--')
    else:
        prediction_label = 'Future Prediction'
        title = 'Daily Occupancy Prediction for {}'.format(property_name)
        ax.plot(historic.index,
                historic,
                ':',
                label='Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0]),
                color='r')
    ax.hlines(.5,
          historic.index[0],
          historic.index[-1],
          linestyles='-')
    ax.plot(predicted.index,
            predicted, label=prediction_label, color='b', alpha=.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability of Occupancy')
    ax.set_ylim(top=1.2)
    ax.set_title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()

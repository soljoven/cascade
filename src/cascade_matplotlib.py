import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import psycopg2
import os
from cascade_model import prepare_xy


dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def plot_predict(historic, predicted, property_name, num_years, actual=False, scatter=False, web=False, save=False):
    '''
    Inputs
    historic: pandas timeseries for historic rental occupancy normized by years
    that can be compard to predicted by using retail calendar. So it's not a
    day to day comparison but rather dates equivalent to a given predicted date over
    the years.
    predicted: one year worth of timeseriese for a given property (or aggregate of
    all properties under management)
    property_name: name of given property or all properties. To be used in labels
    num_years: Indicates how many years worth of data is included in the historic
    DataFrame
    actual: indicates whether historic dataframe contain just actual when assessing
    supervised learning (where y_actual values are available)
    scatter: This will plot predicted as scatter while plotting historic as lines
    save: Indicates whether plots need to be saved as .png files for future
    analysis.
    '''

    fig, ax = plt.subplots(figsize=(14, 6))

    prediction_label = 'Future Prediction'
    title = 'Daily Occupancy Prediction for {}'.format(property_name)

    if scatter:
        ax.scatter(predicted.index, predicted, label=prediction_label,
                   color='b', alpha=.5, marker='.')
    else:
        ax.plot(predicted.index,
                predicted, label=prediction_label, color='b', alpha=.5)

    if actual:
        # For supervised learning, can compare actual days when the property was
        # occupied by creating a red dot for indication.
        prediction_label = 'Prediction'
        title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)
        ax.bar(historic.index,
               historic,
               label='Actual Occupancy (Not to Scale)',
               color='r')
    else: #For historic plots
        historic_label = 'Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0])
        # if scatter:
        #     ax.scatter(historic.index, historic, label=historic_label,
        #                color='r', marker='.')
        # else:
        ax.plot(historic.index, historic, ':',label=historic_label,color='r')

    ax.hlines(.5,predicted.index[0],historic.index[-1],linestyles='-')

    ax.set_xlabel('Date', size=15)
    ax.set_ylabel('Probability of Occupancy', size=15)
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(15)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(15)
    # ax.set_ylim(top=1.2)
    ax.set_title(title, size=20)
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig('00{}.png'.format(property_name.replace(" ", "")))

    if web:
        return plt

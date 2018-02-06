import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import psycopg2
import os
from cascade_model import prepare_xy
from cascade_plot import fetch_data_for_plotting


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
    web: Indicates whether the call is intended for the website or not.
    If it is for website, function will return a fig object.
    save: Indicates whether plots need to be saved as .png files for future
    analysis.
    '''

    fig, ax = plt.subplots(figsize=(14, 6))

    prediction_label = 'Prediction'
    title = 'Daily Occupancy Prediction for {}'.format(property_name)

    if actual:
        # For supervised learning, can compare actual days when the property was
        # occupied by creating a red dot for indication.
        prediction_label = 'Prediction'
        title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)
        ax.stem(historic.index, historic.values,
                label='2017 Actual Occupancy (Not to Scale)', size=15, color='r')
    else: #For historic plots
        historic_label = 'Historic Daily Average with at Least {} Year(s)'.format(num_years[0])
        ax.plot(historic.index, historic, ':',label=historic_label,color='red')

    if scatter:
        ax.scatter(predicted.index, predicted, label=prediction_label,
                   color='b', alpha=.8, marker='.')
    else:
        ax.plot(predicted.index,
                predicted, label=prediction_label, color='b', alpha=.8)

    ax.hlines(.5,predicted.index[0],historic.index[-1],linestyles='-')

    ax.set_xlabel('Date', size=15, color='k')
    ax.set_ylabel('Probability of Occupancy', size=15, color='k')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(15)
        xtick.label.set_color('k')
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(15)
        ytick.label.set_color('k')
    # ax.set_ylim(top=1.2)
    ax.set_title(title, size=20)
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig('00{}.png'.format(property_name.replace(" ", "")))

    if web:
        return plt

def plot_model_comparison(df, property_name, gbc_prob, rf_prob, lr_prob, save_fig=False, shorten_historic_legend=False):
    '''
    This module has a very unique purpose and it is to compare 3 models
    used in the project. It creates a matplotlib based plot comparing
    prediction from 3 models against historic actual.

    Inputs
    df: Full year worth (12 months) of data that was used for model.predict_proba
    property_name: Name of property under evaluation
    gbc_prob: Probabiliyt of GBC model based on df
    rf_prob: Probabiliyt of RF model based on df
    lr_prob: Probabiliyt of LR model based on df
    save_fig: Indicates whether the plot should be saved during its run
    shorten_historic_legend: Shortens legend for historic data
    '''

    # This is set to the first(lowest) calendar day of the dataframe
    start_date = str(df.day[0])

    # TO DO: Make so that a dictionary of tuples that contain
    # prediction, unique plot features such as color and label
    # rather than calling fetch_data_for_plotting 3 times.
    y_series_GBC, predicted_GBC, num_years_GBC = fetch_data_for_plotting(df,
                                                                 property_name,
                                                                 gbc_prob,
                                                                 start_date,
                                                                 True )
    y_series_RF, predicted_RF, num_years_RF = fetch_data_for_plotting(df,
                                                             property_name,
                                                             rf_prob,
                                                             start_date,
                                                             True )
    y_series_LR, predicted_LR, num_years_LR = fetch_data_for_plotting(df,
                                                             property_name,
                                                             lr_prob,
                                                             start_date,
                                                             True )
    historic = y_series_GBC

    fig, ax = plt.subplots(figsize=(14, 6))

    prediction_label = 'Future Prediction'
    title = 'Daily Occupancy Prediction for {}'.format(property_name)

    if shorten_historic_legend:
        historic_label = 'Historic Actual'
    else:
        historic_label = 'Historic Actual with {} Years of Daily Average Occupancy Rate'.format(num_years_GBC[0])

    ax.plot(historic.index, historic, ':',label=historic_label, color='r')
    ax.plot(predicted_LR.index,
            predicted_LR, label='Future Prediction with LR', alpha=.6, color='k')
    ax.plot(predicted_RF.index,
            predicted_RF, label='Future Prediction with RF', alpha=.8, color='c')
    ax.plot(predicted_GBC.index,
            predicted_GBC, label='Future Prediction with GBC', alpha=.7, color='b')

    ax.set_xlabel('Date', size=15, color='k')
    ax.set_ylabel('Probability of Occupancy', size=15, color='k')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(15)
        xtick.label.set_color('k')
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(15)
        ytick.label.set_color('k')
    ax.set_title(title, size=20)
    plt.legend(loc="upper left", prop={'size':14})
    plt.tight_layout()

    if save_fig:
        plt.savefig('00_model_comp_{}.png'.format(property_name))

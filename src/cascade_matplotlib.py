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

    if actual:
        # For supervised learning, can compare actual days when the property was
        # occupied by creating a red dot for indication.
        prediction_label = 'Prediction'
        title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)
        ax.bar(historic.index,
               historic.values,
               label='2017 Actual Occupancy (Not to Scale)',
               color='r')
    else: #For historic plots
        historic_label = 'Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0])
        # if scatter:
        #     ax.scatter(historic.index, historic, label=historic_label,
        #                color='r', marker='.')
        # else:
        ax.plot(historic.index, historic, ':',label=historic_label,color='r')

    if scatter:
        ax.scatter(predicted.index, predicted, label=prediction_label,
                   color='b', alpha=.8, marker='.')
    else:
        ax.plot(predicted.index,
                predicted, label=prediction_label, color='b', alpha=.8)

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

def plot_model_comparison(df, property_name, gbc_prob, rf_prob, lr_prob, save_fig=False):
    start_date = str(df.day[0])

    y_series_GBC, predicted_GBC, num_years_GBC = fetch_data_for_plotting(df,
                                                                 property_name,
                                                                 gbc_prob,
                                                                 start_date,
                                                                 True )
    y_series_RF, predicted_RF, num_years_RF = fetch_data_for_plotting(cascade_test,
                                                             property_name,
                                                             RF_prob,
                                                             start_date,
                                                             True )
    y_series_LR, predicted_LR, num_years_LR = fetch_data_for_plotting(cascade_test,
                                                             property_name,
                                                             LR_prob,
                                                             start_date,
                                                             True )
    historic = y_series_GBC

    fig, ax = plt.subplots(figsize=(14, 6))

    prediction_label = 'Future Prediction'
    title = 'Daily Occupancy Prediction for {}'.format(property_name)

    historic_label = 'Historic Actual Based on {} Years of Daily Average Occupancy Rate'.format(num_years[0])
    ax.plot(historic.index, historic, ':',label=historic_label, color='r')

    ax.plot(predicted_LR.index,
            predicted_LR, label='Future Prediction with LR', alpha=.6, color='k')
    ax.plot(predicted_RF.index,
            predicted_RF, label='Future Prediction with RF', alpha=.8, color='c')
    ax.plot(predicted.index,
            predicted_GBC, label='Future Prediction with GBC', alpha=.7, color='b')

    # ax.hlines(.5,historic.index[0],historic.index[-1],linestyles='-')

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
        plt.savefig('00model_comp{}.png'.format(property_name))

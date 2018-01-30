import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import os
from cascade_model import prepare_xy


dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def plot_predict(historic, predicted, property_name, num_years, actual=False, scatter=False, save=False):
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
        prediction_label = 'Future Prediction'
        historic_label = 'Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0])
        title = 'Daily Occupancy Prediction for {}'.format(property_name)

        # if scatter:
        #     ax.scatter(historic.index, historic, label=historic_label,
        #                color='r', marker='.')
        # else:
        ax.plot(historic.index, historic, ':',label=historic_label,color='r')

    ax.hlines(.5,historic.index[0],historic.index[-1],linestyles='-')

    if scatter:
        ax.scatter(predicted.index, predicted, label=prediction_label,
                   color='b', alpha=.5, marker='.')
    else:
        ax.plot(predicted.index,
                predicted, label=prediction_label, color='b', alpha=.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Probability of Occupancy')
    # ax.set_ylim(top=1.2)
    ax.set_title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig('00{}.png'.format(property_name.replace(" ", "")))

def fetch_data_for_plotting(df, property_name, prob, start_date, historic=True):
    '''
    Inputs
    df: full data set that's reflective of data in cascade_full table.
    property_name: name of the property that needs historic rental data
    prob: once the model has been fitted, result of model.predict_proba
    start_date: this is the date that the model was last fitted. Query will
    retrieve historic data corresponding to the future year using retail calendar
    historic: indicates whether the historic information needs to be returned
    using database query/fetch

    Output
    y_series: To be used by timeseries plot
    predicted_occupied: prediction that corresond to y_series(timeseries)
    num_years: if it's historic, indicates how many years of historical information
    is included in y_series.
    '''
    conn = psycopg2.connect(dbname=dbname,
                        user=username, password=password,
                        host=host,
                        port=port)

    num_years = 0

    X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(df, [], [], True)

    result_df = df[['property_code', 'day']].loc[X_test.index]
    result_df['prob_0'] = prob[:,0]
    result_df['prob_1'] = prob[:,1]
    predicted_occupied = result_df[result_df['property_code'] == property_name]
    predicted_occupied.index = predicted_occupied['day']
    predicted_occupied = predicted_occupied.drop(['day','property_code', 'prob_0'], axis=1)

    if historic:
        query_1 = '''
                select a.day, b.occupied
                  from (select cf1.day, rc1.month_no, rc1.week_no, rc1.day_no
                          from cascade_full cf1,
                               retail_calendar rc1
                         where rc1.day = cf1.day
                           and cf1.property_code = %s
                           and cf1.day >= to_date(%s, 'YYYY-MM-DD')) a,
                       (select rc2.month_no, rc2.week_no, rc2.day_no, avg(cf2.occupied) as occupied
                          from retail_calendar rc2,
                               cascade_full cf2
                         where rc2.day = cf2.day
                           and cf2.property_code = %s
                           and cf2.day < to_date(%s, 'YYYY-MM-DD')
                         group by rc2.month_no, rc2.week_no, rc2.day_no) b
                 where a.month_no = b.month_no
                   and a.week_no = b.week_no
                   and a.day_no = b.day_no
                   limit 366
                ;'''

        query_2 = '''
                    select count(distinct(year)) - 1 as num_years
                      from cascade_full
                     where property_code = %s
                       and day < to_date(%s, 'yyyy-mm-dd');
                  '''

        historic_occupied = pd.read_sql_query(query_1, conn,
                                              params=[property_name, start_date, property_name, start_date])
        cur = conn.cursor()
        cur.execute(query_2, [property_name, start_date])
        num_years = cur.fetchone()

        conn.close()

        historic_occupied.index = historic_occupied['day']
        historic_occupied = historic_occupied.drop('day', axis=1)

        return historic_occupied, predicted_occupied, num_years
    else:
        y_actual = y_test[result_df[result_df['property_code'] == property_name].index]
        y_series = pd.Series(np.divide(y_actual, 30).values, predicted_occupied.index)

        return y_series, predicted_occupied, num_years

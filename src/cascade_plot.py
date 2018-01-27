import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import os
from .cascade_model import prepare_xy


dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def plot_predict(historic, predicted, property_name, num_years, actual=False, scatter=False, save=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    if actual:
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
    conn = psycopg2.connect(dbname=dbname,
                        user=username, password=password,
                        host=host,
                        port=port)

    num_years = 0

    X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(df, True)

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

import numpy as np
import pandas as pd
import psycopg2
import os
import sys
from sqlalchemy import create_engine

home_path = os.environ['CASCADE_HOME']
sys.path.append(home_path + 'cascade/src')
from cascade_model import prepare_xy, make_xy
from cascade_sql import write_to_table

dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

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

    result_df = df.copy()
    result_df = result_df[['property_code', 'day']]
    result_df['prob'] = prob[:,1]

    if property_name == 'All Properties':
        predicted_occupied = pd.DataFrame(result_df.groupby('day').prob.mean())
        predicted_occupied.index = pd.to_datetime(predicted_occupied.index)
    else:
        predicted_occupied = result_df[result_df['property_code'] == property_name]
        predicted_occupied.index = pd.to_datetime(predicted_occupied['day'])
        predicted_occupied = predicted_occupied.drop(['day','property_code'], axis=1)

    if historic:
        individual_prop = '''
                select a.day, b.occupied
                  from (select cf1.day, rc1.month_no, rc1.week_no, rc1.day_no
                          from cascade_test cf1,
                               retail_calendar rc1
                         where rc1.day = cf1.day
                           and cf1.property_code = %s
                           and cf1.day >= to_date(%s, 'YYYY-MM-DD')) a,
                       (select rc2.month_no, rc2.week_no, rc2.day_no, avg(cf2.occupied) as occupied
                          from retail_calendar rc2,
                               cascade_hist cf2
                         where rc2.day = cf2.day
                           and cf2.property_code = %s
                           and cf2.day < to_date(%s, 'YYYY-MM-DD')
                         group by rc2.month_no, rc2.week_no, rc2.day_no) b
                 where a.month_no = b.month_no
                   and a.week_no = b.week_no
                   and a.day_no = b.day_no
                   limit 366
                ;'''

        all_properties = '''
                select a.day, avg(b.occupied)
                  from (select cf1.day, rc1.month_no, rc1.week_no, rc1.day_no, count(*)
                          from cascade_test cf1,
                               retail_calendar rc1
                         where rc1.day = cf1.day
                           and cf1.day >= to_date(%s, 'YYYY-MM-DD')
                         group by cf1.day, rc1.month_no, rc1.week_no, rc1.day_no) a,
                       (select rc2.month_no, rc2.week_no, rc2.day_no, avg(cf2.occupied) as occupied
                          from retail_calendar rc2,
                               cascade_hist cf2
                         where rc2.day = cf2.day
                           and cf2.day < to_date(%s, 'YYYY-MM-DD')
                         group by rc2.month_no, rc2.week_no, rc2.day_no) b
                 where a.month_no = b.month_no
                   and a.week_no = b.week_no
                   and a.day_no = b.day_no
                 group by a.day
                 order by a.day
                 limit 365
                '''
        query_num_years = '''
                    select count(distinct(year)) - 1 as num_years
                      from cascade_full
                     where property_code = %s
                       and day < to_date(%s, 'yyyy-mm-dd');
                  '''
        if property_name == 'All Properties':
            historic_occupied = pd.read_sql_query(all_properties, conn,
                                                  params=[start_date, start_date])
        else:
            historic_occupied = pd.read_sql_query(individual_prop, conn,
                                                  params=[property_name, start_date, property_name, start_date])
        cur = conn.cursor()
        cur.execute(query_num_years, [property_name, start_date])
        num_years = cur.fetchone()

        conn.close()

        historic_occupied.index = pd.to_datetime(historic_occupied['day'])
        historic_occupied = historic_occupied.drop('day', axis=1)

        return historic_occupied, predicted_occupied, num_years
    else:
        y_actual = y_test[result_df[result_df['property_code'] == property_name].index]
        y_series = pd.Series(np.divide(y_actual, 30).values, predicted_occupied.index)

        return y_series, predicted_occupied, num_years

def web_prop_list():
    '''
    This function will return a list of properties that will be eligible for
    getting its prediction. Note that adding 2 years worth of days + 2 to
    account for leap year.
    '''
    conn = psycopg2.connect(dbname=dbname,
                            user=username, password=password,
                            host=host,
                            port=port)

    query_1 = '''
            select distinct(property_code)
              from cascade_test
             order by 1
            ;'''

    list_prop = pd.read_sql_query(query_1, conn)

    return list(['All Properties']) + list(list_prop.property_code)

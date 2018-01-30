import numpy as np
import pandas as pd
import os
import sys
import psycopg2
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DataRange1d, Select
from bokeh.palettes import Blues4
from bokeh.plotting import figure
import pickle

sys.path.append("/Users/youngsun/galvanize/dsi/capstone/cascade/src/")
from cascade_model import prepare_xy
from cascade_plot import fetch_data_for_plotting

dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def make_plot(y_series, predicted, property_name, num_years):
    p = figure(tools="pan,box_zoom,reset,save",
               width=800, x_axis_type="datetime",
               x_axis_label='Date',
               y_axis_label='Predicted Occupancy Rate')

    historic_label = 'Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0])
    title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)

    p.line(y_series.index,
           y_series.values.flatten(),
           color='red',
           alpha=0.7,
           line_dash='dashed',
           legend=historic_label)
    p.line(predicted.index,predicted.values.flatten(),color='blue',legend='Future Prediction')

    p.title.text = title
    p.legend.location = "top_left"
    p.ygrid.band_fill_alpha = 0.2
    p.themed_values()
    p.grid.grid_line_alpha = .3
    p.axis.axis_label_text_font_style = "bold"
    p.x_range = DataRange1d(range_padding=0.0)

    return p

def update_plot(X, gbc_predict, start_date):
    prop = prop_select.value
    p.title.text = 'Actual vs. Predicted Daily Occupancy % for {}'.format(property_name)

    y_series, predicted, num_years = fetch_data_for_plotting(X,
                                                             prop,
                                                             gbc_predict,
                                                             start_date,
                                                             True)
    return y_series, predicted, num_years


conn = psycopg2.connect(dbname=dbname,
                        user=username, password=password,
                        host=host,
                        port=port)

query = '''
        select * from cascade_full
        ;'''

cascade = pd.read_sql_query(query, conn)
conn.close()

X = cascade.copy()
X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, [], [], True)

prop = 'Aspenwood 6540'

prop_select = Select(value=prop, title='Property Code', options=sorted(list(unique_prop_codes)))

start_date = str(X.day.loc[X_test.index].iloc[0])

with open('/Users/youngsun/galvanize/dsi/capstone/model.pkl', 'rb') as f:
    GBC_model = pickle.load(f)

gbc_predict = GBC_model.predict_proba(X_test)

y_series, predicted, num_years = fetch_data_for_plotting(X,
                                                         prop,
                                                         gbc_predict,
                                                         start_date,
                                                         True)


plot = make_plot(y_series, predicted, prop, num_years)

prop_select.on_change('value', update_plot)

controls = column(prop_select)

curdoc().add_root(row(plot, controls))
curdoc().title = 'Actual vs. Predicted Daily Occupancy for {}'.format(prop)
curdoc()

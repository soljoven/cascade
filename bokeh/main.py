import numpy as np
import pandas as pd
import os
import sys
import psycopg2
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DataRange1d, Select
from bokeh.models import BoxAnnotation
from bokeh.palettes import Blues4
from bokeh.plotting import figure
import pickle

home_path = os.environ['CASCADE_HOME']

sys.path.append(home_path + 'cascade/src')
from cascade_model import prepare_xy
from cascade_plot import fetch_data_for_plotting, web_prop_list

dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def get_dataset(X, prop, gbc_predict):
    start_date = str(X.day.loc[X_test.index].iloc[0])
    y_series, predicted, num_years = fetch_data_for_plotting(X,
                                                             prop,
                                                             gbc_predict,
                                                             start_date,
                                                             True)

    joined = pd.merge(y_series, predicted, left_index=True, right_index=True)
    joined.columns = ['occupied', 'prob_1']

    return ColumnDataSource(data=joined), num_years

def make_plot(source, property_name, num_years):
    p = figure(toolbar_location=None,
               width=800,
               x_axis_type="datetime",
               x_axis_label='Date',
               y_axis_label='Predicted Daily Occupancy Rate')

    title = 'Predicted Daily Occupancy For:'

    p.line('day',
           'occupied',
           color='red',
           alpha=0.4,
           # line_dash='dashed',
           legend='Historic Actual of Daily Average Occupancy Rate',
           source=source)

    p.circle('day',
           'prob_1',
           size=3, color='blue', alpha=0.7,
           # color='blue',
           legend='Prediction of Future Daily Occupancy',
           source=source)

    p.title.text = title
    p.title.text_font_size = '34pt'
    p.legend.location = "top_left"
    p.ygrid.band_fill_alpha = 0.2
    p.grid.grid_line_alpha = .3
    p.axis.axis_label_text_font_style = "bold"
    p.x_range = DataRange1d(range_padding=0.0)
    p.add_layout(BoxAnnotation(bottom=.6, fill_alpha=0.1, fill_color='blue'))
    p.themed_values()

    return p

def update_plot():
    prop = prop_select.value
    p.title.text = 'Predicted Daily Occupancy For:'
    src, num_years = get_dataset(X, prop, gbc_predict)
    source.data.update(src.data)


conn = psycopg2.connect(dbname=dbname,
                        user=username, password=password,
                        host=host,
                        port=port)

query = '''select * from cascade_full'''

cascade = pd.read_sql_query(query, conn)
conn.close()

# cascade = pd.read_csv(home_path + '/cascade.csv', index_col=0)
# print(cascade.shape)
# print(cascade.head())
# print(cascade.info())

X = cascade.copy()
# print(X.shape)
X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, [], [], True)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

prop = 'All Properties'
start_date = str(X.day.loc[X_test.index].iloc[0])

with open(home_path+'/model.pkl', 'rb') as f:
    GBC_model = pickle.load(f)

gbc_predict = GBC_model.predict_proba(X_test)

prop_select = Select(value=prop, title='Property Code', options=web_prop_list())

source, num_years = get_dataset(X, prop, gbc_predict)
p = make_plot(source, prop, num_years)

prop_select.on_change('value', lambda attr, old, new: update_plot())

controls = column(prop_select)

curdoc().add_root(row(p, controls))
curdoc().title = 'Historic Actual vs. Predicted Daily Occupancy for Cascade Rental Properties'
curdoc()

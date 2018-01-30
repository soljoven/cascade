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

def get_dataset(X, prop, gbc_predict):
    start_date = str(X.day.loc[X_test.index].iloc[0])
    y_series, predicted, num_years = fetch_data_for_plotting(X,
                                                             prop,
                                                             gbc_predict,
                                                             start_date,
                                                             True)

    joined = pd.merge(y_series, predicted, left_index=True, right_index=True)

    return ColumnDataSource(data=joined), num_years

def make_plot(source, property_name, num_years):
    p = figure(tools="pan,box_zoom,reset,save",
               width=800, x_axis_type="datetime",
               x_axis_label='Date',
               y_axis_label='Predicted Daily Occupancy Rate')

    historic_label = 'Historic Actual Based on at Least {} Year(s) of Daily Average Occupancy Rate'.format(num_years[0])
    title = 'Actual vs. Predicted Daily Occupancy for {}'.format(property_name)

    p.line('day',
           'occupied',
           color='red',
           alpha=0.7,
           line_dash='dashed',
           legend=historic_label,
           source=source)

    p.line('day', 'prob_1',color='blue',legend='Future Prediction', source=source)

    p.title.text = title
    p.legend.location = "top_left"
    p.ygrid.band_fill_alpha = 0.2
    p.themed_values()
    p.grid.grid_line_alpha = .3
    p.axis.axis_label_text_font_style = "bold"
    p.x_range = DataRange1d(range_padding=0.0)

    return p

def update_plot():
    prop = prop_select.value
    p.title.text = 'Actual vs. Predicted Daily Occupancy % for {}'.format(prop)
    src, num_years = get_dataset(X, prop, gbc_predict)
    print('after get_dataset')
    source.data.update(src.data)
    print('after source.data.update')


conn = psycopg2.connect(dbname=dbname,
                        user=username, password=password,
                        host=host,
                        port=port)

query = '''
        select * from cascade_full
        ;'''

cascade = pd.read_sql_query(query, conn)
conn.close()

# cascade = pd.read_csv('/Users/youngsun/galvanize/dsi/capstone/cascade/web/cascade.csv', index_col=0)
# print(cascade.shape)

X = cascade.copy()
X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, [], [], True)

prop = 'Aspenwood 6540'
start_date = str(X.day.loc[X_test.index].iloc[0])

with open('/Users/youngsun/galvanize/dsi/capstone/model.pkl', 'rb') as f:
    GBC_model = pickle.load(f)

gbc_predict = GBC_model.predict_proba(X_test)

prop_select = Select(value=prop, title='Property Code', options=sorted(list(unique_prop_codes)))

source, num_years = get_dataset(X, prop, gbc_predict)
p = make_plot(source, prop, num_years)

prop_select.on_change('value', lambda attr, old, new: update_plot())

controls = column(prop_select)

curdoc().add_root(row(p, controls))
curdoc().title = 'Actual vs. Predicted Daily Occupancy for {}'.format(prop)
curdoc()

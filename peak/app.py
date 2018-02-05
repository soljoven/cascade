from flask import Flask, render_template, request, jsonify, make_response
import os
import sys
from io import BytesIO
import psycopg2
import pandas as pd
home_path = os.environ['CASCADE_HOME']
sys.path.append(home_path + 'cascade/src')
from cascade_plot import fetch_data_for_plotting, web_prop_list
from cascade_model import prepare_xy, make_xy
import matplotlib
matplotlib.use("agg")
from cascade_matplotlib import plot_predict
import pickle
import matplotlib.pyplot as plt

dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

app = Flask(__name__)

cascade_test = pd.read_csv(home_path + 'data/cascade_test.csv', index_col=0)
X = cascade_test.copy()
# X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, [], [], True)
X_test, y_test = make_xy(X, [], [])

property_name = 'All Properties'
start_date = str(cascade_test.day[0])

with open(home_path + 'GBC_model_1802.pkl', 'rb') as f:
    GBC_model = pickle.load(f)
gbc_predict = GBC_model.predict_proba(X_test)

@app.route('/')
def index():
    '''
    Retrieves the list of property list for the dropdown
    and populate it at website load
    '''
    prop_list = web_prop_list()
    return render_template('peak.html',prop_list=prop_list)

@app.route('/plot/<property_name>')
def _plot(property_name):
    '''
    Input: Property Name selected by user using the dropdown

    Function calls fetch_data_for_plotting to retrieve actual
    for last full calendar year for peak period as well as
    prediction for the given property code and
    returns a matplotlib figure object
    '''
    historic, predicted, num_years = fetch_data_for_plotting(X,
                                                             property_name,
                                                             gbc_predict,
                                                             start_date,
                                                             False)

    plt = plot_predict(historic, predicted, property_name, num_years, True, False, True)
    image = BytesIO()
    plt.savefig(image)
    return image.getvalue(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

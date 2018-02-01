from flask import Flask, render_template, request, jsonify, make_response
import os
import sys
from io import BytesIO
import psycopg2
import pandas as pd
home_path = os.environ['CASCADE_HOME']
sys.path.append(home_path + 'cascade/src')
from cascade_plot import fetch_data_for_plotting, web_prop_list
from cascade_model import prepare_xy
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

cascade = pd.read_csv(home_path + 'cascade.csv', index_col=0)
X = cascade.copy()
X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, [], [], True)

property_name = 'All Properties'
start_date = str(X.day.loc[X_test.index].iloc[0])

with open(home_path + 'model.pkl', 'rb') as f:
    GBC_model = pickle.load(f)
gbc_predict = GBC_model.predict_proba(X_test)

@app.route('/')
def index():
    prop_list = web_prop_list()
    return render_template('index.html',prop_list=prop_list)

@app.route('/plot/<property_name>')
def _plot(property_name):

    y_series, predicted, num_years = fetch_data_for_plotting(X,property_name,gbc_predict,start_date,True)

    plt = plot_predict(y_series, predicted, property_name, num_years, False, True, True)
    image = BytesIO()
    plt.savefig(image)
    return image.getvalue(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import *
import numpy as np
import pandas as pd
import os
import sys
import psycopg2
from sqlalchemy import create_engine

home_path = os.environ['CASCADE_HOME']
sys.path.append(home_path + 'cascade/src')

from cascade_sql import *
from cascade_model import *

def remove_element(feature):
    prop_list = ['property_code', 'property_city', 'property_zip', 'series', 'num_guests',
                 'num_bedrooms', 'num_bathrooms', 'allows_pets', 'property_rating',
                 'manager_rating', 'weekend', 'season', 'min_nights']
    dummy_list = ['property_code', 'property_city', 'property_zip', 'series', 'num_guests',
                 'num_bedrooms', 'num_bathrooms', 'season', 'min_nights']

    if feature in dummy_list:
        prop_list.remove(feature)
        dummy_list.remove(feature)

    if feature in prop_list:
        prop_list.remove(feature)

    return prop_list, dummy_list

def model_fit(model, X_train, X_test, y_train, y_test, engine, feature_name, cv_folds=5):

    model.fit(X_train, y_train)

    #Predict training set:
    predictions = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    #Perform cross-validation:
    cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')

    result = [(feature_name,
               log_loss(y_test, prob),
               accuracy_score(y_test, predictions),
               roc_auc_score(y_test, prob),
               np.mean(cv_score),
               np.std(cv_score),
               np.min(cv_score),
               np.max(cv_score))]
    labels = ['removed_feature', 'log_loss','accuracy', 'AUC_score', 'cv_mean', 'cv_std', 'cv_min', 'cv_max']
    df = pd.DataFrame.from_records(result, columns=labels)
    df = df.append([df],ignore_index=True)


    write_to_table(df, engine, 'take_one_results', 'append')

def main():

    dbname = os.environ['CASCADE_DB_DBNAME']
    host = os.environ['CASCADE_DB_HOST']
    username = os.environ['CASCADE_DB_USERNAME']
    password = os.environ['CASCADE_DB_PASSWORD']
    port = 5432

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

    prop_list = ['property_code', 'property_city', 'property_zip', 'series', 'num_guests',
                 'num_bedrooms', 'num_bathrooms', 'allows_pets', 'property_rating',
                 'manager_rating', 'weekend', 'season', 'min_nights']

    GBC = GradientBoostingClassifier(learning_rate=.05,
                                     n_estimators=1000,
                                     max_depth=10,
                                     subsample=1,
                                     max_features='sqrt',
                                     random_state=1)

    lr = LogisticRegression()

    for feature in prop_list:
        address = 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname)
        engine = create_engine(address)

        prop, dummy = remove_element(feature)

        X_train, X_test, y_train, y_test, unique_prop_codes = prepare_xy(X, prop, dummy)

        model_fit(GBC, X_train, X_test, y_train, y_test, engine, feature)

if __name__ == '__main__':
    main()

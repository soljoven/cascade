import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def make_xy(df, columns, property_code=True):
    '''
    Creates X, y split for a given dataframe and creating dummified columns for
    select categorial features (hard coded)
    Input
    df: dataframe to split into X, y
    columns: indicates which columns need to be included in the final X dataframe
    property_code: Indicates whether X dataframe should be inclusive of
    property_code and its dummified values.
    '''

    if property_code:
        X = df[columns]
        X = pd.get_dummies(X, columns=['property_code','property_city',
                                         'property_zip', 'series',
                                         'num_guests', 'num_bedrooms',
                                         'num_bathrooms', 'season', 'min_nights'])
    else:
        if 'property_code' in columns:
            columns.remove('property_code')
        X = df[columns]
        X = pd.get_dummies(X, columns=['property_city',
                                         'property_zip', 'series',
                                         'num_guests', 'num_bedrooms',
                                         'num_bathrooms', 'season', 'min_nights'])

    y = df['occupied']

    return X, y

def split_by_year(df, test_year=2017):
    '''
    As the name indicates, dataframe will be split by year
    Input
    df: dataframe to split by year
    test_year: indicates which year's worth data will be held out as test
    Output: df_train and df_test.
    '''
    df_train = df.copy()

    df_temp = df_train[df_train.year < test_year + 1]
    df_test = df_train[df_train.year == test_year]
    df_train = df_train[df_train.year < test_year]

    property_code_before_test = set(list(df_train.property_code.unique()))
    property_code_test_year = set(list(df_test.property_code.unique()))
    property_code_to_remove = property_code_test_year - property_code_before_test
    to_remove_index = df_test[df_test.property_code.isin(list(property_code_to_remove))].index
    df_test.drop(to_remove_index, inplace=True)

    return df_train, df_test, property_code_to_remove

def prepare_xy(df, year_split=False, property_code=True, test_year=2017):
    '''
    Input
    df: dataframe to split into train and test set
    year_split: indicates whether the train/test set needs to be split along year
    In current scenario, split will be historical date between 2012 to 2016 as train
    while test will be 2017.
    property_code: Indicates whether property_code should be party of feature set

    Output: returns X_train, X_test, y_train, y_test and unique property codes
    as in the case of preparing splits, if data is split into years, it'll need
    unique property codes.
    '''
    columns =['property_code','property_city', 'property_zip', 'series',
              'num_guests', 'num_bedrooms', 'num_bathrooms', 'allows_pets',
              'manager_rating', 'property_rating', 'weekend', 'season',
              'min_nights']

    df_train, df_test, property_code_to_remove = split_by_year(df, test_year)

    unique_property_codes = df_train.property_code.unique()

    if year_split:
        X_train, y_train = make_xy(df_train, columns, property_code)
        X_test, y_test = make_xy(df_test, columns, property_code)

    else:
        to_remove_index = df[df.property_code.isin(list(property_code_to_remove))].index
        df.drop(to_remove_index, inplace=True)
        X, y = make_xy(df, columns, property_code)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=127)

    return X_train, X_test, y_train, y_test, unique_property_codes

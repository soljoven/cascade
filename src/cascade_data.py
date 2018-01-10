from pandasql import sqldf
import pandas as pd
import numpy as np
from datetime import timedelta
import time

def columns_cleanup(df):
    '''
    Input: dataframe which columns names need to be cleaned up.
    Output: dataframe with cleaned up column names

    Following modifications will be done:
    1. lowercase everything
    2. replace space with underscore
    3. remove parenthesis
    4. replace # with num
    5. replace '/' with underscore
    6. replace '-' with underscore
    '''
    df.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('#', 'num').replace('/', '_').replace('-', '_') for c in df.columns]
    return df

def read_csv(year, columns):
    '''
    Input:
    1. Year in which a given file to be read from data directory
    2. List of columns as defined by list_columns() function

    Output: dataframe with read data from csv for corresponding year.

    columns_cleanup() function is called to clean up column names

    Note that for files with year greater than year 2014, there is an extra column
    called integration_tax that will be removed.

    For year 2017, source data was encoded differently and had to apply
    ending of 'latin-1' when reading from csv.

    '''
    year = int(year)
    if year == 17:
        df = pd.read_csv('./data/{}salescycle.csv'.format(year), encoding='latin-1')
    else:
        df = pd.read_csv('./data/{}salescycle.csv'.format(year))

    df = columns_cleanup(df)

    if year > 14:
        df = df.drop('integration_tax', axis=1)

    df = df[columns]

    print("Size of columns for year {}: {}".format(year, len(df.columns)))
    print("Number of rows for year {}: {}".format(year, len(df)))

    return df[:-1]

def one_df(years, columns):
    '''
    Input: list of years corresponding to individual file
    Output: one dataframe to rule them all

    '''
    frames = []
    for year in years:
        frames.append(read_csv(year, columns))

    df = pd.concat(frames)
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], format="%m/%d/%Y", errors='ignore')
    df['date'] = pd.to_datetime(df['arrival_date'], format="%m/%d/%Y", errors='ignore')

    return df

def list_columns():
    '''
    Outpu: list of final columns for the dataframe.

    Because of recent changes in the source file format in which number of columns
    seemed to have dropped, using 2017 data file as a basis for columns for the
    dataframe.
    Note that because integration_tax column was introduced in 2015 onwards, it's
    dropped from the final column list
    '''

    columns = list(columns_cleanup(pd.read_csv('./data/17salescycle.csv', encoding='latin-1')).columns)
    columns_to_remove = {'property_address','property_country', 'property_state', 'property_country',
                         'reservation_type','location_id','agent','company_name','first_name',
                         'last_name','phone','cell_phone','email','address','city','state','zip',
                         'process_status','transaction_status','date_booked','num_of_days_til_arrival',
                         'additional_guest_fee_rent','gross_total_rental_rate',
                         'specials','specials_total','rental_rate_adjusted','gross_total_adjusted',
                         'taxable_totals_subtotals','tax_rate','sales_tax','custom_tax_total',
                         'integration_tax','non_taxable_total','total_balance','deposit',
                         'charges_processed','charges_without_processing','charged_total','check_total',
                         'other_charge_types_total','internal_credit_total','payments_total','refund_total',
                         'balance_due','balance_due_date','cancellation_date','referral_source_comments',
                         'transaction_id', 'rental_rate', 'departure_date'}
    columns = [c for c in columns if c not in columns_to_remove]

    return columns

def breakout_num_nights(df):
    '''
    Input: A pandas dataframe the contains initial cascade information that
    contains arrival_date and num_nights
    Output: A pandas dataframe that contains additional rows that correspond to
    arrival date + num_nights for given property_code

    Method: 1. Make a numpy array from input pandas dataframe
            2. Make a copy of numpy array from step 1
            3. Iterate through as many records there are in the numpy array
            4. if the num_nights (in the current array, the position is -2) is greater than 1
                create additional row using np.vstack
            5. Update the date of the newly created row by adding a day
            6. Convert the numpy array into a pandas dataframe
            7. Read data based on webscraping to retrieve latest list of
               active rental properties. This list will be used to filter out
               ones that are no longer active (on the website)
    '''
    columns = df.columns
    numpy_df = df.values
    np_df = numpy_df.copy()

    for i in range(len(np_df)):
        if np_df[i][-2] > 1:
            for j in range(np_df[i][-2] - 1):
                numpy_df = np.vstack((numpy_df, np_df[[i]]))
                numpy_df[-1][-1] = np.datetime64(numpy_df[-1][-1]) + np.timedelta64(j + 1,'D')

    return_df = pd.DataFrame(numpy_df, columns=columns)

    #Drop unnecesary columns
    return_df = return_df.drop(['arrival_date', 'num_nights'], axis=1)

    #Add a new column to indicate occupancy
    return_df['occupied'] = 1

    return_df['date'] = pd.to_datetime(return_df['date'], format="%m/%d/%Y", errors='ignore')

    cascade_head = pd.read_csv('data/cascade_header.csv')
    cascade_head = cascade_head.drop('Unnamed: 0', axis=1)

    valid_prop = list(cascade_head.property_code.values)
    return_df = return_df[return_df['property_code'].isin(valid_prop)]

    return return_df

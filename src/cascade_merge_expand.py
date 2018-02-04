import pandas as pd
import os
import sys
from sqlalchemy import create_engine
home_path = os.environ['CASCADE_HOME']
sys.path.append(home_path + 'cascade/src')
from cascade_sql import write_to_table

dbname = os.environ['CASCADE_DB_DBNAME']
host = os.environ['CASCADE_DB_HOST']
username = os.environ['CASCADE_DB_USERNAME']
password = os.environ['CASCADE_DB_PASSWORD']
port = 5432

def merge_and_expand():
    '''
    This function will merge a dataframe that contains historical
    occupancy information per property_code (cascade_more.csv) with
    a dataframe that contains property features information
    (cascade_details.csv)

    Output: merged and exploded dataframe that account for all calendar days
    for a given property_code. For each property_code the calendar starts
    from its first occupancy date.
    '''

    #retrieving file that contains historic occupancy info
    cascade_more = pd.read_csv(home_path + 'data/cascade_more.csv',
                               infer_datetime_format=True, index_col=0)

    #processing file that contains property features
    #It turns out that following 8 properties should not be part of
    #this analysis
    remove_list = ['Poplar River Full Home (no loft)', 'Poplar River Full Home (with Loft)',
                   'Poplar River Master (No Loft)', 'Poplar River Master (With Loft)',
                   'Poplar River One Bedroom' ]

    # This is a file that was scraped from the reservation website
    # It contains property attribute information
    cascade_details = pd.read_csv(home_path + 'data/prop_head.csv', index_col=0)
    #records pertaining to remove_list are removed from dataframe
    cascade_details = cascade_details[~cascade_details['property_code'].isin(remove_list)]


    #Join cascade_more and cascade_detail dataframes on property_code
    #This new dataframe, cascade_join will contain a row that combines all
    #the information from cascade_more and add property features to it
    cascade_join = pd.merge(cascade_more, cascade_details,
                            on='property_code', how='inner')
    cascade_join.allows_pets = cascade_join.allows_pets.map(dict(Yes=1, No=0))


    #Retrieve unique property_code name so that we can loop through each
    #of them to fill any holes in calendar when the property was not
    #occupied
    unique_prop_code = sorted(list(cascade_join.property_code.unique()))

    #Columns to be included in the final_dataframe
    columns = ['property_code', 'property_city', 'property_zip', 'series',
               'num_guests', 'num_bedrooms', 'num_bathrooms', 'allows_pets',
               'property_size', 'manager_rating', 'property_rating']

    #Instantiating an empty dataframe
    final_df = pd.DataFrame()

    for prop_code in unique_prop_code:
        #Create a temp dataframe for each loop per property code
        temp_df = cascade_join[cascade_join['property_code'] == prop_code]
        #Set the index to be date
        temp_df.index = pd.to_datetime(temp_df['date'])

        #Another temp dataframe for resampling and foward filling of
        #records for all calendar date
        full_cal_df = pd.DataFrame()
        full_cal_df[columns] = temp_df[columns].resample('D').ffill()
        #set date column to equal the index which is calendar date.
        full_cal_df['date'] = full_cal_df.index

        #merge fully processed temporary dataframe to final dataframe.
        final_df = pd.concat([final_df, full_cal_df], axis=0)

    #drop the time (calendar date) index and reset it with numeric value
    final_df = final_df.reset_index(drop=True)

    #make sure that date column is of type datetime
    final_df['date'] = pd.to_datetime(final_df['date'])
    #create a new column year to only contain year number
    final_df['year'] = final_df['date'].dt.year
    final_df['month'] = final_df['date'].dt.month
    #create a new coulumn to determine whether a given date is
    #weekend or not. Weekend is defined as Fri, Sat, Sun
    final_df['weekend'] = ((final_df['date'].dt.dayofweek) // 4 == 1).astype(float)


    #make sure that date column is of type datetime
    cascade_more['date'] = pd.to_datetime(cascade_more['date'])


    #merge final datafram with cascade_more so that daily_rental_rate
    #and occupied can be added accurately to the final dataframe
    final_df = pd.merge(final_df,
                        cascade_more[['property_code', 'date',
                                      'daily_rental_rate', 'occupied']],
                        how='left',
                        on=['property_code','date'])

    #fill in NaN in occupied and daily_rental_rate columns with 0
    final_df.occupied = final_df.occupied.fillna(0)
    final_df.daily_rental_rate = final_df.daily_rental_rate.fillna(0)

    final_df.series = final_df.series.fillna('Moderate')
    final_df.property_zip = final_df.property_zip.fillna(55615.0)
    hovland_index = final_df[final_df['property_code'] == 'Hovland Pines'].index
    final_df.loc[hovland_index, ['property_city']] = 'Hovland'

    bluefin_index = final_df[final_df['property_code'].isin(['Bluefin Bay 14 Full Home', 'Bluefin Bay 14A', 'Bluefin Bay 14B',
       'Bluefin Bay 56 Full Home', 'Bluefin Bay 56A', 'Bluefin Bay 56B',
       'Bluefin Bay 57 Full Home', 'Bluefin Bay 57A', 'Bluefin Bay 57B'])].index
    final_df.loc[bluefin_index, ['property_city']] = 'Tofte'

    final_df['day'] = pd.to_datetime(final_df['date'])
    final_df.drop('date', axis=1)

    return final_df

def main():
    df = merge_and_expand()
    # df.to_csv(home_path+ 'data/cascade_expanded.csv')

    address = 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname)
    engine = create_engine(address)

    write_to_table(df, engine, 'cascade_hist', if_exists='replace')

if __name__ == '__main__':
    main()

import pandas as pd
import io
from sqlalchemy import create_engine

def write_to_table(df, db_engine, table_name, if_exists='fail'):
    '''
    Using code from user mgoldwasser's responde to the question on stackoverflow
    https://stackoverflow.com/questions/31997859/bulk-insert-a-pandas-dataframe-using-sqlalchemy

    Inputs
    df: input datafram to create/insert/append to the DataBase
    db_engine: instance of sqlalchemy that contains connect information to the
    database
    table_name: name of table that will be based on the passed in df
    if_exists: optional parameter has 3 values
    'fail' - default behavior if nothing is passed in. If the table exist already
    then this funciton will fail.
    'append' - table already exists and information in the df will be appended to
    the existing table in the database
    'replace' - will drop the existing table, create the new one and insert
    records from the df to the table.
    '''
    string_data_io = io.StringIO()
    df.to_csv(string_data_io, sep='|', index=False)
    pd_sql_engine = pd.io.sql.pandasSQL_builder(db_engine)
    table = pd.io.sql.SQLTable(table_name, pd_sql_engine, frame=df,
                               index=False, if_exists=if_exists)
    table.create()
    string_data_io.seek(0)
    string_data_io.readline()  # remove header
    with db_engine.connect() as connection:
        with connection.connection.cursor() as cursor:
            copy_cmd = "COPY %s FROM STDIN HEADER DELIMITER '|' CSV" % table_name
            cursor.copy_expert(copy_cmd, string_data_io)
        connection.connection.commit()

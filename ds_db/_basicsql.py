import urllib
import pyodbc
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, VARCHAR, DATETIME, INTEGER, FLOAT


def get_dtype(pandas_dtype):
    if pandas_dtype in ['object', 'str']:
        return VARCHAR
    elif pandas_dtype in ['datetime64[ns]']:
        return DATETIME
    elif pandas_dtype in ['int64', 'int32']:
        return INTEGER
    elif pandas_dtype in ['float64', 'float32']:
        return FLOAT
    else:
        return VARCHAR


class SQLConnection(object):
    def __init__(self, database, server, username, password, driver='{ODBC Driver 17 for SQL Server}'):
        '''
        Set up a connection between an SQL database (e.g., SQL Server) and the python instance.

        Parameters
        ----------
        database
        server
        username
        password
        driver : str
            Refer to the pyodbc docs for more information on this. (default: ODBC for SQL Server)
        '''
        self.database = database
        self.server = server
        self.username = username
        self.password = password
        self.driver = driver
        self.df_schema = None

        self.connection = pyodbc.connect(driver=driver,
                                         host=server,
                                         database=database,
                                         trusted_connection='yes',
                                         user=username,
                                         password=password)

    def upload_df(self, df, target_table, schema='dbo', if_table_exists='fail', to_sql_kws=None, preprocess_func=None,
                  coerce_dtypes=False):
        '''
        Upload a pandas DataFrame into a SQL database.

        Parameters
        ----------
        df : pandas.DataFrame

        target_table : str

        schema : str

        if_table_exists : str
            in {'fail', 'replace', or 'append'}

        to_sql_kws : dict

        preprocess_func : function

        coerce_dtypes: bool
            Use get_dtype function (in dstools.ds_db) to determine columns' data type

        Returns
        -------

        '''
        db_url = f"DRIVER={self.driver};\
                   SERVER={self.server};\
                   DATABASE={self.database};\
                   DRIVER={self.driver};\
                   TRUSTED_CONNECTION=yes;\
                   USER={self.username};\
                   PASSWORD={self.password}"

        db_url = urllib.parse.quote_plus(db_url)
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_url}')

        if preprocess_func is not None:
            df = preprocess_func(df)

        df.replace([np.nan], [None], inplace=True)

        if coerce_dtypes:
            to_sql_kws['dtype'] = {col: get_dtype(df[col].dtype) for col in df}

        print(f"Loading table (shape {df.shape}) into {target_table}. If the table exists, {if_table_exists}.")
        df.to_sql(target_table, schema=schema, con=engine, if_exists=if_table_exists, **to_sql_kws)

    def download_df(self, query, read_sql_kws=None):
        if read_sql_kws is None:
            read_sql_kws = {}

        df = pd.read_sql_query(query, self.connection, **read_sql_kws)
        return df

    def execute(self, sql_script):
        cursor = self.connection.cursor()
        cursor.execute(sql_script)
        self.connection.commit()
        cursor.close()

    def close(self):
        self.connection.close()

    def get_schema(self):
        query_schema = '''
        SELECT 
            TABLE_NAME AS table_name, 
            COLUMN_NAME AS column_name
        FROM 
            INFORMATION_SCHEMA.COLUMNS'''

        self.df_schema = pd.read_sql_query(query_schema, self.connection)

    def search_schema(self, in_table_name='', in_column_name=''):
        if self.df_schema is None:
            self.get_schema()

        return self.df_schema[(self.df_schema.table_name.str.lower().str.contains(in_table_name.lower())) &
                              (self.df_schema.column_name.str.lower().str.contains(in_column_name.lower()))]

    def upload_folder(self, folder_path, target_table=None, schema='dbo', csv=True, preprocess_func=None,
                      read_csv_kws=None, to_sql_kws=None, sort_columns=True, coerce_dtypes=False):
        if target_table is None:
            target_table = folder_path[folder_path.rfind('/') + 1:]

        files = os.listdir(folder_path)

        for i, file in enumerate(files):
            if csv:
                if read_csv_kws is None:
                    read_csv_kws = {}
                df = pd.read_csv(folder_path + '/' + file, **read_csv_kws)
            else:
                df = pd.read_pickle(folder_path + '/' + file)

            if sort_columns:
                df.sort_index(axis=1, inplace=True)

            print(f"[{i + 1} of {len(files)}] Appending {file} to {target_table} ...")
            self.upload_df(df, target_table, schema=schema, if_table_exists='append', to_sql_kws=to_sql_kws,
                           preprocess_func=preprocess_func, coerce_dtypes=coerce_dtypes)

    def __del__(self):
        self.connection.close()
import urllib
import pyodbc
import pandas as pd
from sqlalchemy import create_engine


class SQLConnection(object):
    def __init__(self, database, server, username, password, driver='{ODBC Driver 17 for SQL Server}'):
        self.database = database
        self.server = server
        self.username = username
        self.password = password
        self.driver = driver

        self.connection = pyodbc.connect(driver=driver,
                                         host=server,
                                         database=database,
                                         trusted_connection='yes',
                                         user=username,
                                         password=password)

    def upload_table(self, df, target_table, schema='dbo', if_table_exists='fail'):
        '''
        Upload a pandas table into a SQL database.

        Parameters
        ----------
        df : pandas.DataFrame

        target_table : str

        schema : str

        if_table_exists : str
            in {'fail', 'replace', or 'append'}

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

        print(f"Loading table (shape {df.shape}) into {target_table}. If the table exists, {if_table_exists}.")
        df.to_sql(target_table, schema=schema, con=engine, if_exists=if_table_exists)

    def download_table(self, query):
        df = pd.read_sql_query(query, self.connection)
        return df

    def execute(self, sql_script):
        cursor = self.connection.cursor()
        cursor.execute(sql_script)
        self.connection.commit()
        cursor.close()

    def close(self):
        self.connection.close()

    def __del__(self):
        self.connection.close()
# Connections

## Snowflake

blah blah

## Microsoft SQL Server

### On Windows

```python
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = 'main\server\path'
database = 'db1'
username = 'username'
password = 'password'
cnxn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',  # You can get the most recent one from pyodbc docs
                      host=server, 
                      database=database,
                      trusted_connection='yes', 
                      user=username,
                      password=password)

# cursor = cnxn.cursor()

query = '''
SELECT
    blah
FROM
    blah;'''

df = pd.read_sql_query(query, cnxn)
```
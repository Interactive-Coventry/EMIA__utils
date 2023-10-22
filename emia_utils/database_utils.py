import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
import pandas as pd
from libs.foxutils.utils.core_utils import settings

# read connection parameters
host = settings["DATABASE"]["host"]
dbname = settings["DATABASE"]["dbname"]
user = settings["DATABASE"]["user"]
password = settings["DATABASE"]["password"]


def engine_connect():
    # establish connections
    conn_string = f'postgresql://{user}:{password}@{host}/{dbname}'
    db = create_engine(conn_string)
    conn = db.connect()
    return conn


def connect():
    conn = None
    try:
        # conn = psycopg2.connect(f"dbname={dbname} user={user} password={password}")
        conn = psycopg2.connect(
            host=host,
            database=dbname,
            user=user,
            password=password)

        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None


def execute_commands(commands):
    conn = None
    try:
        conn = connect()
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def execute_command(command):
    execute_commands([command])


def execute_command_and_do_something(command, target_function, **kwargs):
    conn = None
    result = None
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute(command)
        result = target_function(cur, **kwargs)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
        return result


def fetch_one(cur):
    result = cur.fetchone()
    return result


def check_connection():
    """ Connect to the PostgreSQL database server """
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database...')

    # execute a statement
    print('PostgreSQL database version:')
    command = 'SELECT version()'
    db_version = execute_command_and_do_something(command, fetch_one)
    print(db_version)


def drop_table(table_name):
    print(f'Dropping table: {table_name}')
    command = f'drop table if exists {table_name}'
    execute_command(command)

def create_vehicle_count_table():
    commands = (
        """
        CREATE TABLE vehicle_count (
            datetime TIMESTAMP WITH TIME ZONE,
            camera_id VARCHAR(20),
            total_pedestrians INT4,
            total_vehicles INT4,
            bicycle INT4,
            bus INT4,
            motorcycle INT4,
            person INT4,
            truck INT4,
            PRIMARY KEY (datetime, camera_id)
        );
        """,
    )
    execute_commands(commands)

def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE weather (
            datetime TIMESTAMP WITH TIME ZONE PRIMARY KEY,
            temp FLOAT8, 
            feels_like FLOAT4,
            temp_min FLOAT4,
            temp_max FLOAT4,
            pressure INT4,
            humidity INT4,
            wind_speed FLOAT4,
            wind_deg INT4, 
            clouds_all INT4, 
            visibility INT4, 
            lat	FLOAT8, 
            lon	FLOAT8, 
            dt INT8, 
            timezone INT4
        )
        """,
    )
    execute_commands(commands)


def retrieve_primary_key(table_name):
    command = f"SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS data_type\nFROM pg_index i\n" \
              f"JOIN pg_attribute a ON a.attrelid = i.indrelid  AND a.attnum = ANY(i.indkey)\n" \
              f"WHERE i.indrelid = '{table_name}'::regclass AND i.indisprimary;"
    result = execute_command_and_do_something(command, fetch_one)
    if result is not None:
        return result[0]
    return None


def set_primary_key_from_df(df, table_name):
    current_primary_key = retrieve_primary_key(table_name)
    if current_primary_key is not None and current_primary_key != df.index.name:
        command = f'ALTER TABLE {table_name} ADD PRIMARY KEY ( {df.index.name});'
        execute_command(command)


def replace_df_to_table(df, table_name):
    df.to_sql(table_name, engine_connect(), if_exists='replace')
    set_primary_key_from_df(df, table_name)


def append_df_to_table(df, table_name, append_only_new=True):
    try:
        if append_only_new:
            index = df.index.name
            db_start_index, db_end_index = get_min_max_primary_key(table_name, index)
            print(
                f'Table [{table_name}] with index [{index}] has start index {db_start_index} and end index {db_end_index}.')
            df = df.loc[(df.index < db_start_index) | (df.index > db_end_index)]

        if len(df) > 0:
            df.to_sql(table_name, engine_connect(), if_exists='append', schema='public', chunksize=50)
            set_primary_key_from_df(df, table_name)
            print(f'Appending values outside current bounds only (Total new values: {len(df)}).')

        else:
            print('Nothing to append.')

    except IntegrityError:
        print("Can't append, because key exists")


def get_min_max_primary_key(table_name, id_name=None):
    if id_name is None:
        id_name = retrieve_primary_key(table_name)

    command = f'SELECT MIN({id_name}), MAX({id_name})  from {table_name};'
    result = execute_command_and_do_something(command, fetch_one)
    return result[0], result[1]


def get_timezone():
    timezone = 'Asia/Singapore'
    column = 'datetime'
    table_name = 'weather'
    command = f'select {column} AT TIME ZONE {timezone} from {table_name};'


def enclose_in_quotes(input_str):
    output_str = "'" + str(input_str) + "'"
    return output_str


def read_table_with_select(table_name, params=None, conn=None):
    command = f'SELECT * FROM {table_name} '
    if len(params) > 0:
        where_clause = 'WHERE '
        command = command + where_clause
        for vals in params:
            command = command + ' '.join(vals) + ' '

    if conn is None:
        conn = engine_connect()
    df = pd.read_sql(text(command), conn)
    return df


if __name__ == '__main__':
    check_connection()

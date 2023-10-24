import logging

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine.base import Connection
import pandas as pd
from libs.foxutils.utils.core_utils import settings

logger = logging.getLogger("emia_utils.database_utils")

def get_connection_parameters(host=None, dbname=None, user=None, password=None):
    if settings["TOKENS"]["read_from"] == "local":
        logger.debug(f"Reading from local settings")
        if host is None:
            host = settings["DATABASE"]["host"]
        if dbname is None:
            dbname = settings["DATABASE"]["dbname"]
        if user is None:
            user = settings["DATABASE"]["user"]
        if password is None:
            password = settings["DATABASE"]["password"]

    elif settings["TOKENS"]["read_from"] == "secrets":
        import streamlit as st
        logger.debug(f"Reading from secrets")
        if host is None:
            host = st.secrets.connections.postgresql.host
        if dbname is None:
            dbname = st.secrets.connections.postgresql.database
        if user is None:
            user = st.secrets.connections.postgresql.username
        if password is None:
            password = st.secrets.connections.postgresql.password

    print(f"Connecting to {host} {dbname} {user} {password}")
    return host, dbname, user, password


def engine_connect():
    host, dbname, user, password = get_connection_parameters()
    conn_string = f'postgresql://{user}:{password}@{host}/{dbname}'
    db = create_engine(conn_string)
    conn = db.connect()
    return conn


def connect(host=None, dbname=None, user=None, password=None):
    conn = None
    try:
        host, dbname, user, password = get_connection_parameters(host, dbname, user, password)

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
        CREATE TABLE vehicle_counts (
            datetime TIMESTAMP WITHOUT TIME ZONE,
            camera_id VARCHAR(20),
            total_pedestrians INT4,
            total_vehicles INT4,
            bicycle INT4,
            bus INT4,
            motorcycle INT4,
            person INT4,
            truck INT4,
            car INT4,
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
            datetime TIMESTAMP WITHOUT TIME ZONE PRIMARY KEY,
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
            if db_start_index is not None and db_end_index is not None:
                logger.debug(
                    f'Table [{table_name}] with index [{index}] has start index {db_start_index} and end index {db_end_index}.')
                df = df.loc[(df.index.to_pydatetime() < db_start_index) | (df.index.to_pydatetime() > db_end_index)]

        if len(df) > 0:
            df.to_sql(table_name, engine_connect(), if_exists='append', schema='public', chunksize=50)
            set_primary_key_from_df(df, table_name)
            logger.debug(f'Appending values outside current bounds only (Total new values: {len(df)}).')

        else:
            logger.debug('Nothing to append.')

    except IntegrityError as e:
        logger.info(f"IntegrityError: {e}")
        logger.info("Can't append, because key exists")


def get_min_max_primary_key(table_name, id_name=None):
    if id_name is None:
        id_name = retrieve_primary_key(table_name)

    command = f'SELECT MIN({id_name}), MAX({id_name})  from {table_name};'
    result = execute_command_and_do_something(command, fetch_one)
    if result is None:
        None, None
    else:
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

    if isinstance(conn, Connection): # For SQLAlchemy
        command = text(command)
        df = pd.read_sql(command, conn)
    if type(conn).__name__ == "SQLConnection": # For streamlit
        df = conn.query(command)

    return df


if __name__ == '__main__':
    check_connection()

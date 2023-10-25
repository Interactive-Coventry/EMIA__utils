import logging

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine.base import Connection
import pandas as pd
from libs.foxutils.utils.core_utils import settings

logger = logging.getLogger("emia_utils.database_utils")

READ_DB_CREDENTIALS_FROM = settings["TOKENS"]["read_from"]

def get_connection_parameters(host=None, port=None, dbname=None, user=None, password=None):
    if READ_DB_CREDENTIALS_FROM == "local":
        logger.debug(f"Reading from local settings")
        if host is None:
            host = settings["DATABASE"]["host"]
        if port is None:
            port = settings["DATABASE"]["port"]
        if dbname is None:
            dbname = settings["DATABASE"]["dbname"]
        if user is None:
            user = settings["DATABASE"]["user"]
        if password is None:
            password = settings["DATABASE"]["password"]

    elif READ_DB_CREDENTIALS_FROM == "secrets":
        import streamlit as st
        logger.debug(f"Reading from secrets")
        if host is None:
            host = st.secrets.connections.postgresql.host
        if port is None:
            port = st.secrets.connections.postgresql.port
        if dbname is None:
            dbname = st.secrets.connections.postgresql.database
        if user is None:
            user = st.secrets.connections.postgresql.username
        if password is None:
            password = st.secrets.connections.postgresql.password

    else:
        raise ValueError(f"No connection to database for settings {READ_DB_CREDENTIALS_FROM}.")

    return host, port, dbname, user, password


def engine_connect():
    host, port, dbname, user, password = get_connection_parameters()
    conn_string = f"postgresql://[{host}]:{port}/{dbname}?user={user}&password={password}"
    db = create_engine(conn_string)
    conn = db.connect()
    logger.debug(f"Engine connect:Connecting to {conn} from secrets.")

    return conn


def connect_with_psycopg2(host=None, port=None, dbname=None, user=None, password=None):
    host, port, dbname, user, password = get_connection_parameters(host, port, dbname, user, password)
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=dbname,
        user=user,
        password=password)
    logger.debug(f"Connecting to {conn} from {READ_DB_CREDENTIALS_FROM}.")
    return conn

def connect(host=None, port=None, dbname=None, user=None, password=None):
    try:
        if READ_DB_CREDENTIALS_FROM == "local":
            conn = connect_with_psycopg2(host, port, dbname, user, password)

        elif READ_DB_CREDENTIALS_FROM == "secrets":
            #import streamlit as st
            #conn = st.experimental_connection("postgresql", type="sql")
            conn = connect_with_psycopg2(host, port, dbname, user, password)

        else:
            raise ValueError(f"No connection to database for settings {READ_DB_CREDENTIALS_FROM}.")

        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
        return None


def execute_commands(commands):
    try:
        conn = connect()
        logger.debug(f"Executing commands: {commands}")
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
    finally:
        if conn is not None:
            conn.close()


def execute_command(command):
    execute_commands([command])


def execute_command_and_do_something(command, target_function, **kwargs):
    result = None
    try:
        conn = connect()
        logger.debug(f"Executing command: {command}")
        cur = conn.cursor()
        cur.execute(command)
        result = target_function(cur, **kwargs)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
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
    logger.info('Connecting to the PostgreSQL database...')

    # execute a statement
    logger.info('PostgreSQL database version:')
    command = 'SELECT version()'
    db_version = execute_command_and_do_something(command, fetch_one)
    logger.info(db_version)


def drop_table(table_name):
    logger.info(f'Dropping table: {table_name}')
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
        logger.error(f"IntegrityError: {e}")
        logger.error("Can't append, because key exists")


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

    logger.debug(f"Reading table with select: {command}")
    if isinstance(conn, Connection): # For SQLAlchemy
        command = text(command)
        df = pd.read_sql(command, conn)
    if type(conn).__name__ == "SQLConnection": # For streamlit
        df = conn.query(command)

    return df


if __name__ == '__main__':
    check_connection()

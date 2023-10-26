import logging

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine.base import Connection
import pandas as pd
from libs.foxutils.utils.core_utils import settings

logger = logging.getLogger("emia_utils.database_utils")

READ_DB_CREDENTIALS_FROM = settings["TOKENS"]["read_from"]
DB_MODE = settings["DATABASE"]["db_mode"]  # "local" or "streamlit" or "firebase"
USES_STREAMLIT = DB_MODE == "streamlit"
USES_FIREBASE = DB_MODE == "firebase"


def init_firebase():
    import firebase_admin
    from firebase_admin import credentials
    from google.oauth2 import service_account
    import streamlit as st
    from google.cloud import firestore

    key_dict = dict(st.secrets["firebase"])
    FIREBASE_PROJECT_NAME = settings["FIREBASE"]["project_name"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project=FIREBASE_PROJECT_NAME)
    return db


def collection_reference_to_dataframe(db_collection, is_list=False):
    if not is_list:
        table = list(db_collection.stream())
    else:
        table = db_collection
    table_dict = list(map(lambda x: x.to_dict(), table))
    df = pd.DataFrame(table_dict)
    return df


def insert_row_to_firebase(db, row_dict, table_name, id_name=None):
    if id_name is None:
        update_time, added_ref = db.collection(table_name).add(row_dict)
        logger.debug(f"Added document with id {added_ref.id} to {table_name} at {update_time}.")
    else:
        document_id = row_dict[id_name]
        doc_ref = db.collection(table_name).document(document_id)
        doc = doc_ref.get()
        if doc.exists:
            logger.debug(f"Document with id {document_id} already exists in {table_name}.")
        else:
            db.collection(table_name).document(document_id).set(row_dict)
            logger.debug(f"Added document with id {document_id} to {table_name}.")



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

    elif USES_STREAMLIT or READ_DB_CREDENTIALS_FROM == "secrets":
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
    db = create_engine(conn_string)  # , pool_pre_ping=True
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
    logger.info(f"Connecting to {conn} from {READ_DB_CREDENTIALS_FROM}.")
    return conn


def connect(host=None, port=None, dbname=None, user=None, password=None):
    try:
        if READ_DB_CREDENTIALS_FROM == "local":
            conn = connect_with_psycopg2(host, port, dbname, user, password)

        elif USES_STREAMLIT:
            import streamlit as st
            conn = st.experimental_connection("postgresql", type="sql")

        else:
            raise ValueError(f"No connection to database for settings {READ_DB_CREDENTIALS_FROM}.")

        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
        return None


def execute_commands(commands, target_function=None, **kwargs):
    results = None
    try:
        conn = connect()
        cur = conn.cursor()
        for command in commands:
            logger.debug(f"Executing command: {command}")
            cur.execute(command)
            if target_function is not None:
                result = target_function(cur, **kwargs)
                results.append(result)
        cur.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
    finally:
        if conn is not None:
            conn.close()
        return results


def execute_command(command, target_function=None, **kwargs):
    results = execute_commands([command], target_function, **kwargs)
    return results[0]


def query_with_streamlit(command, conn=None):
    df = None
    if conn is None:
        conn = connect()
    df = conn.query(command)
    return df


def execute_command_with_streamlit(command, conn=None):
    if conn is None:
        conn = connect()
    with conn.session as s:
        s.execute(command)
        s.commit()


def fetch_one(cur):
    result = cur.fetchone()
    return result


def check_connection(conn=None):
    """ Connect to the PostgreSQL database server """
    # connect to the PostgreSQL server
    logger.info('Connecting to the PostgreSQL database...')

    # execute a statement
    command = 'SELECT version()'
    if USES_STREAMLIT:
        df = query_with_streamlit(command, conn)
        if df is not None:
            db_version = df.iloc[0]["version"]
    else:
        db_version = execute_command(command, fetch_one)
    logger.info(f"PostgreSQL database version: {db_version}")


def drop_table(table_name):
    logger.info(f"Dropping table: {table_name}")
    command = f"drop table if exists {table_name}"
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


def retrieve_primary_key(table_name, conn=None):
    command = f"SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS data_type\nFROM pg_index i\n" \
              f"JOIN pg_attribute a ON a.attrelid = i.indrelid  AND a.attnum = ANY(i.indkey)\n" \
              f"WHERE i.indrelid = '{table_name}'::regclass AND i.indisprimary;"

    result = None
    if USES_STREAMLIT:
        df = query_with_streamlit(command, conn)
        if df is not None:
            result = df.iloc[0]["attname"]
    else:
        result = execute_command(command, fetch_one)
        if result is not None:
            result = result[0]
    return result


def set_primary_key_from_df(df, table_name, conn=None):
    current_primary_key = retrieve_primary_key(table_name)
    if current_primary_key is not None and current_primary_key != df.index.name:
        command = f"ALTER TABLE {table_name} ADD PRIMARY KEY ( {df.index.name});"
        if USES_STREAMLIT:
            execute_command_with_streamlit(command, conn)
        else:
            execute_command(command)


def replace_df_to_table(df, table_name, conn=None):
    if USES_STREAMLIT:
        df.to_sql(table_name, engine_connect(), if_exists="replace")
        set_primary_key_from_df(df, table_name, conn)

    else:
        df.to_sql(table_name, engine_connect(), if_exists="replace")
        set_primary_key_from_df(df, table_name)


def append_df_to_table(df, table_name, append_only_new=True, conn=None):
    try:
        if append_only_new:
            index = df.index.name
            db_start_index, db_end_index = get_min_max_primary_key(table_name, index, conn)
            if db_start_index is not None and db_end_index is not None:
                logger.debug(
                    f"Table [{table_name}] with index [{index}] has start index {db_start_index} and end index {db_end_index}.")
                df = df.loc[(df.index.to_pydatetime() < db_start_index) | (df.index.to_pydatetime() > db_end_index)]

        if len(df) > 0:
            if USES_STREAMLIT:
                df.to_sql(table_name, engine_connect(), if_exists="append", schema="public", chunksize=50)
                logger.debug(f"Streamlit connect: appended {len(df)}")
                set_primary_key_from_df(df, table_name, conn)
            else:
                df.to_sql(table_name, engine_connect(), if_exists="append", schema="public", chunksize=50)
                set_primary_key_from_df(df, table_name)
            logger.debug(f"Appended values outside current bounds only (Total new values: {len(df)}).")

        else:
            logger.debug("Nothing to append.")

    except IntegrityError as e:
        logger.debug(f"IntegrityError: {e}")
        logger.error("Can't append, because key exists")


def get_min_max_primary_key(table_name, id_name=None, conn=None):
    if id_name is None:
        id_name = retrieve_primary_key(table_name)

    min = None
    max = None
    command = f"SELECT MIN({id_name}), MAX({id_name})  from {table_name};"
    if USES_STREAMLIT:
        df = query_with_streamlit(command, conn)
        if df is not None:
            min = df.iloc[0]["min"]
            max = df.iloc[0]["max"]
    else:
        result = execute_command(command, fetch_one)
        if result:
            min = result[0]
            max = result[1]

    return min, max


def get_timezone():
    timezone = "Asia/Singapore"
    column = "datetime"
    table_name = "weather"
    command = f"select {column} AT TIME ZONE {timezone} from {table_name};"
    execute_command(command)


def enclose_in_quotes(input_str):
    output_str = "'" + str(input_str) + "'"
    return output_str


def read_table_with_select(table_name, params=None, conn=None):
    command = f"SELECT * FROM {table_name} "
    if len(params) > 0:
        where_clause = "WHERE "
        command = command + where_clause
        for vals in params:
            command = command + ' '.join(vals) + ' '
    logger.debug(f"Reading table with select: {command}")

    if USES_STREAMLIT:
        df = conn.query(command)
    else:
        if conn is None:
            conn = engine_connect()
        # if isinstance(conn, Connection):  # For SQLAlchemy
        command = text(command)
        df = pd.read_sql(command, conn)
    return df


if __name__ == "__main__":
    check_connection()

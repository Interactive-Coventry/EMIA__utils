import logging
import socket

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from libs.foxutils.utils.core_utils import settings

logger = logging.getLogger("emia_utils.database_utils")

from .configuration import (
    ANOMALY_TYPE_KEY_NAME, WEATHER_TYPE_KEY_NAME, WETNESS_TYPE_KEY_NAME,
    DATETIME_KEY_NAME, CAMERA_ID_KEY_NAME, VEHICLE_COUNTS_TABLE_NAME,
    DASHCAM_TABLE_NAME, WEATHER_TABLE_NAME, IMAGE_ANALYSIS_TABLE_NAME,
    WEATHER_DICT, WETNESS_DICT, ANOMALY_DICT, CAMERA_INFO_TABLE_NAME
)
from .process_utils import prepare_features_for_vehicle_counts

READ_DB_CREDENTIALS_FROM = settings["TOKENS"]["read_from"]  # "local" or "secrets"
DB_MODE = settings["DATABASE"]["db_mode"]  # "local" or "streamlit" or "firebase"
USES_STREAMLIT, USES_FIREBASE = DB_MODE == "streamlit", DB_MODE == "firebase"

if USES_FIREBASE:
    from google.cloud import firestore

logger.debug(
    f"READ_DB_CREDENTIALS_FROM: {READ_DB_CREDENTIALS_FROM}, USES_STREAMLIT: {USES_STREAMLIT}, USES_FIREBASE: {USES_FIREBASE}")


def init_firebase():
    """
    Initialize Firebase connection using service account credentials.
    :return: Firestore client object.
    """
    from google.oauth2 import service_account
    import streamlit as st
    creds = service_account.Credentials.from_service_account_info(dict(st.secrets["firebase"]))
    return firestore.Client(credentials=creds, project=settings["FIREBASE"]["project_name"])


def init_connection():  # For psycopg2 connections
    """
    Initialize the connection to the PostgreSQL database.
    :return: psycopg2 connection object.
    """
    import socket
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    logger.info(f"Your Computer Name is: {hostname} and IP Address is:{IPAddr}")

    conn_ = connect()
    logger.debug(f"The type of connection is {type(conn_)}")
    logger.info(f"DB connection is {conn_}")
    check_connection(conn_)
    return conn_


def collection_reference_to_dataframe(db_collection, is_list=False):
    """
    Converts a Firestore collection reference to a pandas DataFrame.
    :param db_collection: Firestore collection reference or list of documents.
    """
    if not is_list:
        table = list(db_collection.stream())
    else:
        table = db_collection
    table_dict = list(map(lambda x: x.to_dict(), table))
    df = pd.DataFrame(table_dict)
    return df


def insert_row_to_firebase(db, row_dict, table_name, id_name=None):
    """
    Inserts a row into a Firestore collection.
    :param db: Firestore client object.
    """
    if id_name is None:
        update_time, added_ref = db.collection(table_name).add(row_dict)
        logger.debug(f"Added document with id {added_ref.id} to {table_name} at {update_time}.")
    else:
        if isinstance(id_name, list):
            document_id = f"{row_dict['datetime']}"
            for x in id_name[1:]:
                document_id += f"_{row_dict[x]}"
        else:
            document_id = str(row_dict[id_name])

        if db is None:
            raise ValueError("Connection must be provided for Firebase.")

        doc_ref = db.collection(table_name).document(document_id)
        doc = doc_ref.get()
        if doc.exists:
            logger.debug(f"Document with id {document_id} already exists in {table_name}.")
        else:
            db.collection(table_name).document(document_id).set(row_dict)
            logger.debug(f"Added document with id {document_id} to {table_name}.")


def get_connection_parameters():
    """
    Retrieves the database connection parameters based on the configured source.
    :return: Tuple containing host, port, dbname, user, password.
    """
    if READ_DB_CREDENTIALS_FROM == "local":
        return tuple(settings["DATABASE"].get(k) for k in ["host", "port", "dbname", "user", "password"])
    elif USES_STREAMLIT or READ_DB_CREDENTIALS_FROM == "secrets":
        import streamlit as st
        return (
            st.secrets.connections.postgresql.host,
            st.secrets.connections.postgresql.port,
            st.secrets.connections.postgresql.database,
            st.secrets.connections.postgresql.username,
            st.secrets.connections.postgresql.password
        )
    else:
        raise ValueError(f"Invalid DB credentials source: {READ_DB_CREDENTIALS_FROM}")


def engine_connect():
    """
    Creates and returns a SQLAlchemy engine connection.

    Uses connection parameters from the configuration or Streamlit secrets to build the connection string.
    """
    try:
        host, port, dbname, user, password = get_connection_parameters()
        conn_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        db = create_engine(conn_string)  # You can configure pool size here if needed
        logger.debug(f"Engine connect: Connecting to {conn_string}.")
        return db.connect()
    except Exception as e:
        logger.error(f"Failed to create engine connection: {e}")
        raise


def connect_to_postgres():
    """
    Establish a connection to a local PostgreSQL database using psycopg2.
    :return: psycopg2 connection object or None if an error occurs.
    """
    try:
        logger.debug("Connecting to local PostgreSQL database...")
        params = get_connection_parameters()
        return psycopg2.connect(
            host=params[0],
            port=params[1],
            database=params[2],
            user=params[3],
            password=params[4]
        )
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error connecting to PostgreSQL: {error}")
        return None


def connect_to_streamlit():
    """
    Establish a connection to a database using Streamlit's connection utilities.
    :return: Streamlit connection object or None if an error occurs.
    """
    try:
        import streamlit as st
        try:
            return st.connection("postgresql", type="sql")
        except AttributeError:
            # Fallback for older Streamlit versions
            return st.experimental_connection("postgresql", type="sql")
    except Exception as error:
        logger.error(f"Error connecting via Streamlit: {error}")
        return None


def connect(host=None, port=None, dbname=None, user=None, password=None):
    """
    Establish a database connection based on the configured credentials source.
    :return: Connection object or None if an error occurs.
    """
    try:
        if READ_DB_CREDENTIALS_FROM == "local":
            return connect_to_postgres()
        elif USES_STREAMLIT:
            return connect_to_streamlit()
        else:
            raise ValueError(f"No connection to database for settings {READ_DB_CREDENTIALS_FROM}.")
    except Exception as error:
        logger.error(f"Error in database connection: {error}")
        return None


def execute_commands(commands, target_function=None, **kwargs):
    """
    Execute a list of SQL commands on the PostgreSQL database.
    :param commands: List of SQL commands to execute.
    :param target_function: Optional function to process the results of each command.
    :param kwargs: Additional keyword arguments to pass to the target function.
    :return: List of results from the target function for each command.
    """
    results = []
    try:
        conn = connect()
        cur = conn.cursor()
        for command in commands:
            logger.debug(f"Executing command: {command}")
            cur.execute(command)
            if target_function is not None:
                result = target_function(cur, **kwargs)
                if isinstance(result, tuple):
                    result = list(result)
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
    """
    Execute a single SQL command on the PostgreSQL database.
    :param command: SQL command to execute.
    :param target_function: Optional function to process the result of the command.
    :param kwargs: Additional keyword arguments to pass to the target function.
    :return: Result from the target function or None if no function is provided.
    """
    results = execute_commands([command], target_function, **kwargs)
    return results[0]


def query_with_streamlit(command, conn=None):
    """
    Execute a SQL query and return the result as a DataFrame.
    :param command: SQL command to execute.
    :param conn: Streamlit database connection object.
    :return: DataFrame containing the query result.
    """
    df = None
    if conn is None:
        conn = connect()
    with conn.session as s:
        try:
            # df = s.query(text(command))
            df = pd.read_sql(command, s.bind)

        finally:
            s.close()

    return df


def execute_command_with_streamlit(command, conn=None):
    """
    Execute a SQL command using a Streamlit connection.
    :param command: SQL command to execute.
    :param conn: Streamlit database connection object.
    :return: None
    """
    if conn is None:
        conn = connect()
    with conn.session as s:
        try:
            s.execute(text(command))
            s.commit()
        finally:
            s.close()


def fetch_one(cur):
    result = cur.fetchone()
    return result


def get_postgresql_version_with_streamlit(command, conn):
    """
    Fetch the PostgreSQL version using a Streamlit connection.
    :param command: SQL command to execute.
    :param conn: Streamlit database connection object.
    :return: PostgreSQL version as a string, or None if an error occurs.
    """
    df = query_with_streamlit(command, conn)
    if df is not None and not df.empty:
        return df.iloc[0]["version"]
    else:
        logger.warning("Failed to fetch PostgreSQL version using Streamlit.")
        return None


def get_postgresql_version_with_standard_connection(command):
    """
    Fetch the PostgreSQL version using a standard PostgreSQL connection.
    :param command: SQL command to execute.
    :return: PostgreSQL version as a string, or None if an error occurs.
    """
    return execute_command(command, fetch_one)


def check_connection(conn=None):
    """
    Check the connection to the PostgreSQL database server and log the version.
    :param conn: Database connection object (optional, for Streamlit connections).
    """
    logger.info('Connecting to the PostgreSQL database...')
    command = 'SELECT version()'

    if USES_STREAMLIT:
        db_version = get_postgresql_version_with_streamlit(command, conn)
    else:
        db_version = get_postgresql_version_with_standard_connection(command)

    if db_version:
        logger.info(f"PostgreSQL database version: {db_version}")
    else:
        logger.error("Failed to determine PostgreSQL version.")


def drop_table(table_name):
    """
    Drop a table from the PostgreSQL database.
    :param table_name: Name of the table to drop.
    """
    logger.info(f"Dropping table: {table_name}")
    command = f"drop table if exists {table_name}"
    execute_command(command)


def create_dashcams_table():
    commands = (
        """
        CREATE TABLE dashcams (
            datetime TIMESTAMP WITHOUT TIME ZONE,
            camera_id VARCHAR(50) NOT NULL,
            lat REAL,
            lng REAL,
            PRIMARY KEY (datetime, camera_id)
        );

        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO elena;
        GRANT ALL PRIVILEGES ON TABLE dashcams TO elena;
        """,
    )
    execute_commands(commands)


def create_vehicle_count_table():
    commands = (
        """
        CREATE TABLE vehicle_counts (
            datetime TIMESTAMP WITHOUT TIME ZONE,
            camera_id VARCHAR(50),
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


def retrieve_primary_key_with_streamlit(command, conn):
    """
    Retrieve the primary key of a table using a Streamlit connection.
    :param command: SQL command to execute.
    :param conn: Streamlit database connection object.
    :return: Primary key column name as a string, or None if not found.
    """
    df = query_with_streamlit(command, conn)
    if df is not None and not df.empty:
        return df.iloc[0]["attname"]
    else:
        logger.warning("Failed to retrieve primary key using Streamlit.")
        return None


def retrieve_primary_key_with_standard_connection(command):
    """
    Retrieve the primary key of a table using a standard PostgreSQL connection.
    :param command: SQL command to execute.
    :return: Primary key column name as a string, or None if not found.
    """
    result = execute_command(command, fetch_one)
    return result[0] if result else None


def retrieve_primary_key(table_name, conn=None):
    """
    Retrieve the primary key column name for a given table.
    :param table_name: Name of the table to query.
    :param conn: Database connection object (optional, for Streamlit connections).
    :return: Primary key column name as a string, or None if not found.
    """
    command = (
        f"SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS data_type\n"
        f"FROM pg_index i\n"
        f"JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)\n"
        f"WHERE i.indrelid = '{table_name}'::regclass AND i.indisprimary;"
    )

    if USES_STREAMLIT:
        return retrieve_primary_key_with_streamlit(command, conn)
    else:
        return retrieve_primary_key_with_standard_connection(command)


def set_primary_key_from_df(df, table_name, conn=None):
    """
    Set the primary key for a table based on the DataFrame's index name.
    :param df: DataFrame containing the data.
    :param table_name: Name of the table to modify.
    :param conn: Database connection object (optional, for Streamlit connections).
    """
    current_primary_key = retrieve_primary_key(table_name)
    if current_primary_key is not None and current_primary_key != df.index.name:
        command = f"ALTER TABLE {table_name} ADD PRIMARY KEY ( {df.index.name});"
        if USES_STREAMLIT:
            execute_command_with_streamlit(command, conn)
        else:
            execute_command(command)


def replace_df_to_table(df, table_name, conn=None):
    """
    Replace the contents of a table with the DataFrame's data.
    :param df: DataFrame containing the data.
    :param table_name: Name of the table to replace.
    :param conn: Database connection object (optional, for Streamlit connections).
    :return: None
    """
    df.to_sql(table_name, engine_connect(), if_exists="replace")
    set_primary_key_from_df(df, table_name, conn)


def append_df_to_table(df, table_name, append_only_new=True, conn=None, append_index=True):
    """
    Append a DataFrame to a PostgreSQL table.
    :param df: DataFrame to append.
    :param table_name: Name of the target table.
    :param append_only_new: Whether to append only new rows.
    :param conn: Database connection object (optional, for Streamlit connections).
    :param append_index: Whether to include the DataFrame's index as a column.
    :return: None
    """
    try:
        if append_only_new:
            if append_index:
                index = df.index.name
                db_start_index, db_end_index = get_min_max_primary_key(table_name, index, conn)
                if db_start_index is not None and db_end_index is not None:
                    logger.debug(
                        f"Table [{table_name}] with index [{index}] has start index {db_start_index} and end index {db_end_index}.")
                    df = df.loc[(df.index.to_pydatetime() < db_start_index) | (df.index.to_pydatetime() > db_end_index)]

        if len(df) > 0:
            df.to_sql(table_name, engine_connect(), if_exists="append", schema="public", chunksize=50,
                      index=append_index)
            logger.debug(f"Streamlit connect: appended {len(df)}")
            set_primary_key_from_df(df, table_name, conn)
            logger.debug(f"Appended values outside current bounds only (Total new values: {len(df)}).")
        else:
            logger.debug("Nothing to append.")

    except IntegrityError as e:
        logger.debug(f"IntegrityError: {e}")
        logger.error("Can't append, because key exists")

    except ProgrammingError as e:
        logger.debug(f"ProgrammingError: {e}")
        logger.debug("Permission denied for sequence dashcams_event_id_seq")


def get_min_max_with_streamlit(table_name, id_name, conn):
    """
    Retrieves the minimum and maximum primary key values using a Streamlit connection.
    :param table_name: Name of the table.
    :param id_name: Name of the primary key column.
    :param conn: Streamlit database connection object.
    :return: Tuple (min, max) of primary key values.
    """
    command = f"SELECT MIN({id_name}) AS min, MAX({id_name}) AS max FROM {table_name};"
    df = query_with_streamlit(command, conn)
    if df is not None and not df.empty:
        return df.iloc[0]["min"], df.iloc[0]["max"]
    else:
        logger.warning(f"Failed to retrieve min and max primary key values for table {table_name} using Streamlit.")
        return None, None


def get_min_max_with_standard_connection(table_name, id_name):
    """
    Retrieves the minimum and maximum primary key values using a standard PostgreSQL connection.
    :param table_name: Name of the table.
    :param id_name: Name of the primary key column.
    :return: Tuple (min, max) of primary key values.
    """
    command = f"SELECT MIN({id_name}), MAX({id_name}) FROM {table_name};"
    result = execute_command(command, fetch_one)
    if result:
        return result[0], result[1]
    else:
        logger.warning(
            f"Failed to retrieve min and max primary key values for table {table_name} using standard connection.")
        return None, None


def get_min_max_primary_key(table_name, id_name=None, conn=None):
    """
    Retrieves the minimum and maximum primary key values for a given table.
    :param table_name: Name of the table to query.
    :param id_name: Name of the primary key column (optional).
    :param conn: Database connection object (optional, for Streamlit connections).
    :return: Tuple (min, max) of primary key values.
    """
    if id_name is None:
        id_name = retrieve_primary_key(table_name)

    if USES_STREAMLIT:
        return get_min_max_with_streamlit(table_name, id_name, conn)
    else:
        return get_min_max_with_standard_connection(table_name, id_name)


def get_timezone():
    timezone = "Asia/Singapore"
    column = "datetime"
    table_name = "weather"
    command = f"select {column} AT TIME ZONE {timezone} from {table_name};"
    execute_command(command)


def enclose_in_quotes(input_str):
    output_str = "'" + str(input_str) + "'"
    return output_str


def read_table_with_firebase(table_name, params, conn):
    """
    Reads data from a Firebase collection and converts it to a DataFrame.
    :param table_name: Name of the Firebase collection.
    :param params: Query parameters as a dictionary.
    :param conn: Firebase connection object.
    :return: DataFrame containing the queried data.
    """
    from google.cloud.firestore_v1 import FieldFilter

    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("Params must be a dictionary for Firebase.")

    if conn is None:
        raise ValueError("Connection must be provided for Firebase.")

    data = conn.collection(table_name)

    if "where" in params.keys():
        for clause in params["where"]:
            data = data.where(filter=FieldFilter(*clause))

    if "order_by" in params.keys():
        data = data.order_by(params["order_by"][0], direction=params["order_by"][1])

    if "limit" in params.keys():
        data = data.limit_to_last(params["limit"])

    data = data.get()
    df = collection_reference_to_dataframe(data, is_list=True)
    return df


def read_table_with_sql(table_name, params, conn, convert_to_text):
    """
    Reads data from a SQL table and converts it to a DataFrame.
    :param table_name: Name of the SQL table.
    :param params: Query parameters as a list of conditions.
    :param conn: Database connection object.
    :param convert_to_text: Whether to convert the SQL command to a SQLAlchemy text object.
    :return: DataFrame containing the queried data.
    """
    command = f"SELECT * FROM {table_name} "
    if params and len(params) > 0:
        where_clause = "WHERE "
        command = command + where_clause
        for vals in params:
            command = command + ' '.join(vals) + ' '

    logger.debug(f"Reading table with select: {command}")

    # Query using Streamlit connection
    if USES_STREAMLIT:
        df = query_with_streamlit(command, conn)

    # Query using standard SQLAlchemy connection
    else:
        if conn is None:
            conn = engine_connect()
        if convert_to_text:
            command = text(command)
        df = pd.read_sql(command, conn)

    return df


def read_table_with_select(table_name, params=None, conn=None, convert_to_text=True):
    """
    Reads a table using either Firebase or SQL, based on the connection type.
    :param table_name: Name of the table or Firebase collection.
    :param params: Query parameters as a dictionary (for Firebase) or list of conditions (for SQL).
    :param conn: Connection object (Firebase or SQL).
    :param convert_to_text: Whether to convert the SQL command to a SQLAlchemy text object.
    :return: DataFrame containing the queried data.
    """
    if USES_FIREBASE:
        return read_table_with_firebase(table_name, params, conn)
    else:
        return read_table_with_sql(table_name, params, conn, convert_to_text)


def get_camera_info_from_db(conn):
    return read_table_with_select(CAMERA_INFO_TABLE_NAME, conn=conn)


def get_target_camera_info(camera_id, conn):
    df_camera = get_camera_info_from_db(conn)
    return df_camera[df_camera[CAMERA_ID_KEY_NAME] == str(camera_id)]


def append_data_to_database(df, table_name, primary_keys, conn):
    """
    Handles appending data to the database, either to Firebase or PostgreSQL.
    :param df: DataFrame to append.
    :param table_name: Target table name.
    :param primary_keys: Primary key(s) for deduplication or indexing.
    :param conn: Database connection object.
    """
    if USES_FIREBASE:
        df.reset_index(inplace=True, drop=False)  # Ensure the index is included as a column
        row_dict = df.iloc[0].to_dict()
        insert_row_to_firebase(conn, row_dict, table_name, primary_keys)
    else:
        append_df_to_table(df, table_name, append_only_new=True, conn=conn, append_index=True)


def append_weather_data_to_database(weather_df, conn):
    append_data_to_database(weather_df, WEATHER_TABLE_NAME, DATETIME_KEY_NAME, conn)


def append_camera_location_data_to_database(location_df, conn):
    append_data_to_database(location_df, DASHCAM_TABLE_NAME, [DATETIME_KEY_NAME, CAMERA_ID_KEY_NAME], conn)


def append_vehicle_counts_data_to_database(vehicle_counts_df, conn):
    append_data_to_database(vehicle_counts_df, VEHICLE_COUNTS_TABLE_NAME, [DATETIME_KEY_NAME, CAMERA_ID_KEY_NAME], conn)


def append_image_analysis_data_to_database(target_datetime, camera_id, anomaly_label, weather_label, wetness_label,
                                           accident, congestion, flood, forecast_30min, forecast_5min, conn):
    row_dict = {
        DATETIME_KEY_NAME: target_datetime,
        CAMERA_ID_KEY_NAME: str(camera_id),
        ANOMALY_TYPE_KEY_NAME: ANOMALY_DICT.get(anomaly_label, 0),
        WEATHER_TYPE_KEY_NAME: WEATHER_DICT.get(weather_label, 0),
        WETNESS_TYPE_KEY_NAME: WETNESS_DICT.get(wetness_label, 0),
        "accident": bool(accident),
        "congestion": bool(congestion),
        "flood": bool(flood),
        "forecast_30min": int(forecast_30min),
        "forecast_5min": int(forecast_5min),
    }

    im_analysis_df = pd.DataFrame([row_dict]).set_index(DATETIME_KEY_NAME)
    append_data_to_database(im_analysis_df, IMAGE_ANALYSIS_TABLE_NAME, [DATETIME_KEY_NAME, CAMERA_ID_KEY_NAME], conn)


def read_vehicle_forecast_data_from_database(current_date, camera_id, history_length, conn):
    batch_size = 32

    def fetch_data(table_name, where_conditions):
        if USES_FIREBASE:
            params = {
                "where": where_conditions,
                "order_by": [DATETIME_KEY_NAME, firestore.Query.ASCENDING],
                "limit": batch_size,
            }
            return read_table_with_select(table_name, params, conn)
        else:
            params = [[DATETIME_KEY_NAME, "<=", enclose_in_quotes(current_date)]]
            fetch_top = f"\nORDER BY {DATETIME_KEY_NAME} DESC\nFETCH FIRST {batch_size} ROWS ONLY"
            params[-1].append(fetch_top)
            return read_table_with_select(table_name, params, conn=conn)

    # Fetch weather data
    df_weather = fetch_data(WEATHER_TABLE_NAME, [[DATETIME_KEY_NAME, "<=", current_date]])

    # Fetch vehicle counts data
    vehicle_conditions = [
        [DATETIME_KEY_NAME, "<=", current_date],
        [CAMERA_ID_KEY_NAME, "==", str(camera_id)] if USES_FIREBASE else [CAMERA_ID_KEY_NAME, "=",
                                                                          enclose_in_quotes(str(camera_id))]
    ]
    df_vehicles = fetch_data(VEHICLE_COUNTS_TABLE_NAME, vehicle_conditions)

    # Process and return data
    latest_weather_info = df_weather.iloc[0].copy()
    df_features = prepare_features_for_vehicle_counts(df_vehicles, df_weather, dropna=True,
                                                      include_weather_description=True)
    return df_features.iloc[-history_length:], latest_weather_info


if __name__ == "__main__":
    check_connection()

import logging

from libs.foxutils.utils import core_utils, dataframe_utils
from . import database_utils

from datetime import timedelta, datetime
# all datetimes are SG timezone

import numpy as np
import pandas as pd
import json
from os import listdir, getcwd
from os.path import join as pathjoin
from os.path import sep
import pytz
import csv

import matplotlib.pyplot as plt
from PIL import Image

unresponsive_delta = timedelta(minutes=10)
tz_SG = pytz.timezone('Asia/Singapore')
tz_GR = pytz.timezone('Europe/Athens')
tz_local = tz_GR
weather_folder = "openweathermap"
datamall_folder = "datamall"
logger = logging.getLogger("emia_utils.process_utils")



def minute_rounder(t):
    return t.replace(second=0, microsecond=0, minute=t.minute, hour=t.hour) + timedelta(minutes=t.second // 30)


def prepare_data(prefix, start_date, end_date):
    path = 'ltaodataservice/BusArrivalv2'
    dataset_dir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))

    target_files = listdir(dataset_dir)
    target_files = [i for i in target_files if (prefix in i)]

    df1 = pd.DataFrame(columns=['fetch_time', 'est_arrival_time', 'visit_number', 'load', 'origin_code',
                                'destination_code', 'latitude', 'longitude', 'feature', 'type'])

    for filename in target_files:
        with open(pathjoin(dataset_dir, filename), mode="r") as f:
            json_data = json.loads(f.read())
            str_date = filename.split('_')[2].split('.')[0]
            current_datetime = core_utils.convert_fully_connected_string_to_datetime(str_date)
            est_arrival_datetime = core_utils.convert_string_to_date(
                json_data[0]["NextBus"]["EstimatedArrival"]).replace(
                tzinfo=None)
            if current_datetime is not None and est_arrival_datetime is not None:
                if start_date < current_datetime <= end_date:
                    values = {'fetch_time': current_datetime,
                              'response_time': current_datetime,
                              'est_arrival_time': est_arrival_datetime,
                              'visit_number': json_data[0]["NextBus"]["VisitNumber"],
                              'load': json_data[0]["NextBus"]["Load"],
                              'origin_code': json_data[0]["NextBus"]["OriginCode"],
                              'destination_code': json_data[0]["NextBus"]["DestinationCode"],
                              'latitude': json_data[0]["NextBus"]["Latitude"],
                              'longitude': json_data[0]["NextBus"]["Longitude"],
                              'feature': json_data[0]["NextBus"]["Feature"],
                              'type': json_data[0]["NextBus"]["Type"],
                              }
                    df1.loc[len(df1.index)] = values

    if len(df1) > 0:
        diffs = np.diff(np.array(df1['est_arrival_time'].values))
        diffs = np.array([pd.Timedelta(x).total_seconds() for x in diffs])

        est_arrival_times = np.array([x.to_pydatetime() for x in df1['est_arrival_time']])
        fetch_times = np.array([x.to_pydatetime() for x in df1['fetch_time']])
        critical_point = [y > x for (x, y) in zip(est_arrival_times[:-1], fetch_times[1:])]
        critical_point.append(False)
        has_data_gap = [(x == True and (d2 - d1) > unresponsive_delta) for (x, d1, d2) in
                        zip(critical_point[:-1], fetch_times[:-1], fetch_times[1:])]
        bus_change = [((x == True and y == False) or (z)) for (x, y, z) in
                      zip(critical_point[:-1], critical_point[1:], has_data_gap[1:])]
        bus_change.append(False)

        df2 = pd.DataFrame({'fetch_time': fetch_times[1:], 'est_arrival_time': est_arrival_times[1:],
                            'diffs': diffs, 'critical_point': critical_point[:-1], 'bus_change': bus_change})

        return df1, df2

    else:
        return df1, pd.DataFrame(columns=['fetch_time', 'est_arrival_time', 'diffs', 'critical_point', 'bus_change'])


def reindex_at_regular_intervals(df, index_column='datetime', freq='1T', missing_vals_columns=None,
                                 interpolation_method='linear'):
    df[index_column] = [minute_rounder(x) for x in df[index_column]]
    df.set_index([index_column], inplace=True)
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq))
    df.rename_axis(index_column, inplace=True)
    for col in missing_vals_columns:
        df[col].interpolate(method=interpolation_method, inplace=True)
    return df


def edit_bus_arrivals(df):
    df_preprocessed = put_zero_bus_arrival(df)
    df_preprocessed.set_index(['fetch_time'], inplace=True)
    _, indexes = np.unique(df_preprocessed.index, return_index=True)
    df_preprocessed = df_preprocessed.iloc[indexes]

    df_preprocessed = df_preprocessed.reindex(pd.date_range(start=df_preprocessed.index.min(),
                                                            end=df_preprocessed.index.max(),
                                                            freq='1T'))
    df_preprocessed.rename_axis('fetch_time', inplace=True)
    df_preprocessed['diffs'].interpolate(method='linear', inplace=True)
    return df_preprocessed


def convert_local_to_target_timezone(local_time, target_timezone, response_timezone=tz_local):
    proper_time = response_timezone.localize(local_time)
    proper_time = proper_time.astimezone(target_timezone).replace(tzinfo=None)
    return proper_time


def read_csv_data_as_dataframe(start_date, end_date, target_path, target_columns=None, response_timezone=tz_local,
                               read_from_db=False):
    dataset_dir = pathjoin(core_utils.datasets_dir, target_path.replace('/', sep).replace('?', ''))

    target_files = listdir(dataset_dir)
    df1 = pd.DataFrame(columns=target_columns)

    for filename in target_files:
        with open(pathjoin(dataset_dir, filename), mode="r") as f:
            json_data = json.loads(f.read())
            str_date = filename.split('.')[0]
            current_datetime = core_utils.convert_fully_connected_string_to_datetime(str_date)
            if current_datetime is not None:
                if start_date < current_datetime <= end_date:
                    values = {'datetime': current_datetime, 'data': json_data}
                    df1.loc[len(df1.index)] = values

    return df1

def make_vehicle_counts_df(values):
    target_columns = { "datetime", "camera_id", "total_pedestrians", "total_vehicles", "bicycle",
                       "bus", "motorcycle", "person", "truck", "car" }
    keys_to_drop = [x for x in values.keys() if x not in target_columns]
    [values.pop(x) for x in keys_to_drop]
    [values.update({x: 0}) for x in target_columns if x not in values.keys()]

    proper_time = values["datetime"]
    proper_time.replace(tzinfo=None)
    values["datetime"] = proper_time
    return values


def make_weather_df(values):
    target_columns = ["datetime", "temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity",
                      "wind_speed", "wind_deg", "clouds_all", "visibility", "lat", "lon", "dt",
                      "timezone", "weather_id", "weather", "description"]
    measure_datetime = values["measured_datetime"]
    keys_to_drop = [x for x in values.keys() if x not in target_columns]
    [values.pop(x) for x in keys_to_drop]
    [values.update({x: None}) for x in target_columns if x not in values.keys()]
    # Convert time with timezone
    proper_time = core_utils.convert_string_to_date(measure_datetime)
    proper_time.replace(tzinfo=None)
    values["datetime"] = proper_time

    return values


def prepare_weather_data(start_date, end_date, response_timezone=tz_local, read_from_db=False):
    # Server response time is GR time, so convert to Singapore Time

    if read_from_db:
        params = []
        if start_date is not None:
            params.append(['datetime', '>=', database_utils.enclose_in_quotes(start_date)])
        if end_date is not None:
            params.append(['datetime', '<=', database_utils.enclose_in_quotes(end_date)])

        if len(params) > 1:
            params[0].append('AND')
        # print(params)
        df1 = database_utils.read_table_with_select('weather', params)

    else:
        dataset_dir = pathjoin(core_utils.datasets_dir, weather_folder.replace('/', sep).replace('?', ''))

        target_files = listdir(dataset_dir)
        df1 = pd.DataFrame()

        for filename in target_files:
            with open(pathjoin(dataset_dir, filename), mode="r") as f:
                json_data = json.loads(f.read())
                str_date = filename.split('.')[0]
                current_datetime = core_utils.convert_fully_connected_string_to_datetime(str_date)
                if current_datetime is not None:
                    if start_date < current_datetime <= end_date:
                        values = json_data
                        make_weather_df(values, response_timezone)
                        df1.loc[len(df1.index)] = values

        df1.drop_duplicates(inplace=True)

    return df1


def get_moving_average(yvals, window_size=5):
    windows = yvals.rolling(window_size)
    moving_averages = windows.mean()
    avgs = moving_averages.tolist()
    return avgs


def prepare_vals_and_avgs(df, window_size=5):
    df_preprocessed = edit_bus_arrivals(df)

    xvals = df_preprocessed.index
    yvals = df_preprocessed['diffs']
    avgs = get_moving_average(yvals, window_size=window_size)

    return xvals, yvals, avgs


def get_data(prefix, start_date, end_date, window_size=5):
    df_bus_arrival, df_bus_arrival_edit = prepare_data(prefix, start_date, end_date)
    fetch_time_series, diffs_series, moving_averages_final = prepare_vals_and_avgs(df_bus_arrival_edit, window_size)
    return df_bus_arrival, df_bus_arrival_edit, fetch_time_series, diffs_series, moving_averages_final


def get_bus_arrival_index(df):
    return df['bus_change'] == True


def ignore_bus_arrival(df):
    idx = df['bus_change'] == False
    df2 = df.copy()
    df2 = df2[idx]
    return df2


def put_zero_bus_arrival(df):
    idx = get_bus_arrival_index(df)
    df2 = df.copy()
    df2.loc[idx, 'diffs'] = 0
    return df2


def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def make_day_data_df(xvals, yvals, name):
    datetimes = [x.to_pydatetime().time() for x in xvals]
    df = pd.DataFrame(
        {'time_of_day': [datetime(1970, 1, 1, x.hour, x.minute, x.second) for x in datetimes], name: yvals})
    df.set_index('time_of_day', inplace=True)
    return df


def day_on_day_comparison(start_date, end_date, prefix="09111_105", window_size=5, elevate=100, with_weather=True):
    df_bus_arrival_1, df_bus_arrival_edit_1, fetch_time_series_1, diffs_series_1, moving_averages_final_1 = get_data(
        prefix, start_date, end_date, window_size)
    df_weather_1 = prepare_weather_data(start_date, end_date)

    start_date_2 = start_date + timedelta(days=1)
    end_date_2 = end_date + timedelta(days=1)
    df_bus_arrival_2, df_bus_arrival_edit_2, fetch_time_series_2, diffs_series_2, moving_averages_final_2 = get_data(
        prefix, start_date_2, end_date_2, window_size)
    df_weather_2 = prepare_weather_data(start_date_2, end_date_2)

    start_date_1_str = core_utils.convert_date_to_string(start_date)
    start_date_2_str = core_utils.convert_date_to_string(start_date_2)

    day_data_1 = make_day_data_df(fetch_time_series_1, diffs_series_1, start_date_1_str)
    day_data = day_data_1.merge(make_day_data_df(fetch_time_series_2[:-1], diffs_series_2[:-1], start_date_2_str),
                                how='outer', on='time_of_day')

    if with_weather:
        scaled = df_weather_1['humidity'] + elevate
        day_data = day_data.merge(
            make_day_data_df(df_weather_1['datetime'], scaled, ' '.join([start_date_1_str, 'Humidity (Scaled)'])),
            how='outer', on='time_of_day')

        scaled = (df_weather_1['temp'] - np.min(df_weather_1['temp'])) / np.ptp(df_weather_1['temp']) * 100 + elevate
        day_data = day_data.merge(
            make_day_data_df(df_weather_1['datetime'], scaled, ' '.join([start_date_1_str, 'Temperature (Scaled)'])),
            how='outer', on='time_of_day')

        scaled = df_weather_2['humidity'] + elevate
        day_data = day_data.merge(
            make_day_data_df(df_weather_2['datetime'], scaled, ' '.join([start_date_2_str, 'Humidity (Scaled)'])),
            how='outer', on='time_of_day')

        scaled = (df_weather_2['temp'] - np.min(df_weather_2['temp'])) / np.ptp(df_weather_2['temp']) * 100 + elevate
        day_data = day_data.merge(
            make_day_data_df(df_weather_2['datetime'], scaled, ' '.join([start_date_2_str, 'Temperature (Scaled)'])),
            how='outer', on='time_of_day')

        # scaled = (df_weather_2['feels_like'] - np.min(df_weather_2['feels_like'])) / np.ptp(
        #    df_weather_2['feels_like']) * 100 + elevate
        # day_data = day_data.merge(
        #    make_day_data_df(df_weather_2['datetime'], scaled, ' '.join([start_date_2_str, 'Feels-like (Scaled)'])),
        #    how='outer', on='time_of_day')

    day_data.sort_index(inplace=True)
    return day_data


def append_new_values_to_db(start_date, end_date, delete_tables=False, prefix="09111_105", window_size=5):
    df_bus_arrival, df_bus_arrival_edit, fetch_time_series, diffs_series, moving_averages_final = get_data(prefix,
                                                                                                           start_date,
                                                                                                           end_date,
                                                                                                           window_size)
    df_preprocessed = edit_bus_arrivals(df_bus_arrival_edit)
    df_weather = prepare_weather_data(start_date, end_date)

    if delete_tables:
        database_utils.drop_table('weather')
        database_utils.drop_table('bus_arrival')
        database_utils.drop_table('bus_arrival_diffs_raw')
        database_utils.drop_table('bus_arrival_diffs')
        print('\n')

    df_weather.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
    df_weather.set_index(['datetime'], inplace=True)
    database_utils.append_df_to_table(df_weather, 'weather', append_only_new=not delete_tables)

    df_bus_arrival.set_index(['fetch_time'], inplace=True)
    database_utils.append_df_to_table(df_bus_arrival, 'bus_arrival', append_only_new=not delete_tables)

    df_bus_arrival_edit.set_index(['fetch_time'], inplace=True)
    database_utils.append_df_to_table(df_bus_arrival_edit, 'bus_arrival_diffs_raw', append_only_new=not delete_tables)

    database_utils.append_df_to_table(df_preprocessed, 'bus_arrival_diffs', append_only_new=not delete_tables)


def add_all_values_to_db_clean():
    start_date = datetime(2023, 1, 2, 0)
    end_date = datetime.now()
    append_new_values_to_db(start_date, end_date, delete_tables=True)


def append_new_values_to_db_clean(target_table='bus_arrival'):
    _, start_date = database_utils.get_min_max_primary_key(target_table)
    end_date = convert_local_to_target_timezone(datetime.now(), tz_SG, tz_local)
    print(f'Appending new values from {start_date} until {end_date} (SG time)')
    append_new_values_to_db(start_date, end_date, delete_tables=False)


def read_classes_from_csv_file(filedir, target_file):
    class_dict = {}
    with open(pathjoin(filedir, target_file), 'r', newline='', encoding='utf-8') as csvfile:
        for line in csv.reader(csvfile):
            class_dict[line[0]] = int(line[1])
    return class_dict


vehicle_classes = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']
street_object_classes = ['traffic light', 'stop sign', 'clock']


def rearrange_class_dict(class_dict, target_classes=None):
    if target_classes is None:
        target_classes = vehicle_classes
    new_dict = {}
    [new_dict.update({x: class_dict[x]}) if x in class_dict.keys() else new_dict.update({x: 0}) for x in target_classes]

    vehicle_list = [new_dict[x] for x in new_dict.keys() if x in vehicle_classes]
    total_pedestrians = new_dict['person'] if 'person' in new_dict.keys() else 0
    total_vehicles = sum(vehicle_list) - total_pedestrians
    new_dict['total_pedestrians'] = total_pedestrians
    new_dict['total_vehicles'] = total_vehicles
    return new_dict


def prepare_features_for_vehicle_counts(df_vehicles, df_weather=None, dropna=True,
                                      include_weather_description=True ):
    index_column = "datetime"

    if df_weather is not None:
        dropped_side_cols = ["lat", "lon", "dt", "timezone", "temp_min", "temp_max"]
        df_weather.drop(columns=dropped_side_cols, inplace=True)

        weather_description_cols = ["weather", "description", "weather_id"]
        if include_weather_description:
            df_weather.dropna(inplace=True)
            weather_dict, weather_classes, df_weather = dataframe_utils.encode_categorical_values(df_weather, "weather")
            weather_description_dict, weather_description_classes, df_weather = dataframe_utils.encode_categorical_values(df_weather, 'description')
            df_weather_categ = df_weather[weather_description_cols + [index_column]]

        df_weather.drop(columns=weather_description_cols, inplace=True)

        join_method = "outer"
        use_interpolation = ["linear", None]
        df_vehicles = dataframe_utils.merge_data_frames([df_weather, df_vehicles], join_method,
                                                        use_interpolation=use_interpolation,
                                                        index_column=index_column,
                                                        dropna=dropna)
        if include_weather_description:
            df_vehicles.reset_index(inplace=True, drop=False)
            use_interpolation = ["nearest", None]
            df_vehicles = dataframe_utils.merge_data_frames([df_weather_categ, df_vehicles], join_method,
                                                            use_interpolation=use_interpolation,
                                                            index_column=index_column,
                                                            dropna=dropna)
    else:
        df_vehicles.set_index(index_column, inplace=True, drop=True)

    if len(df_vehicles) > 0:
        logger.debug(f"\nA total of {len(df_vehicles)} data points were fetched.")
        start_date = np.min(df_vehicles.index)
        end_date = np.max(df_vehicles.index)
        logger.debug(f"Fetched data corresponds to period with Start date = {start_date} and End date = {end_date}.")
        logger.debug(f"Feature Names:\n{df_vehicles.columns.tolist()}")

    return df_vehicles


def fetch_features_for_vehicle_counts(filedir, include_weather=False, explore_data=False,
                                      keep_all_detected_classes=False, dropna=True,
                                      include_weather_description=True, target_files=None):
    orig_filedir = filedir
    filedir = pathjoin(orig_filedir, 'labels')
    if target_files is None:
        target_files = [x for x in listdir(filedir) if '.csv' in x]

    logger.debug(f'Reading files from directory {filedir}.')

    if len(target_files) == 0:
        raise ValueError(f"No .csv label files found in directory {filedir}.")

    rows = [read_classes_from_csv_file(filedir, file) for file in target_files]
    dates = [core_utils.convert_fully_connected_string_to_datetime(file.split('_')[1].split('.')[0]) for file in
             target_files]

    detected_classes = np.unique(np.array(core_utils.flatten([list(row.keys()) for row in rows])))

    if explore_data:
        logger.info(f'Total detected classes are:\n {detected_classes}\n')
        wrong_val = 'fire hydrant'  # 'airplane'
        if wrong_val in detected_classes:
            logger.info(f"Incorrect predictions as '{wrong_val}':")
            wrong_pred_tuple = [(x, y, z) for (x, y, z) in zip(rows, dates, target_files) if wrong_val in x.keys()]
            for x in wrong_pred_tuple:
                wrong_filename = x[2].replace('.csv', '.jpg')
                logger.info(f"{x[1]} (File: {wrong_filename}): {x[0]}")
                plt.imshow(Image.open(pathjoin(orig_filedir, wrong_filename)))
                plt.show()

    if keep_all_detected_classes:
        new_rows = [rearrange_class_dict(row, detected_classes) for row in rows]
    else:
        new_rows = [rearrange_class_dict(row) for row in rows]

    df_vehicles = pd.DataFrame.from_dict(new_rows)
    index_column = "datetime"
    df_vehicles[index_column] = dates

    if explore_data and keep_all_detected_classes:
        false_classes = [x for x in detected_classes if x not in vehicle_classes if x not in street_object_classes]
        is_false = (df_vehicles['train'] > 0)
        for false_class in false_classes:
            is_false = is_false | (df_vehicles[false_class] > 0)
        logger.info(f"\nUnexpected detected classes are:\n {false_classes}")

        false_positives = sum(list(is_false)) / len(df_vehicles) * 100
        logger.info("False positives: ", "{:.2f}".format(false_positives), "%", " in a total of ", len(df_vehicles), " frames.")

    if include_weather:
        start_date = np.min(df_vehicles[index_column]) - timedelta(hours=3)
        end_date = np.max(df_vehicles[index_column])
        df_weather = prepare_weather_data(start_date, end_date, read_from_db=True)
    else:
        df_weather = None

    df_features = prepare_features_for_vehicle_counts(df_vehicles, df_weather, dropna, include_weather_description)
    return df_features


def distance(point1, point2, metric="euclidean"):
    x1, y1 = point1
    x2, y2 = point2
    if metric == "euclidean":
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    else:
        throw(ValueError("Invalid distance metric type."))


def group_points_by_distance(df, threshold=0.001, latitude_column="lat", longitude_column="lng"):
    groups = np.zeros(len(df))
    coordinates = [(x, y) for (x, y) in zip(df[latitude_column], df[longitude_column])]

    i = 0
    c = 1
    origin = coordinates[i]
    groups[i] = c
    i += 1
    while i < len(df):
        point = coordinates[i]
        d = distance(origin, point)
        if d < threshold:
            groups[i] = c
        else:
            c += 1
            groups[i] = c
            origin = coordinates[i]

        i += 1
    df["group"] = groups
    return df

def get_grouped_data(df, threshold=0.001):
    df = group_points_by_distance(df, threshold)
    total_groups = max(df["group"])
    logger.debug(f"Total groups: {int(total_groups)}")
    grouped_means = df.reset_index().drop(columns=["camera_id"]).groupby("group").mean()
    #grouped_means["group"] = grouped_means.index
    #print(grouped_means.head())
    #grouped_means.plot("datetime", ["total_vehicles", "total_pedestrians"], figsize=(20, 5))
    return grouped
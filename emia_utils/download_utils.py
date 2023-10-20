import json
import sched
import time
from datetime import datetime
from os import makedirs
from os.path import exists, sep
from os.path import join as pathjoin
from urllib.parse import urlparse

import httplib2 as http  # External library
import io
import pandas as pd
import pytz
import zipfile
from libs.foxutils.utils import core_utils
from requests import JSONDecodeError

######################################################################
# Setup
general_headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/116.0.1938.81",
    "Connection": "keep-alive",
}

headers = general_headers
headers.update({"AccountKey": core_utils.get_api_key("datamall.json")})


uri = 'http://datamall2.mytransport.sg/'
tz_SG = pytz.timezone('Asia/Singapore')

openweathermap_token_key = core_utils.get_api_key("openweathermap.json")
target_city = "SINGAPORE"
openweathermap_target_uri = "http://api.openweathermap.org/data/2.5/weather?"

weather_folder = "openweathermap"
mapfiles_folder = "mapfiles"
datamall_folder = "datamall"


def get_json_object_from_http_request(path):
    # Build query string & specify type of API call
    target = urlparse(uri + path)
    # print(target.geturl())
    method = 'GET'
    body = ''
    # Get handle to http
    h = http.Http()
    # Obtain results
    response, content = h.request(
        target.geturl(),
        method,
        body,
        headers)

    jsonObj = json.loads(content)
    return jsonObj


def run_scheduler_for_download(period_in_secs, function_handler):
    fetch_scheduler = sched.scheduler(time.time, time.sleep)
    fetch_scheduler.enter(period_in_secs, 1, function_handler, (fetch_scheduler,))
    fetch_scheduler.run()


def write_at_file_with_current_datetime_as_filename(data, current_time, filedir):
    if data is not None:
        filename = pathjoin(filedir, '_'.join([current_time.strftime("%Y%m%d%H%M%S")]) + '.json')
        json_string = json.dumps(data)
        with open(filename, "w") as jsonFile:
            jsonFile.write(json_string)


def fetch_data_from_value(path, page_size='10000'):
    target = urlparse(uri + path)
    success, req = core_utils.get_request(target.geturl(), headers=headers)
    if success:
        current_time = datetime.now(tz_SG)

        json_obj = req.json()
        if "value" in json_obj:
            data = json_obj["value"]
            filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))
            if not exists(filedir):
                makedirs(filedir)

            write_at_file_with_current_datetime_as_filename(data, current_time, filedir)
        else:
            print(f'Data fetching failed at time {datetime.now()} with response:')
            print(json_obj)

        return json_obj

    else:
        return []


def fetch_data_from_link_in_value(path, page_size='10000'):
    json_obj = get_json_object_from_http_request(path)

    if "value" in json_obj:
        if "Link" in json_obj["value"][0]:
            zip_file_url = json_obj["value"][0]["Link"]
            filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))
            if not exists(filedir):
                makedirs(filedir)

            print(f'Target url: {zip_file_url}')
            success, r = core_utils.get_request(zip_file_url)
            if success:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(filedir)
                print(f'Download and extract files in {filedir}')

    else:
        print(f'Data fetching failed at time {datetime.now()} with response:')
        print(json_obj)

    return json_obj


def fetch_df_from_value(path, filename, folder_suffix=None):
    json_obj = get_json_object_from_http_request(path)

    df = pd.DataFrame()
    if 'value' in json_obj:
        for x in json_obj['value']:
            if df.empty:
                df = pd.DataFrame([x])
            else:
                df = pd.concat([df, pd.DataFrame([x])], ignore_index=True)

    filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))
    if folder_suffix is not None:
        filedir = filedir + '_' + folder_suffix

    if not exists(filedir):
        makedirs(filedir)

    df.reset_index(inplace=True, drop=True)
    df.to_csv(pathjoin(filedir, filename + '.csv'))
    return df


######################################################################
# Dataset 1
def fetch_bus_arrival(path, bus_stop_code, service_no='', page_size='10000'):
    if service_no is None:
        params = {'BusStopCode': bus_stop_code}
        prefix = bus_stop_code
    else:
        params = {'BusStopCode': bus_stop_code, 'ServiceNo': service_no}
        prefix = '_'.join([bus_stop_code, service_no])

    # Build query string & specify type of API call
    target = urlparse(uri + path)
    success, req = core_utils.get_request(target.geturl(), params=params, headers=headers)
    if success:
        # print(req.url)
        current_time = datetime.now(tz_SG)

        try:
            json_obj = req.json()
            if "BusStopCode" in json_obj and "Services" in json_obj and json_obj["BusStopCode"] != "null":
                data = json_obj["Services"]
                if service_no != '':
                    data = [x for x in json_obj["Services"] if x["ServiceNo"] == service_no]

                filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))
                if not exists(filedir):
                    makedirs(filedir)
                if data:
                    filename = pathjoin(filedir, '_'.join([prefix, current_time.strftime("%Y%m%d%H%M%S")]) + '.json')
                    jsonString = json.dumps(data)
                    with open(filename, "w") as jsonFile:
                        jsonFile.write(jsonString)
            else:
                print(f'Data fetching for bus arrival failed at time {datetime.now()} with response:')
                print(json_obj)

            return json_obj

        except JSONDecodeError as je:
            print(je)
            return []

    else:
        return []


def download_dataset_1_bus_arrival(bus_stop_code='09111', service_no='105'):
    path = 'ltaodataservice/BusArrivalv2'
    fetch_bus_arrival(path, bus_stop_code, service_no)


def run_scheduler_dataset_1_bus_arrival(bus_stop_code='09111', service_no='105'):
    run_scheduler_for_download(60, download_dataset_1_bus_arrival(bus_stop_code, service_no))


######################################################################
# Dataset 7
def download_dataset_7_passeger_volume_od_train():
    path = 'ltaodataservice/PV/ODTrain?'
    fetch_data_from_link_in_value(path)


######################################################################
# Dataset 9
def download_dataset_9_taxi_availability():
    path = 'ltaodataservice/Taxi-Availability'
    fetch_data_from_value(path)


######################################################################
# Dataset 12
def download_dataset_12_car_park_availability():
    path = 'ltaodataservice/CarParkAvailabilityv2'
    fetch_data_from_value(path)


######################################################################
# Dataset 21

def fetch_traffic_images_from_link(path, page_size='10000', target_camera_id=None):
    json_obj = get_json_object_from_http_request(path)
    tz_SG = pytz.timezone('Asia/Singapore')
    current_time = datetime.now(tz_SG)

    if 'value' in json_obj:
        for x in json_obj['value']:
            camera_id = x['CameraID']
            if (target_camera_id is None) or (target_camera_id is not None and camera_id == target_camera_id):
                latitude = x['Latitude']
                longtitude = x['Longitude']
                img_url = x['ImageLink']

                folder = camera_id
                filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''),
                                   folder)
                if not exists(filedir):
                    makedirs(filedir)

                # print(f'Target image url: {img_url}')

                img_filename = img_url.split('?')[0].split('/')[-1]
                current_time_string = img_filename.split('_')[2]
                filename = pathjoin(filedir, '_'.join([camera_id, current_time_string]) + '.jpg')
                core_utils.save_image_from_link(img_url, filename)
    else:
        print(f'Data fetching for traffic images failed at time {datetime.now()} with response:')
        print(json_obj)

    return json_obj


def download_dataset_21_traffic_images():
    path = 'ltaodataservice/Traffic-Imagesv2'
    fetch_traffic_images_from_link(path)


def run_scheduler_dataset_21_traffic_images():
    run_scheduler_for_download(60 * 50, download_dataset_21_traffic_images)


def get_static_dataset_df(path, filename):
    filedir = pathjoin(core_utils.datasets_dir, datamall_folder, path.replace('/', sep).replace('?', ''))
    df = pd.read_csv(pathjoin(filedir, filename + '.csv'), index_col=0)
    return df


def download_dataset_3_bus_routes():
    path = 'ltaodataservice/BusRoutes'
    filename = 'bus_routes'
    df = fetch_df_from_value(path, filename)
    return df


def get_dataset_3_bus_routes():
    path = 'ltaodataservice/BusRoutes'
    filename = 'bus_routes'
    return get_static_dataset_df(path, filename)


def download_dataset_4_bus_stops_locations():
    path = 'ltaodataservice/BusStops'
    filename = 'bus_stops'
    df = fetch_df_from_value(path, filename)
    return df


def get_dataset_4_bus_stops_locations():
    path = 'ltaodataservice/BusStops'
    filename = 'bus_stops'
    return get_static_dataset_df(path, filename)


def download_dataset_10_taxi_stands():
    path = 'ltaodataservice/TaxiStands'
    filename = 'taxi_stands'
    df = fetch_df_from_value(path, filename)
    return df


def get_dataset_10_taxi_stands():
    path = 'ltaodataservice/TaxiStands'
    filename = 'taxi_stands'
    return get_static_dataset_df(path, filename)


def download_dataset_21_traffic_images_camera_locations():
    path = 'ltaodataservice/Traffic-Imagesv2'
    filename = 'camera_ids'
    folder_suffix = filename
    df = fetch_df_from_value(path, filename, folder_suffix)
    return df


def get_dataset_21_traffic_images_camera_locations():
    path = 'ltaodataservice/Traffic-Imagesv2' + '_camera_ids'
    filename = 'camera_ids'
    df = get_static_dataset_df(path, filename)
    df.drop(columns=['ImageLink'], inplace=True)
    return df


def get_mapfiles_filedir(filename='SGP_adm0.shp'):
    filedir = pathjoin(core_utils.datasets_dir, mapfiles_folder, filename)
    return filedir


######################################################################
# weather

def download_weather_info_from_openweather(city_name):
    url = f"{openweathermap_target_uri}q={city_name}&appid={openweathermap_token_key}"
    success, response = core_utils.get_request(url, headers=general_headers)
    if success:
        res = response.json()
        current_time = datetime.now(tz_SG)
        target_metrics = dict()

        if res["cod"] != "404":
            if 'main' in res:
                target_metrics = res["main"]
            if 'weather' in res:
                target_metrics.update({'weather_id': res["weather"][0]["id"],
                                       'weather': res["weather"][0]["main"],
                                       'description': res["weather"][0]["description"]})
            if 'wind' in res:
                target_metrics.update({'wind_speed': res["wind"]["speed"], 'wind_deg': res["wind"]["deg"]})
            if 'clouds' in res:
                target_metrics.update({'clouds_all': res["clouds"]["all"]})
            if 'visibility' in res:
                target_metrics.update({'visibility': res["visibility"]})
            if 'coord' in res:
                target_metrics.update({'lat': res["coord"]["lat"], 'lon': res["coord"]["lon"]})
            if 'dt' in res:
                measured_datetime = datetime.fromtimestamp(res["dt"], tz=tz_SG)
                target_metrics.update({'dt': res["dt"], 'timezone': res["timezone"],
                                       'measured_datetime': core_utils.convert_datetime_to_string(measured_datetime)})
                current_time = measured_datetime
            if 'weather' in res:
                if len(res["weather"]) > 0:
                    if 'description' in ["weather"][0]:
                        target_metrics.update({'description': res["weather"][0]["description"]})

            filedir = pathjoin(core_utils.datasets_dir, weather_folder.replace('/', sep).replace('?', ''))
            if not exists(filedir):
                makedirs(filedir)
            write_at_file_with_current_datetime_as_filename(target_metrics, current_time, filedir)

        else:
            print(f"Please enter a valid city name, other than {city_name}")

        return res
    else:
        return []

from os.path import join as pathjoin
from libs.foxutils.utils import core_utils

DEFAULT_IMAGE_FILE = "1703_20230913183132.jpg"
DEFAULT_CAMERA_ID = "1703"

CAMERA_INFO_TABLE_NAME = "camera_specs"
DASHCAM_TABLE_NAME = "dashcam"
VEHICLE_COUNTS_TABLE_NAME = "vehicle_counts"
WEATHER_TABLE_NAME = "weather"
BUS_ARRIVAL_TABLE_NAME = "bus_arrival"
IMAGE_ANALYSIS_TABLE_NAME = "image_analysis"

CAMERA_INFO_PATH = pathjoin("assets", "maps", "camera_ids.csv")

CAMERA_ID_KEY_NAME = "camera_id"
LATITUDE_KEY_NAME = "lat"
LONGITUDE_KEY_NAME = "lng"
DATETIME_KEY_NAME = "datetime"
ANOMALY_TYPE_KEY_NAME = "anomaly_type"
WEATHER_TYPE_KEY_NAME = "weather_type"
WETNESS_TYPE_KEY_NAME = "wetness_type"

CAMERA_TYPES = {0: "Expressway CCTV", 1: "Dashcam"}

WEATHER_DICT = {"Clear": 0, "Clouds": 1, "Rain": 2, "Thunderstorm": 3}
WEATHER_CLASSES = {v: k for k, v in WEATHER_DICT.items()}

WETNESS_DICT = {'Dry road surface': 0, 'Wet road surface': 1, 'Flooded road surface': 2}
WETNESS_CLASSES = {v: k for k, v in WETNESS_DICT.items()}

WEATHER_DESCRIPTION_DICT = {"broken clouds": 0, "clear sky": 1, "heavy intensity rain": 2,
                            "light intensity shower rain": 3, "light rain": 4, "moderate rain": 5,
                            "scattered clouds": 6, "thunderstorm": 7, "thunderstorm with heavy rain": 8,
                            "thunderstorm with light rain": 9, "thunderstorm with rain": 10}
WEATHER_DESCRIPTION_CLASSES = {v: k for k, v in WEATHER_DESCRIPTION_DICT.items()}


ANOMALY_DICT = {"Normal": 0, "Anomaly": 1}


DATASETS_DIR = core_utils.settings["DIRECTORY"]["datasets_dir"]
DEFAULT_DATASET_DIR = pathjoin(DATASETS_DIR, "test", DEFAULT_CAMERA_ID, "")
DEFAULT_FILEPATH = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)

RUNS_DIR = core_utils.settings["DIRECTORY"]["runs_dir"]
OBJECT_DETECTION_DIR = pathjoin(RUNS_DIR, "detect", "exp", "")
DEFAULT_VEHICLE_FORECAST_FEATURES_DF = pathjoin(DEFAULT_DATASET_DIR, "vf_feature_df.csv")

DEMO_INSTRUCTIONS = ("Click the start button to begin streaming from the selected camera. Click the stop button to end "
                     "the stream. Wait for a few seconds for the dashcam to disconnect, then press refresh. The "
                     "start button will then be active.")

dashcam_ids = [
               "6206af3f2ac0770155d598c1",
               "5b5649993e120205554b961c",
               "sgvideo1",
               ]

dashcam_imeis = ["357730090001398", "351609080169660", "sgvideo1"]

DASHCAM_IDS = {"Camera " + str(i+1): x for (i, x) in enumerate(dashcam_ids)}
DASHCAM_NAMES = {x: "test_dashcam_" + str(i+1) for (i, x) in enumerate(dashcam_ids)}
DASHCAM_IMEIS = {x: y for (x, y) in zip(dashcam_ids, dashcam_imeis)}

TRAFFIC_IMAGES_PATH = "ltaodataservice/Traffic-Imagesv2"
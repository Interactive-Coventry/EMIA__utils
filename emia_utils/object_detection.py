from .process_utils import *
from libs.foxutils.utils import core_utils
from libs.foxutils.tools.object_detection import detect_command
from os.path import join as pathjoin
from os.path import sep, exists
import shutil
from natsort import natsorted
from PIL import Image, ImageFile
import torch
import argparse
import csv
from numpy import random
from pathlib import Path
import os
import time
import torch.backends.cudnn as cudnn
import cv2

from libs.yolov7.models.experimental import attempt_load
from libs.yolov7.general import clean_str
from libs.yolov7.utils.datasets import LoadImages, LoadImages_2, letterbox
from libs.yolov7.utils.general import check_img_size, check_requirements, check_imshow, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from libs.yolov7.utils.plots import plot_one_box
from libs.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
ImageFile.LOAD_TRUNCATED_IMAGES = True

#os.system('git clone https://github.com/WongKinYiu/yolov7')
#os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt')
# os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt')


def read_classes_from_csv_file(filedir, target_file):
    class_dict = {}
    with open(pathjoin(filedir, target_file), 'r', newline='', encoding='utf-8') as csvfile:
        for line in csv.reader(csvfile):
            class_dict[line[0]] = int(line[1])
    return class_dict


def detect_from_directory(dataset_dir, return_results=False, keep_all_detected_classes=True, file_list=None, delete_previous=True):

    logger.info(f'Running on {core_utils.device}.')
    folder = pathjoin(core_utils.get_base_path(), core_utils.project_name, "runs", "detect")
    if delete_previous and exists(folder):
        shutil.rmtree(folder)

    if file_list is None:
        logger.info(f'Reading images from {dataset_dir}')
    with torch.no_grad():  # to avoid OOM
        _ = detect_command(image_source=dataset_dir, file_list=file_list)

    exp_folder = natsorted(listdir(folder))[-1]
    filedir = pathjoin(folder, exp_folder)
    logger.info(f"\nDetection results are saved at {filedir}\n")

    if return_results:
        target_files = [x for x in listdir(filedir) if '.csv' in x]

        if len(target_files) == 0:
            raise ValueError(f"No .csv label files found in directory {filedir}.")

        rows = [read_classes_from_csv_file(filedir, file) for file in target_files]
        dates = [core_utils.convert_fully_connected_string_to_datetime(file.split('_')[1].split('.')[0]) for file in
                 target_files]

        detected_classes = np.unique(np.array(core_utils.flatten([list(row.keys()) for row in rows])))

        if keep_all_detected_classes:
            new_rows = [rearrange_class_dict(row, detected_classes) for row in rows]
        else:
            new_rows = [rearrange_class_dict(row) for row in rows]

        class_df = pd.DataFrame.from_dict(new_rows)
        class_df['datetime'] = dates

        return class_df

    else:
        return filedir


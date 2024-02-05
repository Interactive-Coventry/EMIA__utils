from . import object_detection
from .process_utils import read_classes_from_csv_file, rearrange_class_dict
from libs.foxutils.utils import core_utils, torch_models, image_utils

from os.path import join as pathjoin
from os import listdir
import torch
import numpy as np
import pandas as pd
from natsort import natsorted


def read_images_from_filenames(target_filenames, dataset_dir, im_height=None, im_width=None):
    imgs = [image_utils.read_image_to_tensor(x, dataset_dir, im_height, im_width) for x in target_filenames]
    return imgs


def reconstruct_from_autoencoder(model, input_imgs):
    reconst_imgs, reconstruction_errors = torch_models.reconstruct_images(model, input_imgs)
    return reconst_imgs, reconstruction_errors


def embed_image(model, input_imgs):
    img_embeds = torch_models.embed_imgs(model, input_imgs)
    return img_embeds


def detect_objects(dataset_dir, keep_all_detected_classes=True, file_list=None):
    print(f'Reading images from {dataset_dir}')
    with torch.no_grad():  # to avoid OOM
        _ = object_detection.detect_command(image_source=dataset_dir, file_list=file_list)
    folder = pathjoin(core_utils.get_base_path(), core_utils.project_name, "runs", "detect")
    exp_folder = natsorted(listdir(folder))[-1]
    print(f"Detection results are saved at {exp_folder}")

    filedir = pathjoin(folder, exp_folder, 'labels')
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


def detect_vehicles(dataset_dir):
    return detect_objects(dataset_dir, keep_all_detected_classes=False)


def apply_transform(input_imgs, transform_function=None):
    if transform_function is not None:
        input_imgs = [transform_function(x) for x in input_imgs]
    return input_imgs

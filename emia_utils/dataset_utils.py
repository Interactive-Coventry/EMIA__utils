from libs.foxutils.utils import core_utils, train_functionalities, image_utils

from datetime import datetime
# all datetimes are SG timezone

from os import listdir
from os import stat
from os.path import join as pathjoin
from os.path import sep, isdir
from torch.utils.data import Dataset
import torch

IM_WIDTH = 640
IM_HEIGHT = 368


def get_express_way_dataset_dir(camera_id='', has_anomaly=False, folder_name=''):
    if has_anomaly:
        path = 'ltaodataservice/traffic_images'
        dataset_dir = pathjoin(core_utils.datasets_dir, 'datamall', path.replace('/', sep).replace('?', ''), camera_id,
                               folder_name)

    else:
        path = 'ltaodataservice/Traffic-Imagesv2'
        dataset_dir = pathjoin(core_utils.datasets_dir, 'datamall', path.replace('/', sep).replace('?', ''), camera_id)

    return dataset_dir


def get_datetime_from_filename(filename):
    image_datetime = core_utils.convert_fully_connected_string_to_datetime(filename.split('_')[1].split('.')[0])
    return image_datetime


def get_timestamp_from_filename(filename):
    image_timestamp = datetime.timestamp(get_datetime_from_filename(filename))
    return image_timestamp


def read_image_and_timestamp(filename, dataset_dir, im_height=None, im_width=None):
    image = image_utils.read_image_to_tensor(filename, dataset_dir, im_height, im_width)
    image_timestamp = get_timestamp_from_filename(filename)
    return image, image_timestamp


def check_image_file(filename, dataset_dir):
    if '.jpg' in filename:
        if stat(pathjoin(dataset_dir, filename)).st_size > 500:
            return True
    return False


def get_target_frame_filenames(dataset_dir, max_num=None):
    data_folders = [x for x in listdir(dataset_dir) if isdir(pathjoin(dataset_dir, x))]

    if len(data_folders) > 0:
        target_files = []
        for folder in data_folders:
            sub_dir = pathjoin(dataset_dir, folder)
            tf = [pathjoin(folder, i) for i in listdir(sub_dir) if check_image_file(i, sub_dir)]
            if max_num is not None:
                tf = tf[0:max_num]
            target_files = target_files + tf
    else:
        target_files = [i for i in listdir(dataset_dir) if check_image_file(i, dataset_dir)]

    return target_files


class TrafficImagesDataset(Dataset):
    def __init__(self, image_num, dataset_dir, im_height=IM_HEIGHT, im_width=IM_WIDTH, max_num=None):
        self.dataset_dir = dataset_dir
        target_files = get_target_frame_filenames(self.dataset_dir, max_num=max_num)
        # target_files = [x for x in target_files if get_datetime_from_filename(x) is not None]
        self.image_num = image_num
        if image_num is not None:
            target_files = target_files[0:image_num]

        self.target_files = target_files
        self.im_width = im_width
        self.im_height = im_height

    def __len__(self):
        # this should return the size of the dataset
        return len(self.target_files)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        filename = self.target_files[idx]
        image, image_timestamp = read_image_and_timestamp(filename, self.dataset_dir, self.im_height, self.im_width)
        return image, image_timestamp


def get_frames_from_all_cameras(batch_size, max_num=20, im_height=None, im_width=None, shuffle=True):
    dataset_dir = get_express_way_dataset_dir()
    dataset = TrafficImagesDataset(image_num=None, dataset_dir=dataset_dir, im_height=im_height,
                                   im_width=im_width, max_num=max_num)
    data_generator = train_functionalities.make_data_loader_with_torch(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                       show_size=True)

    return data_generator


def get_example_images_from_all_cameras(num, batch_size, im_height=None, im_width=None, shuffle=False):
    data_generator = get_frames_from_all_cameras(batch_size, 1, im_height, im_width, shuffle=shuffle)
    first_batch = next(iter(data_generator))
    return torch.stack([first_batch[0][i] for i in range(num)], dim=0)

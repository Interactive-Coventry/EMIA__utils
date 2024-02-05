from .process_utils import *
from libs.foxutils.utils import core_utils
from os.path import join as pathjoin
from os.path import sep, exists
import shutil
from natsort import natsorted
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

figsize = (20, 8)
seed = 42

import argparse

import csv
from numpy import random
from pathlib import Path
import os
import time
import torch.backends.cudnn as cudnn
import cv2

from libs.yolov7.models.experimental import attempt_load
from libs.yolov7.utils.datasets import LoadStreams, LoadImages, LoadImages_2, letterbox
from libs.yolov7.utils.general import check_img_size, check_requirements, check_imshow, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from libs.yolov7.utils.plots import plot_one_box
from libs.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#os.system('git clone https://github.com/WongKinYiu/yolov7')
#os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt')
# os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt')

def LoadModel(options, device, half, classify=False):
    logger.info(f"Load from {options.weights}")
    model = attempt_load(options.weights, map_location=device)
    stride = int(model.stride.max())
    _ = check_img_size(options.img_size, s=stride)
    if options.trace:
        model = TracedModel(model, device, options.img_size)
    if half:
        model.half()

    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()
    else:
        modelc = None

    return model, stride, modelc

################################################

class InParams:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def load_object_detection_model(save_img=True, save_txt=True, device="cuda"):
    classify = False
    if device == "cuda" or device == "gpu":
        device = select_device("0")
        half = device.type != "cpu"
    elif device == "cpu":
        half = False
    else:
        half = True

    opt = InParams(dict(agnostic_nms=False,
                        augment=False,
                        classes=None,
                        conf_thres=0.25,
                        device=device,
                        exist_ok=True,
                        img_size=640,
                        iou_thres=0.45,
                        name="exp",
                        nosave=False,
                        project="runs/detect",
                        save_conf=False,
                        save_txt=save_txt,
                        source="",
                        trace=False,
                        update=False,
                        view_img=False,
                        save_img=save_img,
                        save_dir="",
                        classify=classify,
                        half=half,
                        stride=0,
                        names=[],
                        colors=[],
                        weights=YOLO_MODEL + ".pt"))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    opt.save_dir = save_dir

    model, stride, modelc = LoadModel(opt, device, half, classify)
    opt.stride = stride
    logger.info(f"New object detection model loaded from Yolov7 on device {device}. Model type {type(model)}.\n")

    opt.names = model.module.names if hasattr(model, "module") else model.names
    opt.colors = [[random.randint(0, 255) for _ in range(3)] for _ in opt.names]
    if opt.half:
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(opt.device).type_as(next(model.parameters())))

    return model, opt


def detect_from_image(img, od_model, od_opt, device):
    names = od_opt.names
    colors = od_opt.colors

    # Padded resize
    od_img = letterbox(img, od_opt.img_size, od_opt.stride)[0]
    # Convert
    od_img = od_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    od_img = np.ascontiguousarray(od_img)
    od_img = torch.from_numpy(od_img).to(device)
    od_img = od_img.half() if od_opt.half else od_img.float()
    od_img /= 255.0
    if od_img.ndimension() == 3:
        od_img = od_img.unsqueeze(0)

    with torch.no_grad():
        #temporarily disable gradient calculation.
        #This is particularly useful when you're performing inference and can lead to faster and more memory-efficient computations.
        od_pred = od_model(od_img, augment=od_opt.augment)[0]
        od_pred = non_max_suppression(od_pred, od_opt.conf_thres, od_opt.iou_thres, classes=od_opt.classes,
                                      agnostic=od_opt.agnostic_nms)

    od_dict = {}

    for i, det in enumerate(od_pred):
        if len(det):
            det[:, :4] = scale_coords(od_img.shape[2:], det[:, :4], img.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                od_dict[names[int(c)]] = int(n)

            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=2)

    od_img = Image.fromarray(img[:, :, ::-1])

    return od_img, od_dict


def detect(model, opt, image_source=None, file_list=None):
    if file_list is None:
        logger.info(f"Reading images from {image_source}")

    opt.source = image_source
    names = opt.names
    colors = opt.colors
    webcam = opt.source.isnumeric() or opt.source.endswith(".txt") or opt.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://"))

    vid_path, vid_writer = None, None

    logger.info(f"Is webcam: {webcam}")
    if webcam:
        opt.view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size, stride=opt.stride)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size, stride=opt.stride, file_list=file_list)

    logger.info(f"Number of images to process: {len(dataset)}")

    im0 = None
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if img is None:
            print(f"None img for path {path}")

        if img is not None:
            img = torch.from_numpy(img).to(opt.device)
            img = img.half() if opt.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)
                save_path = str(opt.save_dir / p.name)
                txt_path = str(opt.save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}")  # img.txt
                s += "%gx%g " % img.shape[2:]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    class_dict = {}
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        class_dict[names[int(c)]] = int(n)

                    for *xyxy, conf, cls in reversed(det):
                        if opt.save_txt:
                            with open(txt_path + ".csv", "w", newline="", encoding="utf-8") as csvfile:
                                writer = csv.writer(csvfile)
                                for new_k, new_v in class_dict.items():
                                    writer.writerow([new_k, new_v])

                        if opt.save_img or opt.view_img:
                            label = f"{names[int(cls)]} {conf:.2f}"
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                if opt.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if opt.save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer.write(im0)

    if opt.save_txt or opt.save_img:
        s = f"\n{len(list(opt.save_dir.glob('labels/*.csv')))} labels saved to {opt.save_dir / 'labels'}" if opt.save_txt else ""
        print(s)

    logger.info(f"Done. ({time.time() - t0:.3f}s)")

    if im0 is not None:
        return Image.fromarray(im0[:, :, ::-1])
    else:
        return

###############################################
# Overloads
def detect_command(image_source=None, save_img=True, save_txt=True, file_list=None, device=None):
    od_model, opt = load_object_detection_model(save_img, save_txt, device)
    od_model.eval()

    return detect(od_model, opt, image_source, file_list)


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


def detect_from_video(model, opt, stream_link):
    opt.save_img = False
    with torch.no_grad():  # to avoid OOM
        _ = detect(model, opt, image_source=stream_link)
    return
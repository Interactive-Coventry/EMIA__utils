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
from models.experimental import attempt_load
import cv2
from yolov7.utils.datasets import LoadStreams, LoadImages, LoadImages_2
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

os.system('git clone https://github.com/WongKinYiu/yolov7')
os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt')


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


def Custom_detect(image_source=None, save_img=True, save_txt=True, file_list=None):
    model = 'yolov7'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=model + ".pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Temp_files/', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt, unknown = parser.parse_known_args()  # opt = parser.parse_args()
    opt.save_txt = save_txt
    opt.source = image_source
    # print(f'Unknown arguments:\n {unknown}')

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()

    classify = False
    device = select_device('0')
    model, stride, modelc = LoadModel(opt, device, classify)
    half = device.type != 'cpu'

    vid_path, vid_writer = None, None

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, file_list=file_list)

    print(f'Number of images to process: {len(dataset)}')

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if img is not None:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    class_dict = {}
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        class_dict[names[int(c)]] = int(n)

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            # with open(txt_path + '.txt', 'a') as f:
                            #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            with open(txt_path + '.csv', 'w', newline='', encoding='utf-8') as csvfile:
                                writer = csv.writer(csvfile)
                                for new_k, new_v in class_dict.items():
                                    writer.writerow([new_k, new_v])

                        if save_img or view_img:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if save_img:
                    if dataset.mode == 'image':
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
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.csv')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(s)

    print(f'Done. ({time.time() - t0:.3f}s)')

    return Image.fromarray(im0[:, :, ::-1])




def read_classes_from_csv_file(filedir, target_file):
    class_dict = {}
    with open(pathjoin(filedir, target_file), 'r', newline='', encoding='utf-8') as csvfile:
        for line in csv.reader(csvfile):
            class_dict[line[0]] = int(line[1])
    return class_dict


def apply_object_detection(dataset_dir, file_list=None, delete_previous=True):
    print(f'Running on {core_utils.device}.')
    folder = pathjoin(core_utils.get_base_path(), core_utils.project_name, "runs", "detect")
    if delete_previous and exists(folder):
        shutil.rmtree(folder)

    if file_list is None:
        print(f'Reading images from {dataset_dir}')
    with torch.no_grad():  # to avoid OOM
        _ = Custom_detect(image_source=dataset_dir, file_list=file_list)

    exp_folder = natsorted(listdir(folder))[-1]
    filedir = pathjoin(folder, exp_folder)
    print(f"\nDetection results are saved at {filedir}\n")

    return filedir

def detect_objects(dataset_dir, keep_all_detected_classes=True, file_list=None, delete_previous=True):

    filedir = apply_object_detection(dataset_dir, file_list, delete_previous)

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
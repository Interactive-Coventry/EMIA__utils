from libs.foxutils.utils import train_functionalities
from libs.foxutils.utils import core_utils

import plotly.express as px
import pandas as pd
from torchvision import transforms

from .display_utils import show_anomaly_detection_on_image



def get_anomaly_detection_model(save_name, train_data=None, model_function=None, reload_from_saved=True):
    if not reload_from_saved:
        print("[INFO] fitting model...")
        model = model_function(train_data)
        print("[INFO] Finished fitting model...")
        train_functionalities.pickle_model(model, save_name)

    else:
        print("[INFO] Loading model...")
        model = train_functionalities.unpickle_model(save_name)

    return model


def display_anomaly_detection_results(test_data, test_labels, model=None):
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(test_data.transpose())}
    encoded_sample['label'] = test_labels
    encoded_samples = pd.DataFrame(encoded_sample)
    if model is not None:
        predicted_labels = model.predict(test_data)
    else:
        predicted_labels = test_labels

    fig = px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1',
                     color=predicted_labels, opacity=0.7)  # encoded_samples.label.astype(str)
    fig.show()


from sklearn.metrics import mean_squared_error


def apply_anomaly_detection(target_imgs, encoded_imgs, model, key, num_select=None, show_wrong_only=False,
                            labels=None, anomaly_label=-1):
    if num_select is None:
        num_select = len(encoded_imgs)

    print(f'\nAnomaly prediction for images in {key} dataset\n')
    preds = model.predict(encoded_imgs)
    print(f'Predictions: {preds}')

    rmse = mean_squared_error(preds, labels)
    print(f'MSE is {rmse}')

    preds = preds[0:num_select]
    labels = labels[0:num_select]

    target_imgs = target_imgs[0:num_select]
    if show_wrong_only:
        print('Display misclassified items only')
        vals = [(img, pred) for (img, pred, label) in zip(target_imgs, preds, labels) if pred != label]
    else:
        vals = [(img, pred) for (img, pred, label) in zip(target_imgs, preds, labels)]

    [show_anomaly_detection_on_image(transforms.ToPILImage()(img), pred, anomaly_label) for (img, pred) in vals]

############################################################
# With anomalib

import collections.abc
import os
from os.path import join as pathjoin
import shutil
from os.path import exists

import torch
import yaml
from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import cv2
from anomalib.post_processing.post_process import (
    superimpose_anomaly_map,
)

torch.set_float32_matmul_precision('medium')

anomaly_detection_model = core_utils.settings['MODELS']['anomaly_detection_model']
infer_results = pathjoin(os.getcwd(), 'runs', anomaly_detection_model, 'anomaly')
heatmap_results = pathjoin(os.getcwd(), 'runs', anomaly_detection_model, 'anomaly_heatmap')


CONFIG_PATHS = core_utils.settings['DIRECTORY']['anomalib_models_dir']

MODEL_CONFIG_PAIRS = {
    'patchcore': f'{CONFIG_PATHS}/patchcore/config.yaml',
    'padim': f'{CONFIG_PATHS}/padim/config.yaml',
    'cflow': f'{CONFIG_PATHS}/cflow/config.yaml',
    'dfkde': f'{CONFIG_PATHS}/dfkde/config.yaml',
    'dfm': f'{CONFIG_PATHS}/dfm/config.yaml',
    'ganomaly': f'{CONFIG_PATHS}/ganomaly/config.yaml',
    'stfpm': f'{CONFIG_PATHS}/stfpm/config.yaml',
    'fastflow': f'{CONFIG_PATHS}/fastflow/config.yaml',
    'draem': f'{CONFIG_PATHS}/draem/config.yaml',
    'reverse_distillation': f'{CONFIG_PATHS}/reverse_distillation/config.yaml',
}

MODEL = 'reverse_distillation'
# print(open(os.path.join(MODEL_CONFIG_PAIRS[MODEL]), 'r').read())

dataset_dir = ''
normal_dir = ''
abnormal_dir = ''

batch_size = 4
im_height = 256
im_width = 256
new_update = {
    'dataset': {'path': dataset_dir, 'name': '1704', 'format': 'folder', 'category': '1704',
                'task': 'classification', 'image_size': im_height, 'train_batch_size': batch_size,
                'test_batch_size': batch_size, 'num_workers': 0,
                'normal_dir': normal_dir, 'abnormal_dir': abnormal_dir,
                'normal_test_dir': None, 'mask_dir': None, 'extensions': None},
    'metrics': {'image': ['F1Score', 'AUROC'], 'pixel': []},
    'project': {'path': './runs', 'seed': core_utils.SEED},
    'model': {'early_stopping': {'metric': 'train_loss', 'mode': 'min', 'patience': 3}},
    # `train_loss`, `train_loss_step`, `train_loss_epoch`, `image_F1Score`, `image_AUROC`
    'trainer': {'accelerator': 'gpu', 'devices': 1, 'max_epochs': 4, 'enable_model_summary': True,
                'num_sanity_val_steps': False, 'gradient_clip_val': 0.1, 'val_check_interval': 1.0, }
}


def update_yaml(old_yaml, new_yaml, new_update):
    # load yaml
    with open(old_yaml) as f:
        old = yaml.safe_load(f)

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    old = update(old, new_update)

    # save the updated / modified yaml file
    with open(new_yaml, 'w') as f:
        yaml.safe_dump(old, f, default_flow_style=False)


new_yaml_path = CONFIG_PATHS + '/' + MODEL + '_new.yaml'
# print(new_yaml_path)
update_yaml(MODEL_CONFIG_PAIRS[MODEL], new_yaml_path, new_update)

with open(new_yaml_path) as f:
    updated_config = yaml.safe_load(f)
# pprint.pprint(updated_config) # check if it's updated


if updated_config['project']['seed'] != 0:
    print(updated_config['project']['seed'])
    seed_everything(updated_config['project']['seed'])

# It will return the configurable parameters in DictConfig object.
config = get_configurable_parameters(
    model_name=updated_config['model']['name'],
    config_path=new_yaml_path
)


def infer(model_path, filepath, config_path=new_yaml_path):
    """Run inference."""

    folder = pathjoin(infer_results)
    if exists(folder):
        shutil.rmtree(folder)

    # args = get_args()
    config = get_configurable_parameters(config_path=config_path)

    # config.visualization.show_images = args.show
    config.visualization.mode = "simple"
    if infer_results:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = infer_results
    else:
        config.visualization.save_images = False

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    model = model.load_from_checkpoint(model_path, hparams=config)

    dataset = InferenceDataset(
        filepath, image_size=tuple(config.dataset.image_size),  # transform_config=transform_config
    )
    dataloader = DataLoader(dataset)
    results = trainer.predict(model=model, dataloaders=[dataloader])
    # dict_keys(['image', 'image_path', 'anomaly_maps', 'pred_scores', 'pred_labels', 'pred_masks', 'pred_boxes', 'box_scores', 'box_labels'])
    return results


def get_heatmap(image_filepath, results):
    """Generate heatmap overlay and segmentations, convert masks to images."""

    anomaly_map = results['anomaly_maps'].squeeze().numpy()
    new_dim = anomaly_map.shape
    img = cv2.imread(image_filepath)
    img_opencv = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    if anomaly_map is not None:
        heat_map = superimpose_anomaly_map(anomaly_map, img_opencv, normalize=False)
        return heat_map
    else:
        return None

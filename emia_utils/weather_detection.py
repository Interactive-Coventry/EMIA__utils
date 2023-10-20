import json
from os.path import join as pathjoin

import torch
import torch.nn as nn
from PIL import Image
from efficientnet_pytorch import EfficientNet
from libs.foxutils.utils import core_utils
from torchvision import transforms

weather_dict = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Thunderstorm': 3}
weather_classes = {v: k for k, v in weather_dict.items()}
device = core_utils.device

def prepare_image_for_model_input(filepath):
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ])

    img = Image.open(filepath)
    img = tfms(img).unsqueeze(0)

    return img


def predict_weather_class(filename, model=None, labels_map=None, model_name='efficientnet-b7'):
    if model_name == 'efficientnet-b7':
        # Load ImageNet class names
        labels_map_file = pathjoin(core_utils.models_dir, 'label_maps', 'ImageNet', 'labels_map.txt')
        labels_map = json.load(open(labels_map_file))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        if model is None:
            model = EfficientNet.from_pretrained('efficientnet-b7')

    num_show = min(5, len(labels_map))

    img = prepare_image_for_model_input(filename)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img.to(device))

    # Print predictions
    print(f'\nPredictions with {model_name}-----')
    for idx in torch.topk(outputs, k=num_show).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))

    idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()[0]
    return labels_map[idx], torch.softmax(outputs, dim=1)[0, idx].item() * 100


def apply(filepath, modeldir=None):
    model_name = 'resnet-18-v1_TL_pl'

    if modeldir is None:
        modeldir = pathjoin('EMIA', 'weather_from_image')
    model_filename = pathjoin(modeldir, model_name)
    model_b7 = EfficientNet.from_pretrained('efficientnet-b7')
    for param in model_b7.parameters():
        param.requires_grad = False

    num_ftrs = model_b7._fc.in_features
    model_b7._fc = nn.Linear(num_ftrs, len(weather_classes))
    model_b7 = model_b7.to(device)
    model = core_utils.load_trained_model(model_b7, model_filename)

    label, prob = predict_weather_class(filepath, model, weather_classes, model_name)
    return label, prob
import os
import sys

# To handle relative importing
module_paths = [
    os.path.abspath(os.path.join('../..')),
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

import argparse
import ast
import pickle
import time
import torch
import numpy as np
import copy
import yaml
from tqdm import tqdm
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from src.dataloaders.dataloader import ImgNetDataset
from torch.utils.data import DataLoader
from src.utils import Config
from torchvision.models import resnet50, ResNet50_Weights

# Command line arguments, only need to provide config path
argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, default='/root/CS570-Final-Project/config/relabel_base.yaml')
args = argparser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = Config(**config)

# Initialize the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Load the model from a checkpoint if specified
if config.load_from_checkpoint is not False:
    model.load_state_dict(torch.load(config.load_from_checkpoint))

# Move the model to the device
model = model.to(config.device)

# Initialize the data loaders
dataset = ImgNetDataset(json_path=config.dataset.json_dir, type='val')
# For now, we traverse image by image
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)

dataset_json = json.load(open(config.dataset.json_dir, 'r'))

def save_relabeled():
    print(f'Now saving the current labels to {config.relabeled_dataset_save_path}')
    json.dump(dataset_json, open(config.relabeled_dataset_save_path, 'w'))
    print(f'Saved the current labels to {config.relabeled_dataset_save_path}')


# traverse the images
for i, (x, y) in enumerate(dataloader):
    new_labels = func_call(x, y, model) # TODO: Replace this with the crop & relabeller (I provide y here too, in case we may need to refer to it)
    dataset_json['data'][i]['labels'] = new_labels

    if i % 10000 == 0:
        print(f'Processed {i} images')
        save_relabeled()

save_relabeled()

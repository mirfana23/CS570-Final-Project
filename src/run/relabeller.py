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
from src.dataloaders.dataloader import ImgNetDataset, public_tr
from torch.utils.data import DataLoader
from src.utils import Config
from torchvision.models import resnet50, ResNet50_Weights

from src.cropping.crop import Cropper
from src.cropping.classify_crop import ImageClassifier
from src.confidence import mc_dropout, mc_perturbation


def save_relabeled():
    print(f'Now saving the current labels to {config.relabeled_dataset_save_path}')
    json.dump(result_json, open(config.relabeled_dataset_save_path, 'w'))
    print(f'Saved the current labels to {config.relabeled_dataset_save_path}')

def relabel(x, y, model, confidence_measure="mc_dropout", post_proc_logits='average', threshold=0.0):
    '''
    Some parameters to consider:
        - threshold
        - number of regions to crop
        - size of regions to crop
    Also, keep in mind the time complexity
    '''
    # Instantiate the cropper and classifier
    cropper = Cropper(method= config.crop_method, num_crops=config.num_crops, crop_size_method = config.crop_size_method)
    classifier = ImageClassifier(model = model)
    n_classes = 1000

    # Crop the image
    cropped_images = cropper.crop(x) # n_cropped_images x 224 x 224 (already resized)
    y_pred = classifier.classify(cropped_images) # n_cropped_images x n_classes
    scores = None
    
    if confidence_measure == 'mc_dropout':
        cropped_images = torch.stack([public_tr(img/255.0) for img in cropped_images], dim=0)
        cropped_images = cropped_images.squeeze(dim=1).cuda()
        scores = mc_dropout(model, cropped_images, n_classes=n_classes, n_iter=100, rate=config.dropout_rate)
    elif confidence_measure == 'mc_perturbation':
        cropped_images = torch.stack(cropped_images, dim=0)
        cropped_images = cropped_images.type(torch.uint8)
        cropped_images = cropped_images.squeeze(dim=1).cuda()
        scores = mc_perturbation(model, cropped_images, n_classes=n_classes, transforms=public_tr)
    elif confidence_measure == 'naive':
        # Classify the cropped images
        y_pred = classifier.classify(cropped_images) # n_cropped_images x n_classes
    else:
        raise NotImplementedError('You should select confidence measure in [mc_dropout, mc_perturbation, naive]')

    if scores is not None:
        y_pred = torch.zeros(n_classes)
        if post_proc_logits == 'average':
            scores = scores.mean(dim=1)
            idx = torch.argmax(scores, dim=-1)
        elif post_proc_logits == 'majority': 
            preds = scores.argmax(dim=2)
            idx, scores = [], []
            for p in preds:
                counts = torch.bincount(p, minlength=1000)
                idx.append(torch.argmax(counts))
                scores.append(counts / torch.sum(counts))
            idx = torch.tensor(idx)
            scores = torch.stack(scores)
        else:
            raise NotImplementedError('You should select post_proc_logits in [average]')    
        idx_filter = torch.max(torch.softmax(scores, dim=-1), dim=-1)[0] > threshold
        idx = idx[idx_filter]
        y_pred[idx] = 1
        y_pred = y_pred.tolist()
    return y_pred

if __name__ == '__main__':
    # Command line arguments, only need to provide config path
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='/root/CS570-Final-Project/config/relabel_mc_perturbation.yaml')
    args = argparser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(**config)
    print(config)
    # Initialize the model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Load the model from a checkpoint if specified
    if config.load_from_checkpoint is not False:
        model.load_state_dict(torch.load(config.load_from_checkpoint))

    # Move the model to the device
    model = model.to(config.device)

    # Initialize the data loaders
    dataset = ImgNetDataset(json_path=config.dataset.json_dir, type='relabel')
    # For now, we traverse image by image
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.dataset.num_workers)

    dataset_json = json.load(open(config.dataset.json_dir, 'r'))
    dataset_json = dataset_json['data']

    result_json = {}
    result_json['data'] = dataset_json
    
    # traverse the images
    for i, (x, y, img_idx) in tqdm(enumerate(dataloader), total = min(len(dataloader), config.num_relabel)):
        
        if i == config.num_relabel:
            break

        new_labels = relabel(x, y, model, confidence_measure=config.confidence_measure, post_proc_logits=config.post_proc_logits, 
                            threshold=config.threshold) 
        new_labels = [idx for idx, val in enumerate(new_labels) if int(val) == 1]
        orig_label = torch.argmax(y).item()
        
        if orig_label not in new_labels:
            new_labels += [orig_label]
    
        result_json['data'][img_idx[0]]['labels'] = new_labels

        if i % 10000 == 0 and i > 0:
            print(f'Processed {i} images')
            save_relabeled()

    save_relabeled()

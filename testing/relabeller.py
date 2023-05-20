# To handle relative importing
import os
import sys
module_paths = [
    os.path.abspath(os.path.join('..')),
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

# import necessary files
from tqdm import tqdm
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
# dataloader
from src.dataloaders.dataloader import ImgNetDataset
from torch.utils.data import DataLoader
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pred_dataset_path', type=str, default='/root/CS570-Final-Project/datasets/imgnet1k_original/val.json', help='path to the prediction dataset')
parser.add_argument('--gt_dataset_path', type=str, default='/root/CS570-Final-Project/datasets/imgnet1k_ReaL/val.json', help='path to the ground truth dataset')
args = parser.parse_args()



# json file of our validation set
relabel_json = json.load(open(args.pred_dataset_path, 'r')) # original validation set
# json file of ReaL validation set
real_json = json.load(open(args.gt_dataset_path, 'r'))

# check how much the labels per sample align
labels_dict_relabel = {}
for sample in relabel_json['data']:
    labels_dict_relabel[sample['img_path']] = sample['labels']
labels_dict_real = {}
for sample in real_json['data']:
    labels_dict_real[sample['img_path']] = sample['labels']

avg_jaccard_sim = 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        continue
    jaccard_sim = len(labels_relabel.intersection(labels_real)) / len(labels_relabel.union(labels_real))
    avg_jaccard_sim += jaccard_sim
    cnt += 1
avg_jaccard_sim /= cnt
print(f'Average jaccard similarity: {avg_jaccard_sim}')

avg_prec, avg_recall = 0, 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        continue
    precision = len(labels_relabel.intersection(labels_real)) / len(labels_relabel)
    recall = len(labels_relabel.intersection(labels_real)) / len(labels_real)
    avg_prec += precision
    avg_recall += recall
    cnt += 1
avg_prec /= cnt
avg_recall /= cnt
print(f'Average precision: {avg_prec}, Average recall: {avg_recall}')

avg_f1 = 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        continue
    precision = len(labels_relabel.intersection(labels_real)) / len(labels_relabel)
    recall = len(labels_relabel.intersection(labels_real)) / len(labels_real)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    avg_f1 += f1
    cnt += 1
avg_f1 /= cnt
print(f'Average f1 score: {avg_f1}')


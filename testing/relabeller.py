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
parser.add_argument('--exclude_val', type=str, default=None, help='whether to exclude the validation set from the training set')
args = parser.parse_args()

def safe_division(x, y):
    try:
        return x / y
    except:
        return 0

# json file of our validation set
relabel_json = json.load(open(args.pred_dataset_path, 'r')) # original validation set
# json file of ReaL validation set
real_json = json.load(open(args.gt_dataset_path, 'r'))

if args.exclude_val is not None:
    
    exclude_val = json.load(open(args.exclude_val, 'r'))
    exclude_label_map = {}
    for exclude_sample in exclude_val['data']:
        exclude_label_map[exclude_sample['img_path']] = exclude_sample['labels'][0]
    
    for sample in relabel_json['data']:
        if sample['img_path'] in exclude_label_map:
            rem_label = exclude_label_map[sample['img_path']]
            if rem_label in sample['labels']:
                sample['labels'].remove(rem_label)
    
    for sample in real_json['data']:
        if sample['img_path'] in exclude_label_map:
            rem_label = exclude_label_map[sample['img_path']]
            if rem_label in sample['labels']:
                sample['labels'].remove(rem_label)

# check how much the labels per sample align
labels_dict_relabel = {}
for sample in relabel_json['data']:
    labels_dict_relabel[sample['img_path']] = sample['labels']
labels_dict_real = {}
for sample in real_json['data']:
    labels_dict_real[sample['img_path']] = sample['labels']

# NOTE: Turn this off later
# path_old_label_mapping = '/root/CS570-Final-Project/datasets/map_clsloc.txt'
# path_cur_label_mapping = '/root/CS570-Final-Project/datasets/imagenet_class_index.json'
# old_label_mapping = {}
# with open(path_old_label_mapping, 'r') as f:
#     old_label_info = f.readlines()
#     for line in old_label_info:
#         label_id, label_idx, _ = line.split()
#         old_label_mapping[label_id] = int(label_idx)
# label_mapping = {}
# with open(path_cur_label_mapping, 'r') as f:
#     cur_label_mapping = json.load(f)
#     for label_idx, label_info in cur_label_mapping.items():
#         label_mapping[old_label_mapping[label_info[0]]] = int(label_idx)

# # transform labels to the same format
# for key in labels_dict_relabel.keys():
#     labels_dict_relabel[key] = [label_mapping[label] for label in labels_dict_relabel[key]]

avg_jaccard_sim = 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        jaccard_sim = 1 if len(labels_relabel) == 0 else 0    
    else:
        jaccard_sim = safe_division(len(labels_relabel.intersection(labels_real)), len(labels_relabel.union(labels_real)))
    avg_jaccard_sim += jaccard_sim
    cnt += 1
avg_jaccard_sim = safe_division(avg_jaccard_sim, cnt)

avg_prec, avg_recall = 0, 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        precision = 1 if len(labels_relabel) == 0 else 0
        recall = 1 if len(labels_relabel) == 0 else 0
    else:
        precision = safe_division(len(labels_relabel.intersection(labels_real)), len(labels_relabel))
        recall = safe_division(len(labels_relabel.intersection(labels_real)), len(labels_real))
    avg_prec += precision
    avg_recall += recall
    cnt += 1
avg_prec = safe_division(avg_prec, cnt)
avg_recall = safe_division(avg_recall, cnt)

avg_f1 = 0
cnt = 0
for key in labels_dict_relabel.keys():
    labels_relabel = set(labels_dict_relabel[key])
    labels_real = set(labels_dict_real[key])
    if len(labels_real) == 0:
        precision = 1 if len(labels_relabel) == 0 else 0
        recall = 1 if len(labels_relabel) == 0 else 0
    else:
        precision = safe_division(len(labels_relabel.intersection(labels_real)), len(labels_relabel))
        recall = safe_division(len(labels_relabel.intersection(labels_real)), len(labels_real))
    try:
        f1 = safe_division(2 * precision * recall, (precision + recall))
    except ZeroDivisionError:
        f1 = 0
    avg_f1 += f1
    cnt += 1
avg_f1 = safe_division(avg_f1, cnt)

print(f'Average jaccard similarity: {avg_jaccard_sim:.2f}, Average precision: {avg_prec:.2f}, Average recall: {avg_recall:.2f}, Average f1 score: {avg_f1:.2f}')